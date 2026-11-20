"""E101 observer states: DeepSeek-R1-Distill-Llama-8B reading what Qwen3.5-9B saw, along the 9B's InjecForks reasoning.
Mirrors injecforks_states.py (E79 v2) so injecforks_probe.py trains on these states unchanged. Plan and formatting decisions:
docs/plan-2026-10-30-distill-observer.md; rendering: observer_render.py.

  ~/.venvs/e01/bin/python injecforks_observer_states.py configs/e101-distill-observer-pod.yaml prep    # local CPU: every item, every assertion
  ~/.venvs/e01/bin/python injecforks_observer_states.py configs/e101-distill-observer-pod.yaml smoke   # local GPU: a 1B Llama, 2 cases, cache check
  python injecforks_observer_states.py configs/e101-distill-observer-pod.yaml run                      # pod: every selected row
  ~/.venvs/e01/bin/python injecforks_observer_states.py configs/e101-distill-observer-pod.yaml fetch --prefix RUN_ID

Rows come only from injecforks_load.load (rollouts + labels-v1.2 + cuts, every join asserted) at the config's Hub revision.
Per case, both the 9B's own render (injecforks_states.render, asserted equal to the billed prompt tokens) and the observer's
are built, and every tool JSON line the observer reads is asserted to be in the 9B's prompt. Per row the 9B's tokenizer must
put its N-th token's end at cut[N] (as E79 v2 asserted), and the observer's prefix ids must decode to reasoning[:cut[N]].

The prefix sits inside the observer's own <think> as the current turn's reasoning (observer_render.py).
Reads (common.LayerTap, each layer's OUTPUT residual, 1-indexed): N = 0 at the context's last token (one per case), N in ncut
at the last token of reasoning[:cut[N]]. Per case the context runs once (use_cache); each rollout extends it and is cropped
back. `smoke`, and the first 3 cases of `run`, assert the cached path equals one full forward at every read (cosine).
Each rollout's mean NLL of its longest prefix under the observer is stored.
Output: data/e101/states/states-NNN.npy (points x layers x d, float16) + meta-NNN.jsonl, injecforks_states.py's layout.
"""
import argparse, json, os, re, shutil, sys, time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_load import load                                   # noqa: E402
from injecforks_states import REPO, sha, verify, render as actor_render   # noqa: E402
import observer_render as R                                         # noqa: E402

OUT = "e101"


def snapshot(job):
    from huggingface_hub import snapshot_download
    pats = []
    for cfg, splits in job["select"].items():
        for sp in splits:
            pats += [f"{d}{cfg}/{sp}.jsonl" for d in ("", "prompts/", "cuts/", "eval-v1.2/", "labels-v1.2/")]
    return Path(snapshot_download(REPO, repo_type="dataset", revision=job["dataset_revision"], allow_patterns=pats + ["SHA256SUMS*"]))


def tokenizers(job, gen):
    from transformers import AutoTokenizer
    a = job["actor_tokenizer"]
    return (AutoTokenizer.from_pretrained(gen.get("tokenizer_id", gen["model_id"]), revision=gen.get("tokenizer_revision", gen["revision"])),
            AutoTokenizer.from_pretrained(a["model_id"], revision=a["revision"]))


def build_items(job, root, tok, atok, limit_cases=None):
    """[(case_key, context ids, [(row_id, [prefix chain ids], {N: (chain, position)})])], asserting every join."""
    ncut = job["ncut"]; maxn = max(ncut)
    out = []
    for cfg, splits in job["select"].items():
        for sp, which in splits.items():
            rows = load(cfg, sp, subset=None if which == "all" else "eval", root=root)
            by = {}
            for r in rows:
                by.setdefault(r["case"], []).append(r)
            want = set(sorted(by)[:limit_cases] if limit_cases else by)
            with open(root / "prompts" / cfg / f"{sp}.jsonl") as f:
                bodies = {}
                for l in f:
                    case = re.match(r'\{"case": "([^"]+)"', l).group(1)
                    if case in want:
                        bodies[case] = json.loads(l)
            assert set(bodies) == want, ("prompts missing", cfg, sp, len(want - set(bodies)))
            for case in sorted(want):
                b = bodies[case]
                actor = atok.decode(a_ids := actor_render(atok, b))
                assert len(a_ids) == b["prompt_tokens"], ("not the prompt the 9B was billed for", cfg, sp, case)
                assert all(t in actor for t in R.tool_lines(b)), ("tool JSON differs from the 9B's", cfg, sp, case)
                _, ctx = R.context_ids(tok, b)
                rr = []
                for r in sorted(by[case], key=lambda r: r["rollout"]):
                    enc = atok(r["reasoning"], add_special_tokens=False, return_offsets_mapping=True)
                    for n in ncut:   # the cut is the 9B's N-th token end, exactly as E79 v2 asserted
                        c = r["cut"].get(n)
                        assert (c is None) == (len(enc.input_ids) < n), ("cut null-ness", r["row_id"], n)
                        assert c is None or enc.offset_mapping[n - 1][1] == c, ("cut offset", r["row_id"], n)
                    chains, reads = R.prefix_chains(tok, r["reasoning"], r["cut"], ncut)
                    assert set(reads) == {n for n in ncut if r["cut"].get(n) is not None}
                    assert all(len(ch) <= 4 * maxn for ch in chains), ("prefix far longer than expected", r["row_id"])
                    rr.append((r["row_id"], chains, reads))
                out.append((f"{cfg}|{sp}|{case}", ctx, rr))
    return out


def prep(job):
    tok, atok = tokenizers(job, job["generator"])
    root = snapshot(job); print("verified files:", verify(root))
    items = build_items(job, root, tok, atok)
    C = sorted(len(c) for _, c, _ in items)
    rows = [x for _, _, rr in items for x in rr]
    chains = sum(len(x[1]) for x in rows); multi = sum(len(x[1]) > 1 for x in rows)
    toks = sum(len(c) for _, c, _ in items) + sum(len(ch) for x in rows for ch in x[1])
    reads = len(items) + sum(len(x[2]) for x in rows)
    gb = reads * len(job["layers"]) * job["d_model"] * 2 / 1e9
    print(dict(cases=len(items), rows=len(rows), context_tokens_p50_max=[C[len(C) // 2], C[-1]],
               prefix_forwards=chains, rows_needing_2plus_forwards=multi, tokens_forwarded=toks, state_reads=reads,
               states_gb=round(gb, 1)))
    print("by config/split:", {k: sum(1 for c, _, _ in items if c.startswith(k + "|")) for k in
                               {f"{cfg}|{sp}" for cfg, s in job["select"].items() for sp in s}})


def run(job, smoke=False):
    import torch
    from common import load_generator, LayerTap
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    api = None
    if smoke:   # the small Llama over 2 cases, no Hub upload; the observer's tokenizer so every rendering assertion holds
        run_id = "smoke"
        job = dict(job, generator=dict(job["smoke_generator"]), layers=job["smoke_layers"], d_model=job["smoke_d_model"])
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        if not api.repo_info(job["artifact_repo"]).private:
            raise SystemExit("artifact repo must be private")
        for rel, expected in job["config_sha256"].items():
            if sha(HERE / rel) != expected:
                raise SystemExit(f"configuration drift since launch: {rel}")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = snapshot(job); nver = verify(root)
    out = job.get("out_name", OUT)   # out_name: a run beside E101's
    base = HERE / "data" / out / ("smoke" if smoke else "")
    D = base / "states"; D.mkdir(parents=True, exist_ok=True)
    RES = HERE / f"results/{out}-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"])
    if smoke and job["generator"].get("smoke_cast") == "float16":   # sm75 cards have no bf16 memory-efficient attention
        model = model.half()
    _, atok = tokenizers(job, job["generator"])
    lm, lm_head = model.model, model.lm_head
    items = build_items(job, root, tok, atok, limit_cases=job.get("smoke_cases") if smoke else None)
    if smoke:   # one smoke case and one realistic case, 3 rollouts each
        items = [(c, x, rr[:3]) for c, x, rr in [next(i for i in items if i[0].startswith(p)) for p in ("smoke|", "realistic|")]]
    keep = job["layers"]
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, started=time.time(), layers=keep, ncut=job["ncut"],
                   model=job["generator"]["model_id"], revision=job["generator"]["revision"], template_fix=dict(add=R.CALL_CLOSE, source=list(R.TEMPLATE_FIX_SOURCE)),
                   dataset_revision=job["dataset_revision"], verified_files=nver, cases=len(items), shards=[], cache_check=[])
    buf_X, buf_M, shard, last = [], [], 0, time.time()

    def flush(final=False):
        nonlocal buf_X, buf_M, shard
        if buf_X:
            np.save(D / f"states-{shard:03d}.npy", np.concatenate(buf_X).astype(np.float16))
            with open(D / f"meta-{shard:03d}.jsonl", "w") as f:
                f.writelines(json.dumps(m) + "\n" for m in buf_M)
            receipt["shards"].append(dict(shard=shard, points=len(buf_M))); shard += 1; buf_X, buf_M = [], []
        receipt.update(uploaded=time.time(), final=final, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=D, path_in_repo=f"{run_id}/states")
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"{run_id}/results")

    def forward(ids, pos, past=None, start=0):
        """Forward ids (1 x L); returns ({layer: [len(pos), d]} at positions pos, last_hidden_state, cache)."""
        got = {}
        tap = LayerTap(layers, keep, lambda li, h: got.__setitem__(li, h[0, pos].float().cpu()))
        with torch.no_grad():
            o = lm(input_ids=torch.tensor([ids], device="cuda"), past_key_values=past, use_cache=True,
                   position_ids=torch.arange(start, start + len(ids), device="cuda")[None])
        tap.close()
        return got, o.last_hidden_state[0], o.past_key_values

    stack = lambda g: np.stack([g[li].numpy() for li in keep], 1)
    t0 = time.time()
    try:
        for ci, (case, ctx, rr) in enumerate(items):
            P = len(ctx)
            got, hs, cache = forward(ctx, [P - 1])
            first_logit = lm_head(hs[-1:]).float()                       # predicts the first prefix token
            c0 = stack(got); full_c0 = None
            buf_X.append(c0); buf_M.append(dict(case=case, N=0, kind="c0"))
            for row_id, chains, reads in rr:
                sx_all, pos_all, ns_all = [], [], []
                for k, ch in enumerate(chains):
                    ns = [n for n in job["ncut"] if n in reads and reads[n][0] == k]
                    pos = [reads[n][1] for n in ns]
                    g2, h2, cache = forward(ch, pos, past=cache, start=P)
                    cache.crop(-len(ch)); assert cache.get_seq_length() == P
                    if k == 0:   # chain 0 is the longest prefix
                        logits = torch.cat([first_logit, lm_head(h2[:-1]).float()])
                        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(ch, device=logits.device)[:, None])[:, 0]
                        nll = [round(float(-lp.mean()), 4), round(float(-lp[:8].mean()), 4)]
                    sx_all.append(stack(g2)); pos_all.append((ch, pos)); ns_all += ns
                if not ns_all:
                    continue
                buf_X.append(np.concatenate(sx_all))
                buf_M.extend([dict(row_id=row_id, N=n, kind="prefix", nll_mean_first8=nll, forwards=len(chains)) for n in ns_all])
                if smoke or ci < 3:   # the cached path must equal one full forward at every read, per layer
                    if full_c0 is None:
                        gf, _, _ = forward(ctx, [P - 1]); full_c0 = stack(gf)
                    full, cached = [full_c0], [c0]
                    for (ch, pos), sx in zip(pos_all, sx_all):
                        gf, _, _ = forward(ctx + ch, [P + p for p in pos]); full.append(stack(gf)); cached.append(sx)
                    full, cached = np.concatenate(full), np.concatenate(cached)
                    cos = float(np.min((full * cached).sum(-1) / np.linalg.norm(full, axis=-1) / np.linalg.norm(cached, axis=-1)))
                    receipt["cache_check"].append(dict(row_id=row_id, min_cosine=round(cos, 5), nll=nll, forwards=len(chains)))
                    assert cos > job.get("min_cache_cosine", 0.995), ("cached path differs from the full forward", row_id, cos)
            del cache
            torch.cuda.empty_cache()
            if ci % 20 == 0:
                print(ci, len(items), f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["forward_seconds"] = round(time.time() - t0, 1); receipt["complete"] = True
    finally:
        receipt["finished"] = time.time(); flush(final=True)
    print("done", json.dumps({k: receipt[k] for k in ("cases", "forward_seconds", "peak_gib", "cache_check") if k in receipt})[:1500])


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data" / OUT / "dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"{prefix}/*"], local_dir=st)
    D = HERE / "data" / OUT / "states"; D.mkdir(parents=True, exist_ok=True)
    for p in (st / prefix / "states").glob("*"):
        shutil.copy2(p, D / p.name)
    if (st / prefix / "results").exists():
        shutil.copytree(st / prefix / "results", HERE / f"results/{OUT}-cloud", dirs_exist_ok=True)
    print("fetched", prefix, len(list(D.glob("*.npy"))), "shards")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["prep", "smoke", "run", "fetch"]); ap.add_argument("--prefix")
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text()); os.chdir(HERE)
    {"prep": lambda: prep(job), "smoke": lambda: run(job, smoke=True), "run": lambda: run(job), "fetch": lambda: fetch(job, a.prefix)}[a.stage]()
