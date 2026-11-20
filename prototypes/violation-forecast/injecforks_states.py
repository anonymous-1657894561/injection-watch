"""E79 v2: Qwen3.5-9B residual states along InjecForks v1.2's native reasoning (experiments/E79-within-prompt-prefix-probes.md).

  ~/.venvs/e01/bin/python injecforks_states.py configs/e79v2-9b-pod.yaml prep    # local, CPU: build every item, run every assertion
  ~/.venvs/e01/bin/python injecforks_states.py configs/e79v2-9b-pod.yaml smoke   # local GPU: Qwen3.5-0.8B on 2 cases, cache check
  python injecforks_states.py configs/e79v2-9b-pod.yaml run                      # pod: every selected row
  ~/.venvs/e01/bin/python injecforks_states.py configs/e79v2-08b-local.yaml local  # 0.8B over a local config: code checks only
  ~/.venvs/e01/bin/python injecforks_states.py configs/e79v2-9b-pod.yaml fetch --prefix RUN_ID
  python injecforks_states.py configs/e101-9b-testcache.yaml cachecheck              # pod (E101): the cache check alone, first max_cases cases
  ~/.venvs/e01/bin/python injecforks_states.py configs/e101-9b-testcache.yaml cachecheck-local   # the same with the 0.8B, local GPU

Input: InjecForks at the config's Hub revision (the run downloads it; nothing is uploaded first). Each case's prompt is rendered
from its exact OpenRouter request body (prompts/<config>/<split>.jsonl) the way e77_variants.local_count does, and its token count
is asserted equal to the prompt_tokens DeepInfra billed (the file's `prompt_tokens`). The served reasoning is tokenized alone
(add_special_tokens=False), as the cuts file was: the N-th token's end offset is asserted equal to cuts[N].

Reads (each layer's OUTPUT residual, 1-indexed, common.LayerTap): N = 0 at the last prompt token (one per case: identical for every
rollout of the case by construction), and N in ncut at the N-th reasoning token where the reasoning has >= N tokens. Only the first
max(ncut) reasoning tokens are ever fed. Per case the prompt runs once (use_cache); each rollout then continues a deep copy of that
cache with its reasoning tokens. `smoke` asserts the cached path equals a full-sequence forward at every read (cosine).
Each rollout's mean NLL of its reasoning tokens under the model is stored: far above the sampling entropy = not its own text.

Rows (config `select`): per config and split, `all` rows or the `eval` subset (eval-v1.2), injected and benign twins alike.
Output: data/e79v2/states/states-NNN.npy (points x layers x d, float16) + meta-NNN.jsonl (row_id or case, N, kind, nll).
"""
import argparse, copy, hashlib, json, os, re, shutil, sys, time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sha = lambda p: hashlib.file_digest(open(p, "rb"), "sha256").hexdigest()
REPO = "anonymous/InjecForks"


def snapshot(job):
    if job.get("_root"):   # a local copy in the dataset's layout (E112's smoke rollouts)
        return Path(job["_root"])
    from huggingface_hub import snapshot_download
    pats = []
    for cfg, splits in job["select"].items():
        for sp in splits:
            pats += [f"{cfg}/{sp}.jsonl", f"prompts/{cfg}/{sp}.jsonl", f"cuts/{cfg}/{sp}.jsonl", f"eval-v1.2/{cfg}/{sp}.jsonl"]
    return Path(snapshot_download(job.get("dataset_repo", REPO), repo_type="dataset", revision=job["dataset_revision"], allow_patterns=pats + ["SHA256SUMS*"]))


def verify(root):
    """Every file this run read must match the dataset's own manifests, and every data file fetched must be in one."""
    ok = set()
    for man in sorted(root.glob("SHA256SUMS*")):
        for l in open(man):
            h, rel = l.split()
            if (root / rel).exists():
                assert sha(root / rel) == h, rel; ok.add(rel)
    fetched = {str(p.relative_to(root)) for p in root.rglob("*.jsonl")}
    assert fetched <= ok, ("fetched files no manifest covers", sorted(fetched - ok)[:5])
    return len(ok)


def render(tok, body):
    """e77_variants.local_count's transform, applied to the stored request body."""
    m = [dict(x) for x in body["messages"]]
    a = m[2]
    assert a["role"] == "assistant" and len(m) == 4 and m[3]["role"] == "tool"
    if "reasoning" in a:
        a["reasoning_content"] = a.pop("reasoning")
    a["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}} for c in a["tool_calls"]]
    chat = tok.apply_chat_template(m, tools=body["tools"], add_generation_prompt=True, enable_thinking=True, tokenize=False)
    assert chat.endswith("<think>\n"), chat[-40:]
    return tok(chat, add_special_tokens=False).input_ids


def build_items(job, root, tok, limit_cases=None):
    """[(case_key, prompt_ids, [(row_id, reasoning_ids[:maxN], {N: read offset in the suffix})])], asserting every join."""
    maxn = max(job["ncut"]); think_end = tok.convert_tokens_to_ids("</think>")
    out = []
    for cfg, splits in job["select"].items():
        for sp, which in splits.items():
            rows = [json.loads(l) for l in open(root / cfg / f"{sp}.jsonl")]
            cuts = {(c["case"], c["rollout"]): c for c in map(json.loads, open(root / "cuts" / cfg / f"{sp}.jsonl"))}
            keep = None
            if which == "eval":
                keep = {(e["case"], e["rollout"]) for e in map(json.loads, open(root / "eval-v1.2" / cfg / f"{sp}.jsonl"))}
            elif which == "evalextra":   # the eval cases' other rollouts (injecforks_load subset="evalextra")
                ev = {(e["case"], e["rollout"]) for e in map(json.loads, open(root / "eval-v1.2" / cfg / f"{sp}.jsonl"))}
                keep = {(r["case"], r["rollout"]) for r in rows if r["case"] in {c for c, _ in ev}} - ev
            else:
                assert which == "all", which
            by = {}
            for r in rows:
                if keep is None or (r["case"], r["rollout"]) in keep:
                    by.setdefault(r["case"], []).append(r)
            want = set(sorted(by)[:limit_cases] if limit_cases else by)
            with open(root / "prompts" / cfg / f"{sp}.jsonl") as f:
                bodies = {}
                for l in f:
                    case = re.match(r'\{"case": "([^"]+)"', l).group(1)      # the file's first key; parse only the rows needed
                    if case in want:
                        bodies[case] = json.loads(l)
            assert set(bodies) == set(want), ("prompts missing", cfg, sp, len(set(want) - set(bodies)))
            for case in sorted(want):
                b = bodies[case]
                ids = render(tok, b)
                assert len(ids) == b["prompt_tokens"], ("billed prompt tokens differ", cfg, sp, case, len(ids), b["prompt_tokens"])
                rr = []
                for r in sorted(by[case], key=lambda r: r["rollout"]):
                    enc = tok(r["reasoning"], add_special_tokens=False, return_offsets_mapping=True)
                    rid, offs = enc.input_ids, enc.offset_mapping
                    assert rid and think_end not in rid, (case, r["rollout"])
                    cu = cuts[(case, r["rollout"])]["cut"]
                    reads = {}
                    for n in job["ncut"]:
                        c = cu.get(str(n))
                        assert (c is None) == (len(rid) < n), ("cut null-ness", case, r["rollout"], n)
                        if c is not None:
                            assert offs[n - 1][1] == c, ("cut offset", case, r["rollout"], n, offs[n - 1][1], c)
                            reads[n] = n - 1
                    rr.append((f"{cfg}|{sp}|{case}|{r['rollout']}", rid[:maxn], reads))
                out.append((f"{cfg}|{sp}|{case}", ids, rr))
    return out


def prep(job):
    from transformers import AutoTokenizer
    g = job["generator"]
    tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"])
    root = snapshot(job); print("verified files:", verify(root))
    items = build_items(job, root, tok)
    P = sorted(len(p) for _, p, _ in items); R = sum(len(rr) for _, _, rr in items)
    toks = sum(len(p) for _, p, _ in items) + sum(len(x[1]) for _, _, rr in items for x in rr)
    reads = len(items) + sum(len(x[2]) for _, _, rr in items for x in rr)
    gb = reads * len(job["layers"]) * job["d_model"] * 2 / 1e9
    print(dict(cases=len(items), rows=R, prompt_tokens_p50_max=[P[len(P) // 2], P[-1]], tokens_forwarded=toks, state_reads=reads,
               states_gb=round(gb, 1)))
    print("by config/split:", {k: sum(1 for c, _, _ in items if c.startswith(k + "|")) for k in
                               {f"{cfg}|{sp}" for cfg, s in job["select"].items() for sp in s}})


def run(job, smoke=False, local=False, check=False):
    import torch
    from common import load_generator, LayerTap
    assert torch.cuda.is_available()
    api = None
    if check:   # the cached-vs-full check on the first max_cases cases of the selection, every read; no Hub, no assertion
        run_id = "testcache-local" if local else "testcache"   # (e79_final.py applies the threshold to results/.../cache_check.json)
        if local:
            job = dict(job, generator=dict(job["smoke_generator"]), layers=job["smoke_layers"], d_model=job["smoke_d_model"])
    elif smoke or local:   # local: the small model over the config's whole selection, no Hub (code checks only, not E79 data)
        if job.get("smoke_dataset_root"):   # E112: the smoke rollouts e112_rollouts.py wrote locally
            job = dict(job, _root=str(HERE / job["smoke_dataset_root"]))
        run_id = "local" if local else "smoke"
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
    pre = job.get("out_name", "")   # out_name: a run beside E79 v2's; its smoke/local outputs go beside it too, not over E79 v2's
    name = run_id if check else f"{pre}-smoke".lstrip("-") if smoke else f"{pre}-local".lstrip("-") if local else pre
    D = HERE / "data/e79v2" / (f"{name}-states" if name else "states"); D.mkdir(parents=True, exist_ok=True)
    RES = HERE / "results/e79v2-cloud" / name; (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"])
    lm = model.model.language_model
    items = build_items(job, root, tok, limit_cases=job.get("smoke_cases") if smoke else job["max_cases"] if check else None)
    if smoke:   # one smoke case and one realistic case, both injected, plus one benign
        pick = [next(i for i in items if i[0].startswith(p)) for p in ("smoke|", "realistic|", "realistic-benign|") if any(i[0].startswith(p) for i in items)]
        items = [(c, p, rr[:3]) for c, p, rr in pick]
    keep = job["layers"]
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, started=time.time(), layers=keep, ncut=job["ncut"],
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
        if check:   # the list e79_final.py --cache-check reads
            (RES / "cache_check.json").write_text(json.dumps([dict(row_id=x["row_id"], min_cosine=x["min_cosine"]) for x in receipt["cache_check"]], indent=1))
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

    t0 = time.time()
    try:
        for ci, (case, pids, rr) in enumerate(items):
            P = len(pids)
            got, hs, cache = forward(pids, [P - 1])
            first_logit = model.lm_head(hs[-1:]).float()             # predicts the first reasoning token
            c0 = np.stack([got[li].numpy() for li in keep], 1)
            buf_X.append(c0); buf_M.append(dict(case=case, N=0, kind="c0"))
            for k, (row_id, rid, reads) in enumerate(rr):
                pos = [reads[n] for n in job["ncut"] if n in reads]
                g2, h2, _ = forward(rid, pos, past=copy.deepcopy(cache), start=P)
                logits = torch.cat([first_logit, model.lm_head(h2[:-1]).float()])
                lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(rid, device=logits.device)[:, None])[:, 0]
                nll = [round(float(-lp.mean()), 4), round(float(-lp[:8].mean()), 4)]
                sx = np.stack([g2[li].numpy() for li in keep], 1) if pos else np.zeros((0, len(keep), c0.shape[-1]), c0.dtype)
                if pos:
                    buf_X.append(sx)
                    buf_M += [dict(row_id=row_id, N=n, kind="prefix", nll_mean_first8=nll) for n in job["ncut"] if n in reads]
                if smoke or check or ci < 3:   # the cached path must equal one full forward at every read, per layer
                    gf, _, _ = forward(pids + rid, [P - 1] + [P + p for p in pos])
                    full = np.stack([gf[li].numpy() for li in keep], 1); cached = np.concatenate([c0, sx])
                    cos = float(np.min((full * cached).sum(-1) / np.linalg.norm(full, axis=-1) / np.linalg.norm(cached, axis=-1)))
                    receipt["cache_check"].append(dict(row_id=row_id, min_cosine=round(cos, 5), nll=nll))
                    assert check or cos > job.get("min_cache_cosine", 0.995), ("cached path differs from the full forward", row_id, cos)
            del cache, hs
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
    st = HERE / "data/e79v2/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"{prefix}/*"], local_dir=st)
    D = HERE / "data/e79v2/states"; D.mkdir(parents=True, exist_ok=True)
    for p in (st / prefix / "states").glob("*"):
        shutil.copy2(p, D / p.name)
    if (st / prefix / "results").exists():
        shutil.copytree(st / prefix / "results", HERE / "results/e79v2-cloud", dirs_exist_ok=True)
    print("fetched", prefix, len(list(D.glob("*.npy"))), "shards")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["prep", "smoke", "local", "run", "fetch", "cachecheck", "cachecheck-local"]); ap.add_argument("--prefix")
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text()); os.chdir(HERE)
    {"prep": lambda: prep(job), "smoke": lambda: run(job, smoke=True), "local": lambda: run(job, local=True), "run": lambda: run(job), "fetch": lambda: fetch(job, a.prefix),
     "cachecheck": lambda: run(job, check=True), "cachecheck-local": lambda: run(job, check=True, local=True)}[a.stage]()
