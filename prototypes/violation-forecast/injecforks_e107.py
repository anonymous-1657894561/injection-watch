"""E107: the N = 16 reminder, always on and gated on the frozen comply probe, on InjecForks v1.4 realistic look-alikes beside
the benign twins (experiments/E107-gated-reminder-lookalike-cost.md). Qwen3.5-9B BF16, E090's sampler.

  ~/.venvs/e01/bin/python injecforks_e107.py configs/e107-9b-dev-pod.yaml dry [--split dev|test]   # CPU: tokenizer only, no model
  ~/.venvs/e01/bin/python injecforks_e107.py configs/e107-9b-dev-pod.yaml smoke                    # local GPU: Qwen3.5-0.8B, short
  python injecforks_e107.py configs/e107-9b-dev-pod.yaml run --split dev                           # pod (run_e107_pod.sh)
  python injecforks_e107.py configs/e107-9b-test-pod.yaml run --split test --final                 # pod, once (testread.py)
  ~/.venvs/e01/bin/python injecforks_e107.py configs/e107-9b-dev-pod.yaml fetch --prefix RUN_ID
  ~/.venvs/e01/bin/python injecforks_e107.py configs/e107-9b-dev-pod.yaml score --prefix RUN_ID    # states -> comply ensemble scores

E097 part (a) in procedure, on realistic-{benign,lookalike}: every eval-v1.2 row (4 per case) is continued from its first 16
reasoning tokens under control and under E093's reminder, and its layer-20 state at N = 16 (the comply probe's read; the prefix is
unsteered and shared by both arms) is saved. The gate is applied offline: a gated reply is the reminder continuation where the
frozen 5-seed comply ensemble fires at E097's deployment threshold, and the control continuation elsewhere, which is what a deployed
gate would have produced from the same prefix. A base's twins and look-alikes share the generator seed, and work is ordered by base
so a timer kill keeps whole pairs. No dataset label is read into a record. Output: results/e107-cloud/{run.json, e107.jsonl,
states.npy, states-meta.jsonl}; uploaded under e107/<run_id>/.
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_e097 import prefixes, rows
from injecforks_e105 import cases, parse
from injecforks_states import render, sha, snapshot, verify
import testread

NAME = "e107-n16"
FAMILY = "realistic"   # the probe and its threshold are realistic's
CFG = {"twin": "realistic-benign", "lookalike": "realistic-lookalike"}
COUNTS = {"dev": dict(cases=24, rows=4), "test": dict(cases=176, rows=4)}   # per config, asserted


def frozen(job):
    """The fixed numbers, asserted against E097's frozen config (decisions/, 2026-10-30) and E105's."""
    e097, e105 = (yaml.safe_load((HERE / f"configs/{c}.yaml").read_text()) for c in ("e097-9b-test-pod", "e105-9b-test-pod"))
    assert job["sampling"] == e097["sampling"] == e105["sampling"], "E090's sampler"
    assert job["generator"] == e097["generator"] == e105["generator"] and job["max_new_tokens"] == e097["max_new_tokens"] == 1536
    assert job["n"] == e097["n"] == 16 and job["state_layer"] == e097["state_layer"] == 20
    assert job["reminder"] == e097["reminder"], "E093's N = 16 reminder"
    assert hashlib.sha256(job["reminder"].encode()).hexdigest() == job["reminder_sha256"]
    assert job["dataset_revision"] == e105["dataset_revision"] == "v1.4"


def work(root, split, max_bases=None):
    """[(base, [(cat, kind, case, body, user_tool, [(rollout, reasoning, attacker_tools, cut)])])], each base holding its two
    twins and two look-alikes (asserted), every row reaching N = 16 (asserted)."""
    by = {}
    for cat, cfg in CFG.items():
        meta = {cs: (ut, at) for cs, _, ut, at, _ in cases(root, split, FAMILY, cat)[0]}
        W = rows(root, split, cfg, 16, None)
        assert len(W) == COUNTS[split]["cases"] and all(len(rs) == COUNTS[split]["rows"] for *_, rs in W), (cfg, split, "rows reaching N = 16")
        assert {c for c, *_ in W} == set(meta), (cfg, split, "cases")
        for cs, body, rs in W:
            base, cat_, kind = parse(cs); assert cat_ == cat
            by.setdefault(base, []).append((cat, kind, cs, body, meta[cs][0], rs))
    for base, vs in by.items():
        assert sorted((c, k) for c, k, *_ in vs) == [("lookalike", "refrain"), ("lookalike", "remind"), ("twin", "imperative"), ("twin", "neutral")], base
        assert len({v[4] for v in vs}) == 1, (base, "twin and look-alike share the user tool")
    return [(b, sorted(by[b], key=lambda v: (v[0] != "twin", v[1]))) for b in sorted(by)][:max_bases or None]


def row_id(cat, split, case, rollout):
    return f"{CFG[cat]}|{split}|{case}|{rollout}"   # E102's score row ids


def select(job, split):
    return dict(job, select={c: [split] for c in CFG.values()})


def dry(job, split):
    """CPU, no model: counts, prefixes against the dataset's cuts, prompt lengths against the billed ones."""
    from transformers import AutoTokenizer
    frozen(job)
    g = job["generator"]; tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"])
    root = snapshot(select(job, split)); nver = verify(root); W = work(root, split); think_end = tok.convert_tokens_to_ids("</think>")
    rem = tok(job["reminder"], add_special_tokens=False).input_ids; n = 0
    for base, vs in W:
        for cat, kind, cs, body, ut, rs in vs:
            assert len(render(tok, body)) == body["prompt_tokens"], ("billed prompt tokens differ", cs)
            n += len(prefixes(tok, rs, job["n"], think_end))
    print(json.dumps(dict(split=split, verified_files=nver, bases=len(W), rows=n, sequences_total=2 * n, reminder_tokens=len(rem)), indent=1))


def run(job, split, final, smoke=False):
    import torch
    from common import LayerTap, load_generator
    from injecforks_resample import expand, label, sample
    if not smoke:
        testread.check(argparse.Namespace(split=split, final=final), NAME)
    assert torch.cuda.is_available()
    frozen(job); api = None
    assert smoke or split == job["split"], ("the config is for", job["split"])
    if smoke:
        job = dict(job, generator=dict(job["smoke_generator"]), state_layer=job["smoke_state_layer"], **job["smoke_overrides"]); run_id = "smoke"
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        if not api.repo_info(job["artifact_repo"]).private:
            raise SystemExit("artifact repo must be private")
        for rel, expected in job["config_sha256"].items():
            if sha(HERE / rel) != expected:
                raise SystemExit(f"configuration drift since launch: {rel}")
        if split == "test":   # the pod has no local receipt: a completed test run in the artifact repo is the receipt
            from huggingface_hub import hf_hub_download
            for f in api.list_repo_files(job["artifact_repo"]):
                if f.startswith("e107/") and f.endswith("results/run.json"):
                    rj = json.load(open(hf_hub_download(job["artifact_repo"], f)))
                    if rj.get("split") == "test" and rj.get("complete"):
                        raise SystemExit(f"E107 test was already run ({f}); a second run is not the pre-registered result")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = snapshot(select(job, split)); nver = verify(root); W = work(root, split, job.get("max_bases"))
    RES = HERE / "results/e107-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"]); lm = model.model.language_model
    eos = tok.convert_tokens_to_ids("<|im_end|>"); think_end = tok.convert_tokens_to_ids("</think>")
    sp, MAXN, NR, L = job["sampling"], job["max_new_tokens"], job["n"], job["state_layer"]
    rem = tok(job["reminder"], add_special_tokens=False).input_ids; V = model.lm_head.weight.shape[0]
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, split=split, started=time.time(), verified_files=nver, sampling=sp,
                   max_new_tokens=MAXN, n=NR, state_layer=L, reminder_tokens=len(rem), reminder_sha256=job["reminder_sha256"],
                   dataset_revision=job["dataset_revision"], generator=job["generator"], bases=len(W), cache_check=[])
    fout = open(RES / "e107.jsonl", "w" if smoke else "a"); SX, SM = [], []; last = time.time(); grab = {}
    tap = LayerTap(layers, [L], lambda li, h: grab.__setitem__("h", h))   # [B, T, d] output of layer L on the latest forward

    def flush(final_=False):
        fout.flush()
        if SX:
            np.save(RES / "states.npy", np.stack(SX).astype(np.float16))
            (RES / "states-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in SM))
        receipt.update(uploaded=time.time(), final=final_, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"e107/{run_id}/results")

    def fwd(ids, cache, start, B=1):
        x = ids if torch.is_tensor(ids) else torch.tensor(ids, device="cuda")
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        with torch.no_grad():
            return lm(input_ids=x, past_key_values=cache, use_cache=True,
                      position_ids=torch.arange(start, start + x.shape[1], device="cuda")[None].expand(x.shape[0], -1))

    def gen(tag):
        return torch.Generator(device="cuda").manual_seed(int(hashlib.sha256(f"e107|{split}|{tag}".encode()).hexdigest()[:8], 16))

    try:
        t0 = time.time()
        for wi, (base, vs) in enumerate(W):
            g = gen(base)   # shared by the base's twins and look-alikes
            for cat, kind, case, b, user_tool, rs in vs:
                pids = render(tok, b); assert len(pids) == b["prompt_tokens"], ("billed prompt tokens differ", case); P = len(pids)
                o = fwd(pids, None, 0); base_kv = o.past_key_values
                pre = prefixes(tok, rs, NR, think_end)[:job.get("k_cap") or None]; B = len(pre); atk = rs[0][2]
                x = torch.tensor([p for _, p in pre], device="cuda"); o1 = fwd(x, expand(base_kv, B), P, B)   # the probe's read, both arms' prefix
                for (rollout, _), h in zip(pre, grab["h"][:, -1].float().cpu().numpy()):
                    SX.append(h); SM.append(dict(row_id=row_id(cat, split, case, rollout), N=NR))
                if not receipt["cache_check"]:   # the copied cache equals one full forward (as E090 / E093 / E097)
                    with torch.no_grad():
                        hf = lm(input_ids=torch.tensor([pids + pre[-1][1]], device="cuda"), use_cache=False).last_hidden_state[0, -1].float()
                    cos = float(torch.nn.functional.cosine_similarity(o1.last_hidden_state[-1, -1].float(), hf, 0))
                    receipt["cache_check"].append(round(cos, 5)); assert cos > 0.995, ("copied cache differs from a full forward", cos)
                for arm, add in (("control", []), ("reminder", rem)):
                    if add:
                        o2 = fwd(torch.tensor([add] * B, device="cuda"), copy.deepcopy(o1.past_key_values), P + NR, B); start = P + NR + len(add)
                    else:
                        o2, start = o1, P + NR
                    seen = torch.zeros(B, V, dtype=torch.bool, device="cuda")
                    seen[torch.arange(B)[:, None], torch.tensor([p + add for _, p in pre], device="cuda")] = True
                    kv = o2.past_key_values if add else copy.deepcopy(o1.past_key_values)   # o1's cache is batch B already; sampling mutates it
                    outs = sample(model, lm, kv, model.lm_head(o2.last_hidden_state[:, -1]), start, seen, sp, eos, MAXN, g)
                    for (rollout, p), y in zip(pre, outs):
                        txt = tok.decode(p + add + y, skip_special_tokens=False); dec, v, calls = label(txt, atk)
                        fout.write(json.dumps(dict(part="n16", split=split, family=FAMILY, category=cat, kind=kind, base=base, case=case,
                                                   rollout=rollout, row_id=row_id(cat, split, case, rollout), user_tool=user_tool, arm=arm,
                                                   seed_tag=base, decided=dec, violates_strict=v, calls=calls, new_tokens=len(y), text=txt)) + "\n")
                    del o2
                del base_kv, o, o1; torch.cuda.empty_cache()
            print("n16", wi + 1, len(W), base, f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["complete"] = True
    finally:
        tap.close(); receipt["finished"] = time.time(); flush(final_=True); fout.close()
    if not smoke and receipt.get("complete"):
        testread.done(argparse.Namespace(split=split, final=final), NAME, run_id=run_id)
    print("done", json.dumps({k: receipt.get(k) for k in ("bases", "peak_gib", "cache_check")}))


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data/e107/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"e107/{prefix}/*"], local_dir=st)
    print("fetched to", st / "e107" / prefix)


def score(prefix):
    """The rows' layer-20 states at N = 16 -> the frozen 5-seed comply ensemble's scores ({row_id, N, score}). CPU; refits the
    ensemble on train (e096_analyze.ensemble, asserted on E079's dev scores). On dev, the scores are compared with E102's, read
    from the dataset's own rollouts by a full forward, and the comparison is written beside them."""
    from e096_analyze import ensemble
    R = HERE / "data/e107/dl/e107" / prefix / "results"
    X = np.load(R / "states.npy").astype(np.float32); M = [json.loads(l) for l in open(R / "states-meta.jsonl")]; assert len(X) == len(M)
    f, _ = ensemble(); s = f(X); out = R / "scores.jsonl"
    out.write_text("".join(json.dumps(dict(row_id=m["row_id"], N=m["N"], score=float(v))) + "\n" for m, v in zip(M, s)))
    ref = {}
    for c in ("benign", "lookalike"):
        p = HERE / f"results/injecforks/scores/e102-9b-comply-realistic-{c}-dev.jsonl"
        ref.update({x["row_id"]: x["score"] for x in map(json.loads, open(p)) if x["N"] == 16})
    # checked where it is used: the gate must fire on the same rows. (A 0.05 bound on |Δ| was set before any pod read was seen; the
    # dev pod's cached-prefix BF16 read differed by up to 0.080, median 0.010, sd of scores 1.53, with no gate decision changed. 2026-11-06)
    from e097_analyze import T
    thr = T["probe_ens"][1]; P = [(v, ref[m["row_id"]]) for m, v in zip(M, s) if m["row_id"] in ref]
    if P:
        a, b = np.array(P).T
        cmp = dict(rows=len(P), max_abs=float(np.max(np.abs(a - b))), median_abs=float(np.median(np.abs(a - b))), corr=float(np.corrcoef(a, b)[0, 1]),
                   fire_agreement=float(np.mean((a >= thr) == (b >= thr))), threshold=thr)
        (R / "scores-vs-e102.json").write_text(json.dumps(cmp, indent=1)); print("vs E102 dev:", cmp)
        assert cmp["rows"] == len(M) and cmp["fire_agreement"] >= 0.98 and cmp["corr"] > 0.999, \
            ("the pod's N = 16 read does not reproduce E102's gate; the gate is not the frozen one", cmp)
    print("wrote", out, len(M))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["dry", "smoke", "run", "fetch", "score"])
    ap.add_argument("--prefix"); testread.args(ap)
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text()); os.chdir(HERE)
    {"dry": lambda: dry(job, a.split), "smoke": lambda: run(job, "dev", False, smoke=True), "run": lambda: run(job, a.split, a.final),
     "fetch": lambda: fetch(job, a.prefix), "score": lambda: score(a.prefix)}[a.stage]()
