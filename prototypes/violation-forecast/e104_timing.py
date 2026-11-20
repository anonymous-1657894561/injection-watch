"""E104: seconds per prediction for every GPU method, with the context cached and from scratch (experiments/E104-monitoring-cost.md).

  python e104_timing.py configs/e104-timing-pod.yaml run     # pod: the 9B, the Distill observer and the classifiers, one card
  python e104_timing.py configs/e104-timing-pod.yaml smoke   # local CPU: small stand-in models, 2 cases (code check, not timings)

Per case (the first eval rollout that reaches N), each timing the median of `reps` after one warm-up; a mutating step gets a fresh
cache copy outside the timed region. Timed pieces:
  9B (bf16)    agent_prefill (prompt), agent_decode_token (one cached step: the unit everything is compared to),
               probe_dot (a 4,096-d dot product on GPU: the in-agent probe's extra work; weights random, timing only),
               probe_uncached (prompt + N tokens re-encoded, all layers: an out-of-process probe; layer 20 needs 20 / 32 of it),
               prefixonly_uncached (the N tokens alone: E090's prefix-only probe),
               selfreport_cached (copy the cache + the `turn` comply question + lm_head), selfreport_uncached (all of it from scratch)
  Distill      distill_cached (the N-token prefix on a cached context), distill_uncached (context + prefix from scratch)
  classifiers  per case's tool result, E094's 510-token windows (no N)
"""
import argparse, copy, importlib.util, json, os, statistics as st, sys, time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_generator                                   # noqa: E402
from injecforks_load import load                                    # noqa: E402
from injecforks_states import build_items, sha, verify              # noqa: E402
from injecforks_selfreport import FMT, Q, answer_ids                # noqa: E402
import injecforks_observer_states as O                              # noqa: E402
import injecforks_input_classifiers as C                            # noqa: E402

torch.set_grad_enabled(False)


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def timed(dev, reps, fn, setup=lambda: None):
    """Median seconds of fn(setup()) over reps, after one warm-up; setup runs outside the timed region."""
    fn(setup())
    ts = []
    for _ in range(reps):
        s = setup(); sync(dev); t = time.perf_counter(); fn(s); sync(dev); ts.append(time.perf_counter() - t)
    return st.median(ts)


def fwd(lm, ids, dev, cache=None, start=0):
    return lm(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True,
              position_ids=torch.arange(start, start + len(ids), device=dev)[None])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["run", "smoke"]); a = ap.parse_args()
    job = yaml.safe_load(open(HERE / a.config)); smoke = a.stage == "smoke"
    dev = "cpu" if smoke else "cuda"
    assert smoke or torch.cuda.is_available()
    if not smoke:
        for rel, expected in job["config_sha256"].items():
            if sha(HERE / rel) != expected:
                raise SystemExit(f"configuration drift since launch: {rel}")
    RES = HERE / job["results_dir"] / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    reps, ncases = (1, job["smoke_cases"]) if smoke else (job["reps"], job["max_cases"])
    root = O.snapshot(job); nver = verify(root)
    out = dict(gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu", run_id=os.environ.get("RPJOB_RUN_ID", a.stage),
               dataset_revision=job["dataset_revision"], verified_files=nver, reps=reps, cases=[], classifiers={},
               # Qwen3.5's linear-attention layers fall back to plain torch without these: the 9B timings are then an upper bound
               fast_kernels={m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d")})

    gen = dict(job["smoke_generator" if smoke else "generator"]); gen["quant"] = "bf16"
    model, tok, _ = load_generator(gen, device=dev)
    if smoke:   # bf16 matmuls crawl on CPU; the pod times bf16
        model = model.float()
    lm, head = model.model.language_model, model.lm_head
    items = build_items(job, root, tok, limit_cases=ncases)
    ans = answer_ids(tok); yes, no = torch.tensor(ans["yes"], device=dev), torch.tensor(ans["no"], device=dev)
    qt = tok(FMT["turn"].format(q=Q["comply"]), add_special_tokens=False).input_ids
    w = torch.randn(lm.config.hidden_size, device=dev, dtype=torch.bfloat16)
    h = torch.randn(lm.config.hidden_size, device=dev, dtype=torch.bfloat16)
    if smoke:
        w, h = w.float(), h.float()
    for case, P, rr in items:
        rec = dict(case=case, prompt_tokens=len(P), n={})
        rec["agent_prefill"] = timed(dev, reps, lambda _: fwd(lm, P, dev))
        base = fwd(lm, P, dev).past_key_values
        rec["agent_decode_token"] = timed(dev, reps, lambda c: fwd(lm, P[-1:], dev, c, len(P)), lambda: copy.deepcopy(base))
        rec["probe_dot"] = timed(dev, reps, lambda _: float(h @ w))
        for n in job["ncut"]:
            row = next((x for x in rr if n in x[2]), None)
            if row is None:
                continue
            rid = row[1][:n]
            c_n = fwd(lm, rid, dev, copy.deepcopy(base), len(P)).past_key_values

            def ask(ids, c=None, start=0):   # q(comply) at the question's last token
                p = torch.softmax(head(fwd(lm, ids, dev, c, start).last_hidden_state[0, -1]).float(), -1)
                return float(p[yes].sum() / (p[yes].sum() + p[no].sum()))

            rec["n"][n] = dict(
                row_id=row[0],
                probe_uncached=timed(dev, reps, lambda _: fwd(lm, P + rid, dev)),
                prefixonly_uncached=timed(dev, reps, lambda _: fwd(lm, rid, dev)),
                selfreport_cached=timed(dev, reps, lambda _: ask(qt, copy.deepcopy(c_n), len(P) + n)),
                selfreport_uncached=timed(dev, reps, lambda _: ask(P + rid + qt)))
            del c_n
        out["cases"].append(rec); print(case, json.dumps({k: v for k, v in rec.items() if k != "n"}), flush=True)
        del base
    del model, lm, head
    if dev == "cuda":
        torch.cuda.empty_cache()

    obs = dict(job["observer_smoke" if smoke else "observer"]); ojob = dict(job, generator=obs)
    omodel, otok, _ = load_generator(obs, device=dev)
    if smoke:
        omodel = omodel.float()
    _, atok = O.tokenizers(ojob, obs)
    olm = omodel.model
    by = {r["case"]: r for r in out["cases"]}
    for case, ctx, rr in O.build_items(ojob, root, otok, atok, limit_cases=ncases):
        rec = by[case]; rec["distill_context_tokens"] = len(ctx)
        cbase = olm(input_ids=torch.tensor([ctx], device=dev), use_cache=True).past_key_values
        for n in job["ncut"]:
            row = next((x for x in rr if n in x[2]), None)
            if row is None or n not in rec["n"]:
                continue
            k, pos = row[2][n]; pre = row[1][k][:pos + 1]
            rec["n"][n].update(
                distill_cached=timed(dev, reps, lambda c: olm(input_ids=torch.tensor([pre], device=dev), past_key_values=c, use_cache=True),
                                     lambda: copy.deepcopy(cbase)),
                distill_uncached=timed(dev, reps, lambda _: olm(input_ids=torch.tensor([ctx + pre], device=dev), use_cache=False)))
        del cbase
    del omodel, olm
    if dev == "cuda":
        torch.cuda.empty_cache()

    texts = {r["case"]: str(r["tool_response"]) for cfg, s in job["select"].items() for sp in s for r in load(cfg, sp, root=root)}
    for name in job["classifiers"]:
        ctok, m = C.model(name, dev)
        out["classifiers"][name] = {c["case"]: timed(dev, reps, lambda _, t=texts[c["case"].split("|")[-1]]: C.score([t], ctok, m, dev))
                                    for c in out["cases"]}
        del m
    out["complete"] = True
    (RES / "timing.json").write_text(json.dumps(out, indent=1))
    print("done", RES / "timing.json")


if __name__ == "__main__":
    main()
