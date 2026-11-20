"""E103, agent side (dev exploratory; test pre-registered 2026-11-11): does Qwen3.5-9B still follow the rewordings that evade PromptGuard 2? Pairs each variant
reply (`e77_variants.py ... --e103=T2|T4`) with the dataset's own T0 reply of the same case and rollout number (same seed,
sampler, provider and prompt apart from the injected block). Compliance on the broad label (the general arm's) and the strict
one; T - T0 with a two-way bootstrap over cases and rollout numbers. CPU, free.

  ~/.venvs/e01/bin/python e103_agent.py [--split dev|test] [--variants T2 T4]   -> results/e103/agent-<split>.json
"""
import argparse, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T0 = HERE / "data/e78/injecforks_v1/realistic/{split}.jsonl"
VAR = HERE / "data/e78/variants/e78v2_important_no_names_qwen_thought_real_e103{v}_{split}_rollouts.jsonl"
NB = 2000


def rows(path):
    return {(x["case"], x["rollout"]): x for x in map(json.loads, open(path))}


def boot(D, seed=0):
    """D: [cases, rollouts] paired differences. Two-way bootstrap: cases and rollout numbers resampled independently."""
    rng = np.random.default_rng(seed); C, R = D.shape
    xs = [D[np.ix_(rng.integers(0, C, C), rng.integers(0, R, R))].mean() for _ in range(NB)]
    return [round(float(D.mean()), 4)] + [round(float(q), 4) for q in np.percentile(xs, [2.5, 97.5])]


ap = argparse.ArgumentParser(); ap.add_argument("--variants", nargs="+", default=["T2", "T4"])
ap.add_argument("--split", default="dev", choices=["dev", "test"]); a = ap.parse_args()
base = rows(str(T0).format(split=a.split)); out = {}
for v in a.variants:
    var = rows(Path(str(VAR).format(v=v, split=a.split)))
    cases = sorted({c for c, _ in var}); R = sorted({r for _, r in var})
    assert all((c, r) in base for c in cases for r in R), "every variant reply has its T0 pair"
    assert all(var[(c, r)]["tool_response"] != base[(c, r)]["tool_response"] for c in cases for r in R), "the block was rewritten"
    o = out[v] = dict(cases=len(cases), rollouts=len(R))
    for lab in ("violates_broad", "violates_strict"):
        t = np.array([[var[(c, r)][lab] for r in R] for c in cases], float)
        b = np.array([[base[(c, r)][lab] for r in R] for c in cases], float)
        o[lab] = dict(T0=round(float(b.mean()), 4), T=round(float(t.mean()), 4), diff=boot(t - b),
                      cases_any_T0=int((b.sum(1) > 0).sum()), cases_any_T=int((t.sum(1) > 0).sum()))
    o["capped"] = sum(var[k]["finish"] == "length" for k in var)
    print(v, json.dumps(o), flush=True)
(HERE / "results/e103").mkdir(parents=True, exist_ok=True)
(HERE / f"results/e103/agent-{a.split}.json").write_text(json.dumps(out, indent=1))
