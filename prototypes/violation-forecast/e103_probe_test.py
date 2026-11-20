"""E103, agent side on test (pre-registered 2026-11-11): the frozen e79final comply probe (5-seed ensemble) on the T2/T4 test
rollouts (InjecForks v1.6), against T0 = the dataset's own rows of the same 299 cases, all 16 rollouts (E79's eval rows plus
E113's evalextra). Reads the scores `injecforks_probe.py --fixed results/injecforks/e79final-selection.json --ensemble 5
--configs realistic --score test --e103 T2 T4 --out e103-9b` writes. CPU, free.

Within-case AUROC at N = 16 (and 8) on each variant's mixed cases; T - T0 on the cases mixed in both, case bootstrap. T0 is
labelled two ways: the v1.2 label (E79/E113's) and the rollouts' own broad column (the variants' labeller).

  python e103_probe_test.py   -> results/e103/probe-test.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from injecforks_load import load, load_variant

HERE = Path(__file__).resolve().parent
SC = HERE / "results/injecforks/scores"
NS, NB = (8, 16), 2000


def scores(*names):
    out = defaultdict(dict)
    for name in names:
        for x in map(json.loads, open(SC / name)):
            out[int(x["N"])][x["row_id"]] = x["score"]
    return out


def wc(rows, s, lab):
    by = defaultdict(list)
    for r in rows:
        if r["row_id"] in s:
            by[r["case"]].append((bool(r[lab]), s[r["row_id"]]))
    return {c: roc_auc_score([v for v, _ in xs], [z for _, z in xs]) for c, xs in by.items() if 0 < sum(v for v, _ in xs) < len(xs)}


def boot(d, seed=0):
    rng = np.random.default_rng(seed); d = np.array(d)
    xs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)]
    return [round(float(d.mean()), 3)] + [round(float(q), 3) for q in np.percentile(xs, [2.5, 97.5])]


T0 = load("realistic", "test") + load("realistic", "test", subset="evalextra")
raw = {(x["case"], x["rollout"]): x["violates_broad"] for x in map(json.loads, open(HERE / "data/e78/injecforks_v1/realistic/test.jsonl"))}
T0 = [{**r, "raw_broad": raw[(r["case"], r["rollout"])]} for r in T0]
V = {v: load_variant(f"realistic-e103{v}", "test") for v in ("T2", "T4")}
assert all(len(V[v]) == 4784 for v in V) and len(T0) == 4784 and {r["case"] for r in T0} == {r["case"] for r in V["T4"]}
S0 = scores("e79final-comply-realistic-test.jsonl", "e79final-e113-comply-realistic-test.jsonl")
SV = scores("e103-9b-comply-realistic-e103T2-test.jsonl", "e103-9b-comply-realistic-e103T4-test.jsonl")
# the refitted probe must reproduce E79's frozen T0 scores (it rescored the eval rows in the same run)
S0re = scores("e103-9b-comply-realistic-test.jsonl")
d = [abs(S0re[16][k] - S0[16][k]) for k in S0re[16] if k in S0[16]]
# Measured 2026-11-11: max |diff| 0.0025 (median 6e-4; E113's refit was 1e-6), rank-identical: the within-case AUROC on E79's
# eval rows is equal to E79's at N = 8 and 16. Guarded at that level: the AUROC must match and the shift stay < 0.01.
a_re, a_fr = wc(load("realistic", "test"), S0re[16], "violates"), wc(load("realistic", "test"), scores("e79final-comply-realistic-test.jsonl")[16], "violates")
res = dict(t0_reproduction=dict(rows=len(d), max_abs_diff=float(max(d)), auroc_refit=float(np.mean(list(a_re.values()))),
                                auroc_frozen=float(np.mean(list(a_fr.values())))))
assert d and max(d) < 0.01 and abs(res["t0_reproduction"]["auroc_refit"] - res["t0_reproduction"]["auroc_frozen"]) < 1e-9, res
for n in NS:
    o = res[n] = {}
    for lab in ("violates", "raw_broad"):
        a0 = wc(T0, S0[n], lab); o[f"T0 within-case ({lab})"] = dict(mixed=len(a0), auroc=round(float(np.mean(list(a0.values()))), 3))
        for v in V:
            av = wc(V[v], SV[n], "violates"); both = sorted(set(a0) & set(av))
            o[f"{v} vs T0 ({lab})"] = dict(mixed=len(av), auroc=round(float(np.mean(list(av.values()))), 3), both=len(both),
                                          T_same_cases=round(float(np.mean([av[c] for c in both])), 3),
                                          T0_same_cases=round(float(np.mean([a0[c] for c in both])), 3),
                                          diff=boot([av[c] - a0[c] for c in both]))
    print(n, json.dumps(o), flush=True)
(HERE / "results/e103").mkdir(parents=True, exist_ok=True)
(HERE / "results/e103/probe-test.json").write_text(json.dumps(res, indent=1))
