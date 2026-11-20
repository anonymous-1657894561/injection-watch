"""E134, dev only (before the test read). Local GPU, $0.

1. C grid on dev for the e5 reasoning-prefix embedding (train-fitted), both configs, N in 8..128. Selection: the C with the
   highest mean within-case AUROC over both configs x the five N. Frozen in results/e134/frozen.json.
2. At the frozen C: dev score files e134-emb-<cfg>-dev.jsonl (N = 0: constant 0, the empty prefix).
3. Probe + embedding combination fitted on dev: conditional logit (case fixed effects) on the within-case-demeaned frozen
   comply probe (e79final) and embedding scores, each divided by its dev SD. Frozen as score = a * probe + b * emb per
   (config, N) (results/e134/frozen.json); within-case ranks equal the z-scored fit's. Dev comb scores are in-sample.
4. Part B: the embedding (realistic train, frozen C) on E103's dev variants T2/T4/T5 vs the probe's e103-9b scores; T0 = the
   dataset's realistic dev rows of the same 55 cases (probe e79final). T - T0 on cases mixed in both, case bootstrap, and the
   difference (emb's T - T0) - (probe's T - T0). T5 has no violations: AUROC of T0's violating rows vs T5 rows instead.

  python e134_dev.py   -> results/e134/{dev-grid.json, frozen.json, variants-dev.json}, scores/e134-{emb,comb}-<cfg>-dev.jsonl
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.discrete.conditional_models import ConditionalLogit

from e134_common import NS, OUT, SC, embed, fit, load, load_variant, pre, read_scores, score_rows
from injecforks_metrics import per_case_auroc, within_case_mean

GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
NB = 2000
OUT.mkdir(parents=True, exist_ok=True)


def boot(d, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    xs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)]
    return [round(float(d.mean()), 3)] + [round(float(q), 3) for q in np.percentile(xs, [2.5, 97.5])]


# 1. C grid
dev = {c: load(c, "dev") for c in ("smoke", "realistic")}
grid = []
for cfg in dev:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr = load(cfg, "train", subset=None)
    for N in NS:
        a = [r for r in tr if r["cut"].get(N) is not None]
        b = [r for r in dev[cfg] if r["cut"].get(N) is not None]
        Xa, Xb = embed(f"{cfg}-train-N{N}", [pre(r, N) for r in a]), embed(f"{cfg}-dev-N{N}", [pre(r, N) for r in b])
        ya = [bool(r["violates"]) for r in a]
        for C in GRID:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")).fit(Xa, ya)
            s = clf.decision_function(Xb)
            grid.append(dict(cfg=cfg, N=N, C=C, auroc=within_case_mean([r["case"] for r in b], [r["violates"] for r in b], s)))
            print(grid[-1], flush=True)
g = pd.DataFrame(grid)
mean_by_C = g.groupby("C").auroc.mean()
C = float(mean_by_C.idxmax())
(OUT / "dev-grid.json").write_text(json.dumps(dict(grid=grid, mean_by_C={str(k): v for k, v in mean_by_C.items()}, chosen_C=C), indent=1))
print(g.pivot_table(index=["cfg", "N"], columns="C", values="auroc").round(3).to_string(), "\nmean by C", mean_by_C.round(4).to_dict(), "chosen", C)

# 2-3. dev scores at frozen C and the combination
frozen = dict(C=C, model="intfloat/e5-large-v2", text='"query: " + reasoning[:cut[N]]', NS=NS, probe="e79final-comply", comb={})
for cfg, rows in dev.items():
    probe = read_scores(SC / f"e79final-comply-{cfg}-dev.jsonl")
    emb_out, comb_out = [], []
    for r in rows:
        emb_out.append({"row_id": r["row_id"], "N": 0, "score": 0.0})
    frozen["comb"][cfg] = {"0": dict(a=1.0, b=0.0)}
    for r in rows:
        comb_out.append({"row_id": r["row_id"], "N": 0, "score": probe[(r["row_id"], 0)]})
    for N in NS:
        e = score_rows(fit(cfg, N, C), f"{cfg}-dev", rows, N)
        b = [r for r in rows if r["cut"].get(N) is not None]
        d = pd.DataFrame(dict(case=[r["case"] for r in b], y=[int(r["violates"]) for r in b],
                              probe=[probe[(r["row_id"], N)] for r in b], emb=[e[r["row_id"]] for r in b]))
        dm = d.groupby("case")[["probe", "emb"]].transform(lambda x: x - x.mean()); sd = dm.std()
        f = ConditionalLogit(d.y, dm / sd, groups=d.case).fit(disp=0)
        a_, b_ = float(f.params["probe"] / sd["probe"]), float(f.params["emb"] / sd["emb"])
        frozen["comb"][cfg][str(N)] = dict(a=a_, b=b_, z_coef_probe=float(f.params["probe"]), z_coef_emb=float(f.params["emb"]),
                                           p_probe=float(f.pvalues["probe"]), p_emb=float(f.pvalues["emb"]))
        for r in b:
            emb_out.append({"row_id": r["row_id"], "N": N, "score": e[r["row_id"]]})
            comb_out.append({"row_id": r["row_id"], "N": N, "score": a_ * probe[(r["row_id"], N)] + b_ * e[r["row_id"]]})
        print(cfg, N, frozen["comb"][cfg][str(N)], flush=True)
    for nm, out in (("emb", emb_out), ("comb", comb_out)):
        with open(SC / f"e134-{nm}-{cfg}-dev.jsonl", "w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in out)
(OUT / "frozen.json").write_text(json.dumps(frozen, indent=1))

# 4. Part B: E103 dev variants (realistic arm)
T0 = dev["realistic"]
V = {v: load_variant(f"realistic-e103{v}", "dev") for v in ("T2", "T4", "T5")}
assert {r["case"] for r in T0} == {r["case"] for r in V["T2"]} and len(T0) == 440
P = {"T0": read_scores(SC / "e79final-comply-realistic-dev.jsonl"),
     **{v: read_scores(SC / f"e103-9b-comply-realistic-e103{v}-dev.jsonl") for v in V}}
res = {}
for N in NS:
    clf = fit("realistic", N, C)
    E = {"T0": score_rows(clf, "realistic-dev", T0, N), **{v: score_rows(clf, f"realistic-e103{v}-dev", V[v], N) for v in V}}
    rows = {"T0": T0, **V}
    o = res[N] = {}
    pc = {}
    for t, rs in rows.items():
        rs = [r for r in rs if r["cut"].get(N) is not None]
        for m, S in (("emb", {r["row_id"]: E[t][r["row_id"]] for r in rs}), ("probe", {r["row_id"]: P[t][(r["row_id"], N)] for r in rs})):
            pc[(t, m)] = per_case_auroc([r["case"] for r in rs], [r["violates"] for r in rs], [S[r["row_id"]] for r in rs])
            o[f"{t} {m}"] = dict(rows=len(rs), viol=int(sum(r["violates"] for r in rs)), mixed=len(pc[(t, m)]),
                                 auroc=round(float(np.mean(list(pc[(t, m)].values()))), 3) if pc[(t, m)] else None)
    for v in ("T2", "T4"):
        both = sorted(set(pc[(v, "emb")]) & set(pc[("T0", "emb")]))
        de = [pc[(v, "emb")][c] - pc[("T0", "emb")][c] for c in both]
        dp = [pc[(v, "probe")][c] - pc[("T0", "probe")][c] for c in both]
        o[f"{v} vs T0"] = dict(cases=len(both), emb_T=round(np.mean([pc[(v, 'emb')][c] for c in both]), 3),
                               emb_T0=round(np.mean([pc[('T0', 'emb')][c] for c in both]), 3),
                               probe_T=round(np.mean([pc[(v, 'probe')][c] for c in both]), 3),
                               probe_T0=round(np.mean([pc[('T0', 'probe')][c] for c in both]), 3),
                               emb_diff=boot(de), probe_diff=boot(dp), emb_minus_probe_diff=boot(np.subtract(de, dp)))
    zv = [r for r in T0 if r["violates"] and r["cut"].get(N) is not None]
    t5 = [r for r in V["T5"] if r["cut"].get(N) is not None]
    o["T0 violating vs T5 rows (pooled AUROC)"] = {
        "emb": round(roc_auc_score([1] * len(zv) + [0] * len(t5), [E["T0"][r["row_id"]] for r in zv] + [E["T5"][r["row_id"]] for r in t5]), 3),
        "probe": round(roc_auc_score([1] * len(zv) + [0] * len(t5), [P["T0"][(r["row_id"], N)] for r in zv] + [P["T5"][(r["row_id"], N)] for r in t5]), 3),
        "violating": len(zv), "t5_rows": len(t5)}
    print(N, json.dumps(o), flush=True)
(OUT / "variants-dev.json").write_text(json.dumps(res, indent=1))
