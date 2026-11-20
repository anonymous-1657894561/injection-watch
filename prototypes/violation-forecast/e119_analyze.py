"""E119 (experiments/E119-27b-risk-target-probes.md): the risk-target question on Qwen3.5-27B's own rollouts and states (A1's E112).
CPU, no calls, no generation.

  ~/.venvs/agentdojo/bin/python e119_analyze.py --rollouts ROOT --states DIR [DIR ...] --tag 27b     # the pre-registered pass
  ~/.venvs/agentdojo/bin/python e119_analyze.py --rollouts ROOT --states DIR --tag 9b-pipeline       # pipeline check on the 9B (not a result)

ROOT: an InjecForks-layout root (<config>/<split>.jsonl with violates_strict per rollout, cuts/<config>/<split>.jsonl); DIR: states in
injecforks_states' format (states-*.npy + meta-*.jsonl, 9 stored layers, N = 0 per case). Every head is fitted on train, chosen on dev,
scored once on test (the injected realistic test cases present in ROOT); the recipes are the 9B's, unchanged:
- comply16: E096's recipe (one random available N per run, stored layer index 3 = the 9B's layer 20 depth, C = 0.001, 5 seeds, balanced),
  per-run strict label; read at N = 16.
- comply_R0: E114's recipe (N = 0, layer index 3, C = 0.001, soft target = the input's share of strict runs); twins as 0 (no 27B look-alikes).
- exposure_M / risk_M: E115's matched pair (identical N = 0 train inputs: injected + twins; the same 9 layers x 5 C grid; the same
  target-neutral dev criterion: share of dev harmful runs flagged at a 10 % weighted reminder rate, weights attack 0.25 / twin 0.75).
- r0a: comply_R0 and comply16 stacked on dev runs (balanced logistic regression, C = 1).
Test (attack runs only: E112 has no test twins or look-alikes, and no reminder arms): AUROC for an untreated strict call among attack runs;
capture@k = share of harmful runs flagged when the gate reminds the top k of attack runs (ties fractional), k = 10 / 20 / 30 %;
within-input AUROC (comply16, r0a) on mixed inputs. 95 % CIs from 2,000 draws of whole test cases, seed 0, cutoffs recomputed per draw.
"""
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e114_analyze import soft_fit
from e115_analyze import fire_exact

L20, CS, SEEDS, C_FROZEN, B = 3, [1e-4, 1e-3, 1e-2, 1e-1, 1.0], 5, 0.001, 2000
KS = (0.10, 0.20, 0.30)
CONTRASTS = [("target", "risk_M", "exposure_M"), ("r0_vs_comply16", "comply_R0", "comply16"), ("r0_vs_exposure", "comply_R0", "exposure_M"),
             ("n16_adds", "r0a", "comply_R0")]


def rows(root, cfg, split):
    p = Path(root) / cfg / f"{split}.jsonl"
    if not p.exists():
        return []
    cuts = {(c["case"], c["rollout"]): {int(n): v for n, v in c["cut"].items()} for c in map(json.loads, open(Path(root) / "cuts" / cfg / f"{split}.jsonl"))}
    out = []
    for l in open(p):
        r = json.loads(l)
        out.append(dict(row_id=f'{cfg}|{split}|{r["case"]}|{r["rollout"]}', case=r["case"], cfg=cfg, split=split,
                        y=int(bool(r.get("violates_strict"))) if cfg == "realistic" else 0, cut=cuts[(r["case"], r["rollout"])]))
    return out


def inputs(R):
    by = defaultdict(list)
    for r in R:
        by[r["row_id"].rsplit("|", 1)[0]].append(r)
    return by


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from injecforks_probe import States, train_pairs
    ap = argparse.ArgumentParser(); ap.add_argument("--rollouts", required=True); ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--test-cases", type=int, default=0, help="pipeline check only: cap test inputs")
    a = ap.parse_args()
    S = States(a.states)
    D = {sp: rows(a.rollouts, "realistic", sp) + rows(a.rollouts, "realistic-benign", sp) for sp in ("train", "dev")}
    T = [r for r in rows(a.rollouts, "realistic", "test") if S.has(r["row_id"], 16)]
    if a.test_cases:
        keep = set(random.Random(0).sample(sorted({r["case"] for r in T}), a.test_cases)); T = [r for r in T if r["case"] in keep]
    for sp, R in D.items():
        miss = [r["row_id"] for r in R if not S.has(r["row_id"], 0)]; assert not miss, (sp, len(miss), miss[:2])
    # per-input targets
    def inp(R):
        by = inputs(R); ks = sorted(by)
        return ks, np.array([float(by[k][0]["cfg"] == "realistic") for k in ks]), np.array([np.mean([x["y"] for x in by[k]]) for k in ks]), by
    ktr, exp_tr, risk_tr, _ = inp(D["train"]); kdv, _, _, bydv = inp(D["dev"])
    X0 = lambda ks, li: S.get([(k + "|0", 0) for k in ks], li)
    # dev criterion (E115's, with twins in the look-alikes' place): share of dev harmful runs flagged at a 10 % weighted reminder rate
    dv_side = np.repeat([bydv[k][0]["cfg"] for k in kdv], [len(bydv[k]) for k in kdv]); dv_y = np.array([x["y"] for k in kdv for x in bydv[k]])
    n_a, n_t = (dv_side == "realistic").sum(), (dv_side != "realistic").sum()
    dv_w = np.where(dv_side == "realistic", 0.25 / n_a, 0.75 / n_t)
    def capture_dev(s_in):
        p = fire_exact(np.repeat(s_in, [len(bydv[k]) for k in kdv]), dv_w, 0.10); return float((p * dv_y).sum() / max(1, dv_y.sum()))
    sel, F = {}, {}
    for h, tgt in (("exposure_M", exp_tr), ("risk_M", risk_tr)):
        best = None
        for li in range(S.n_layers):
            Xt, Xd = X0(ktr, li), X0(kdv, li)
            for C in CS:
                sc, clf = soft_fit(Xt, tgt, C); cap = capture_dev(clf.decision_function(sc.transform(Xd)))
                sel.setdefault(h, []).append(dict(layer_index=li, C=C, dev_capture=cap))
                key = (round(cap, 6), -C, -abs(li - L20))
                if best is None or key > best[0]:
                    best = (key, li, C, sc, clf)
        _, li, C, sc, clf = best   # one prompt state per input: round so batched float noise (< 1e-6) cannot split an input's tie
        F[h] = (lambda li, sc, clf: lambda k0, x16: np.round(clf.decision_function(sc.transform(S.get([(k + "|0", 0) for k in k0], li))), 5))(li, sc, clf)
        sel[h + "_chosen"] = dict(layer_index=li, C=C)
    sc, clf = soft_fit(X0(ktr, L20), risk_tr, C_FROZEN)
    F["comply_R0"] = (lambda sc, clf: lambda k0, x16: np.round(clf.decision_function(sc.transform(S.get([(k + "|0", 0) for k in k0], L20))), 5))(sc, clf)
    tr_att = [r for r in D["train"] if r["cfg"] == "realistic"]; fs = []
    for s in range(SEEDS):
        pairs = train_pairs(tr_att, S, s, "e119|comply"); X = S.get(pairs, L20)
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler().fit(X); fs.append((ss, LogisticRegression(C=C_FROZEN, max_iter=3000, class_weight="balanced").fit(ss.transform(X), [r["y"] for r in tr_att])))
    F["comply16"] = lambda k0, x16: np.mean([c.decision_function(s_.transform(x16)) for s_, c in fs], 0)
    dv_rows = [x for k in kdv for x in bydv[k]]; dv16 = [r for r in dv_rows if S.has(r["row_id"], 16)]
    Z = np.c_[F["comply_R0"]([r["row_id"].rsplit("|", 1)[0] for r in dv16], None), F["comply16"](None, S.get([(r["row_id"], 16) for r in dv16], L20))]
    st = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Z, [r["y"] for r in dv16])
    F["r0a"] = lambda k0, x16: np.c_[F["comply_R0"](k0, None), F["comply16"](None, x16)] @ st.coef_[0] + st.intercept_[0]
    def per_input(f0):   # prompt reads: score each input once and copy it to its runs, so batched float noise cannot split a tie
        def f(k0, x16):
            u = sorted(set(k0)); m = dict(zip(u, f0(u, None))); return np.array([m[k] for k in k0])
        return f
    for g in ("exposure_M", "risk_M", "comply_R0"):
        F[g] = per_input(F[g])
    # test
    k0 = [r["row_id"].rsplit("|", 1)[0] for r in T]; x16 = S.get([(r["row_id"], 16) for r in T], L20)
    sc_t = {g: f(k0, x16) for g, f in F.items()}; y = np.array([r["y"] for r in T]); case = np.unique([r["case"] for r in T], return_inverse=True)[1]
    def stats(mult):
        out = {}
        for g, s in sc_t.items():
            w = mult / mult.sum(); d = {}
            for k in KS:
                p = fire_exact(s, w, k); d[f"capture@{int(k * 100)}"] = float((p * y * mult).sum() / max(1e-9, (y * mult).sum()))
            out[g] = d
        return out
    one = stats(np.ones(len(T)))
    rng = np.random.default_rng(0); K = case.max() + 1; draws = [stats(np.bincount(rng.integers(0, K, K), minlength=K)[case].astype(float)) for _ in range(B)]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = dict(tag=a.tag, test=dict(runs=len(T), inputs=int(K), harmful=int(y.sum()), mixed=int(sum(0 < y[case == c].sum() < (case == c).sum() for c in range(K)))),
               selection=sel, capture=one, auroc={}, contrasts={})
    for g, s in sc_t.items():
        res["auroc"][g] = dict(attack_runs=float(roc_auc_score(y, s)))
        mixed = [c for c in range(K) if 0 < y[case == c].sum() < (case == c).sum()]
        res["auroc"][g]["within_input"] = float(np.mean([roc_auc_score(y[case == c], s[case == c]) for c in mixed])) if mixed else None
    for name, g1, g2 in CONTRASTS:
        for k in KS:
            m = f"capture@{int(k * 100)}"
            res["contrasts"][f"{name}|{m}"] = dict(a=g1, b=g2, point=one[g1][m] - one[g2][m], ci=ci([d[g1][m] - d[g2][m] for d in draws]))
    out = HERE / f"results/e119/{a.tag}"; out.mkdir(parents=True, exist_ok=True); (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("test", "capture", "auroc")}, indent=1))
    for k, v in res["contrasts"].items():
        print(f"{k:32s} {v['point']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
