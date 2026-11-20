"""Training-set-size curve for the comply probe (paper appendix). Run from this directory: python3 trainsize.py

Same model as E144's linear all-N control (layer 3, C = 3e-4, balanced logistic regression on every train prefix N), which
reproduces the published probe within +-0.01 on test. Train on a random fraction of the train *cases* (all replies and N of a
chosen case), score test, within-case AUROC per N (299 mixed cases per arm). 5 seeds per fraction below 1.
"""
import json, sys, types, numpy as np
sys.modules.setdefault("torch", types.ModuleType("torch"))  # common.py imports torch; the linear fit does not use it
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FRACS, SEEDS = [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0], range(5)
X = layer(3)
out = []
for arm in ("realistic", "smoke"):
    tr, te = np.where(mask(arm, "train"))[0], np.where(mask(arm, "test"))[0]
    Xte = X[te].astype(np.float32)
    cases = np.unique(CASE[tr])
    for f in FRACS:
        for seed in (SEEDS if f < 1 else [0]):
            pick = np.random.default_rng(seed).choice(cases, max(2, round(f * len(cases))), replace=False)
            t = tr[np.isin(CASE[tr], pick)]
            if len(set(Y[t])) < 2:
                continue
            sc = StandardScaler().fit(X[t].astype(np.float32))
            clf = LogisticRegression(C=3e-4, max_iter=3000, class_weight="balanced").fit(sc.transform(X[t].astype(np.float32)), Y[t])
            r = per_n(te, clf.decision_function(sc.transform(Xte)))
            out.append(dict(arm=arm, frac=f, seed=seed, cases=len(pick), rows=len(t), **{f"N{n}": r[n][0] for n in NS}))
            print(arm, f"{f:.4f}", seed, len(pick), len(t), round(r[16][0], 3), flush=True)
(RES / "trainsize").mkdir(exist_ok=True)
json.dump(out, open(RES / "trainsize/curve.json", "w"), indent=1)
