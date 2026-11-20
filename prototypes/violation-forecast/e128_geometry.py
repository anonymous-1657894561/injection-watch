"""E128: how close are E79's comply and detect directions? CPU. Pre-registration: experiments/E128-comply-detect-geometry.md.

  python e128_geometry.py --states ~/e79v2/<run>/states --split dev
  python e128_geometry.py --states ... --split test --final      # once, pre-registered
  -> results/e128/<split>.json

Per config and stored layer, fit on train with E79's recipe (injecforks_probe.fit's form: standardised logistic, balanced,
one seeded N per row): comply (C 0.001, train-pair seed 0 and 1) and detect (C 0.01, seed 0). Geometry is train-only:
  cos_lr      comply vs detect weights in raw residual space (w / scale)
  cos_seed    comply seed 0 vs comply seed 1 (what "the same direction" looks like at this sample size)
  cos_dm      class-mean differences (violates - not; injected - benign)
Functional, on the eval rows of --split at N in {8, 16, 32}:
  auc_comply, auc_detect   case-mean within-case AUROC
  auc_perp                 comply refit on train states with the detect direction projected out, scored on projected states
  d_perp                   auc_perp - auc_comply, paired per case, case bootstrap 1,000 (seed 128)
  corr                     Pearson correlation of comply and detect scores
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from injecforks_load import load
from injecforks_metrics import per_case_auroc
from injecforks_probe import States, at_n, train_pairs

HERE = Path(__file__).resolve().parent
LAYERS = [8, 12, 16, 20, 24, 26, 28, 30, 32]   # configs/e79v2-9b-pod.yaml, asserted below
NS = [8, 16, 32]
C_COMPLY, C_DETECT = 0.001, 0.01               # E79's chosen C (results/injecforks/e79final-selection.json)


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def lr(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    return clf.coef_[0] / sc.scale_, lambda Z: clf.decision_function(sc.transform(Z))


def boot(d, B=1000, seed=128):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    assert a.split == "dev" or a.final, "test is read once, pre-registered: pass --final"
    import yaml
    assert yaml.safe_load(open(HERE / "configs/e79v2-9b-pod.yaml"))["layers"] == LAYERS
    S = States(a.states)
    out = dict(split=a.split, commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
               states=str(a.states), rows={})
    for cfg in ["realistic", "smoke"]:
        inj, ben = load(cfg, "train", subset=None), load(cfg + "-benign", "train", subset=None)
        ev = load(cfg, a.split)
        y = np.array([int(r["violates"]) for r in inj]); yd = np.array([1] * len(inj) + [0] * len(ben))
        P0, P1 = train_pairs(inj, S, 0, f"{cfg}|comply"), train_pairs(inj, S, 1, f"{cfg}|comply")
        Pd = train_pairs(inj, S, 0, f"{cfg}|detect-inj") + train_pairs(ben, S, 0, f"{cfg}|detect-ben")
        evn = {n: at_n(ev, S, n) for n in NS}
        for li, L in enumerate(LAYERS):
            Xc, Xc1, Xd = S.get(P0, li), S.get(P1, li), S.get(Pd, li)
            wc, fc = lr(Xc, y, C_COMPLY); wc1, _ = lr(Xc1, y, C_COMPLY); wd, fd = lr(Xd, yd, C_DETECT)
            u = wd / np.linalg.norm(wd)
            perp = lambda X: X - np.outer(X @ u, u)
            _, fp = lr(perp(Xc), y, C_COMPLY)
            r = dict(L=L, cos_lr=cos(wc, wd), cos_seed=cos(wc, wc1),
                     cos_dm=cos(Xc[y == 1].mean(0) - Xc[y == 0].mean(0), Xd[yd == 1].mean(0) - Xd[yd == 0].mean(0)), N={})
            for n, rs in evn.items():
                Xv = S.get([(q["row_id"], n) for q in rs], li)
                case, lab = [q["case"] for q in rs], [q["violates"] for q in rs]
                sc, sd, sp = fc(Xv), fd(Xv), fp(perp(Xv))
                pc, pdt, pp = (per_case_auroc(case, lab, s) for s in (sc, sd, sp))
                r["N"][n] = dict(rows=len(rs), mixed=len(pc), auc_comply=float(np.mean(list(pc.values()))),
                                 auc_detect=float(np.mean(list(pdt.values()))), auc_perp=float(np.mean(list(pp.values()))),
                                 d_perp=boot([pp[k] - pc[k] for k in pc]), corr=float(np.corrcoef(sc, sd)[0, 1]))
            out["rows"].setdefault(cfg, []).append(r)
            n16 = r["N"][16]
            print(f"{cfg} L{L}: cos {r['cos_lr']:.3f} (seed {r['cos_seed']:.3f}, means {r['cos_dm']:.3f}) | N16 comply "
                  f"{n16['auc_comply']:.3f} detect {n16['auc_detect']:.3f} perp-d {n16['d_perp'][0]:+.3f} "
                  f"[{n16['d_perp'][1]:+.3f}, {n16['d_perp'][2]:+.3f}] corr {n16['corr']:.2f}", flush=True)
    d = HERE / "results/e128"; d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{a.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
