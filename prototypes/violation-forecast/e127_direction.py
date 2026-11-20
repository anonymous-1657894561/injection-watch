"""E127: the steering directions, frozen to a small file before any steered generation (experiments/E127-comply-steering.md).

  ~/.venvs/e01/bin/python e127_direction.py --states data/e79v2/states-train/user-e79v2-9b-pod-20261026T122152Z/states

Refits E79's frozen realistic comply ensemble exactly as injecforks_probe.py --ensemble 5 --fixed e79ens-selection.json does
(layer 20, C = 0.001, train pairs of seeds 0..4), then maps it to the raw residual: the ensemble's decision function is
mean_k(w_k / s_k) . h + const, so its direction in residual units is u = mean_k(w_k / s_k), unit-normalised.
  sigma    SD of realistic train states' projection on u at N = 16 (the strength unit)
  h_norm   median L2 norm of those states (so alpha * sigma can be read as a fraction of the residual)
  random   a Gaussian direction (seed 127) orthogonalised to u, unit norm
  dim      within-case diff-in-means at N = 16 on train: per mixed case, mean(complying states) - mean(other states),
           averaged over cases, unit norm (the usual steering direction, beside the probe's read-out weights)
Check: the raw-space ensemble reproduces the sklearn ensemble's scores on dev (correlation > 0.9999), and its dev case-mean
within-case AUROC is printed beside the frozen selection's.
Output: configs/e127_directions.npz (u, random, dim, sigma, h_norm, layer) + results/e127/direction.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_load import load
from injecforks_metrics import per_case_auroc
from injecforks_probe import States, fit, train_pairs, at_n

FROZEN = HERE / "results/injecforks/e79ens-selection.json"
STORED_LAYERS = [8, 12, 16, 20, 24, 26, 28, 30, 32]   # configs/e79v2-9b-pod.yaml


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--states", required=True); a = ap.parse_args()
    fz = json.load(open(FROZEN))["realistic"]["comply"]["chosen"]
    L, C, seeds = fz["layer"], fz["C"], fz["ensemble_seeds"]
    assert (STORED_LAYERS[L], C, seeds) == (20, 0.001, [0, 1, 2, 3, 4]), (L, C, seeds)
    S = States(a.states)
    tr = load("realistic", "train", subset=None); y = [int(r["violates"]) for r in tr]
    fits = [fit(S, L, train_pairs(tr, S, s, "realistic|comply"), y, C) for s in seeds]
    W = []   # fit() keeps the scaler and classifier in its closure; read them back out
    for f in fits:
        cells = {type(c.cell_contents).__name__: c.cell_contents for c in f.__closure__}
        sc, clf = cells["StandardScaler"], cells["LogisticRegression"]
        W.append((clf.coef_[0] / sc.scale_, clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum()))
    w = np.mean([x for x, _ in W], 0); b = np.mean([x for _, x in W])
    u = w / np.linalg.norm(w)

    dv = load("realistic", "dev"); out = {}
    for n in (8, 16, 32):
        rs = at_n(dv, S, n); P = [(r["row_id"], n) for r in rs]
        ens = np.mean([f(P) for f in fits], 0); raw = S.get(P, L) @ w + b
        assert np.corrcoef(ens, raw)[0, 1] > 0.9999, ("raw-space direction does not reproduce the ensemble", n)
        pc = per_case_auroc([r["case"] for r in rs], [r["violates"] for r in rs], raw)
        out[f"dev_within_case_auroc_N{n}"] = round(float(np.mean(list(pc.values()))), 4)
    X16 = S.get([(r["row_id"], 16) for r in at_n(tr, S, 16)], L)
    proj = X16 @ u; sigma = float(proj.std()); h_norm = float(np.median(np.linalg.norm(X16, axis=1)))
    rng = np.random.default_rng(127); r = rng.standard_normal(len(u)); r -= (r @ u) * u; r /= np.linalg.norm(r)
    rproj = X16 @ r
    R16 = at_n(tr, S, 16); yy = np.array([int(x["violates"]) for x in R16]); cs = np.array([x["case"] for x in R16])
    diffs = [X16[(cs == c) & (yy == 1)].mean(0) - X16[(cs == c) & (yy == 0)].mean(0)
             for c in sorted(set(cs)) if 0 < yy[cs == c].sum() < (cs == c).sum()]
    dim = np.mean(diffs, 0); dim /= np.linalg.norm(dim)
    for n in (8, 16, 32):
        rs = at_n(dv, S, n)
        pc = per_case_auroc([x["case"] for x in rs], [x["violates"] for x in rs], S.get([(x["row_id"], n) for x in rs], L) @ dim)
        out[f"dim_dev_within_case_auroc_N{n}"] = round(float(np.mean(list(pc.values()))), 4)
    np.savez(HERE / "configs/e127_directions.npz", u=u.astype(np.float32), random=r.astype(np.float32), dim=dim.astype(np.float32),
             sigma=np.float32(sigma), h_norm=np.float32(h_norm), layer=np.int32(20))
    out.update(layer=20, C=C, seeds=seeds, train_rows_N16=len(X16), sigma=round(sigma, 4), h_norm=round(h_norm, 2),
               sigma_over_h_norm=round(sigma / h_norm, 5), random_proj_sd=round(float(rproj.std()), 4),
               proj_mean=round(float(proj.mean()), 4), frozen_sel_pooled=fz["sel_pooled"],
               alpha8_over_h_norm=round(8 * sigma / h_norm, 4),
               dim_cases=len(diffs), cos_dim_u=round(float(dim @ u), 4), dim_proj_sd=round(float((X16 @ dim).std()), 4))
    (HERE / "results/e127").mkdir(parents=True, exist_ok=True)
    (HERE / "results/e127/direction.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
