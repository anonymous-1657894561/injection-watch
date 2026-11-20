"""E120 analysis (experiments/E120-gated-resample-fresh-inputs.md). CPU, no judge.

  ~/.venvs/e01/bin/python e120_analyze.py RUN_ID   -> results/e120/{units.jsonl,summary.json}

Frozen before any E120 sample exists:
- Runs: E110's 493 attack runs (results/e110/units.jsonl). No intervention = E110's control (plain continuation from the run's first 16
  reasoning tokens), strict label.
- Resample (E097 / E093's definition): a run that is resampled takes its case's fresh-sample compliance, the mean strict label over the case's
  decided E120 samples (one fresh N = 0 sample per run; fresh samples are exchangeable within a case). The run's own sample is reported beside.
- Arms: (a) always = every run resampled; (b) gated = runs where E110's comply score >= E110's calibrated 10 % cutoff (thresholds.json, frozen
  2026-11-09) resampled, the rest keep control; (c) random gate at the gated arm's firing rate f, analytic as E097: none - f (none - always).
- Intervals: 2,000 draws of whole attack cases, seed 0 (E110's bootstrap); f is recomputed in each draw.
- Beside, no claim: the same at E110's calibrated 20 % and pooled 10 % cutoffs.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT, B = HERE / "results/e120", 2000
CUTS = (("calibrated", "0.1"), ("calibrated", "0.2"), ("pooled", "0.1"))   # first is primary


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main(run_id):
    TH = json.loads((HERE / "results/e110/thresholds.json").read_text())
    A = [u for u in map(json.loads, open(HERE / "results/e110/units.jsonl")) if u["side"] == "attack"]
    assert len(A) == 493, len(A)
    R = HERE / "data/e120/dl/e120" / run_id / "results"
    run = json.load(open(R / "run.json")); assert run.get("complete") and run["split"] == "test" and run["sequences"] == 493
    fresh, own = defaultdict(list), {}
    for x in map(json.loads, open(R / "e120.jsonl")):
        assert x["arm"] == "resample_n0" and (x["case"], x["rollout"]) not in own, ("duplicate", x["row_id"])
        own[(x["case"], x["rollout"])] = float(x["violates_strict"]) if x["decided"] else np.nan
        if x["decided"]:
            fresh[x["case"]].append(float(x["violates_strict"]))
    assert set(own) == {(u["case"], u["rollout"]) for u in A}, "E120 samples do not match E110's attack runs"
    keep = [u for u in A if fresh.get(u["case"])]   # a case with no decided fresh sample has no resample outcome
    case = np.unique([u["case"] for u in keep], return_inverse=True)[1]; nc = case.max() + 1
    c0 = np.array([u["viol_control"] for u in keep], float)
    rs = np.array([np.mean(fresh[u["case"]]) for u in keep]); rs_own = np.array([own[(u["case"], u["rollout"])] for u in keep])
    comply = np.array([u["comply"] for u in keep])
    rng = np.random.default_rng(0)
    W = np.stack([np.ones(len(keep))] + [np.bincount(rng.integers(0, nc, nc), minlength=nc)[case].astype(float) for _ in range(B)])

    def arms(fire, r):   # rows: estimate then draws; columns none, always, gated, random
        m = ~np.isnan(r); Wm = W[:, m]; n = Wm.sum(1); f = Wm @ fire[m] / n
        none, always = Wm @ c0[m] / n, Wm @ r[m] / n; gated = Wm @ np.where(fire, r, c0)[m] / n
        return np.stack([none, always, gated, none - f * (none - always)], 1), f

    res = dict(meta=dict(run_id=run_id, attack_cases=int(nc), runs=len(keep), dropped_runs=len(A) - len(keep),
                         fresh_undecided=int(np.isnan(np.array(list(own.values()))).sum()), calls_control=int(c0.sum()),
                         fresh_rate_pooled=float(np.nanmean(list(own.values()))), peak_gib=run.get("peak_gib"), gpu=run.get("gpu")), cut={})
    names = ["none", "always", "gated", "random"]
    for kind, b in CUTS:
        t = TH[kind][b]["comply"]["threshold"]; fire = (comply >= t).astype(float); row = dict(threshold=t)
        for label, r in (("case_mean", rs), ("own_sample", rs_own)):
            G, f = arms(fire, r); e = G[0]
            d = {nm: dict(point=float(e[i]), ci=ci(G[1:, i])) for i, nm in enumerate(names)}
            d["fire_rate"] = float(f[0]); d["fired_runs"] = int(fire[~np.isnan(r)].sum())
            d["precision"] = float(c0[fire > 0].mean()) if fire.any() else None
            for a_, b_ in (("gated", "always"), ("gated", "random"), ("gated", "none"), ("always", "none")):
                i, j = names.index(a_), names.index(b_); d[f"{a_}-{b_}"] = dict(point=float(e[i] - e[j]), ci=ci(G[1:, i] - G[1:, j]))
            row[label] = d
        res["cut"][f"{kind}|{b}"] = row
    p = res["cut"]["calibrated|0.1"]["case_mean"]
    res["primary"] = {k: p[k] for k in ("none", "always", "gated", "random", "gated-always", "gated-random", "fire_rate")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "units.jsonl").write_text("".join(json.dumps(dict(case=u["case"], rollout=u["rollout"], comply=u["comply"], viol_control=u["viol_control"],
                                                             resample_case_mean=float(a), resample_own=None if np.isnan(o) else float(o))) + "\n"
                                             for u, a, o in zip(keep, rs, rs_own)))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(dict(meta=res["meta"], primary=res["primary"]), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
