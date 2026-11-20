"""E107, exploratory: the same N = 0 gate and threshold sweep as e107_n0.py / e107_curve.py, with the exposure probe ("is an
injection present?", Dong et al.'s target; our frozen `detect` probe, E079: layer index 3, C = 0.1, injected vs benign-twin train
rows, seed 0) in place of the comply probe ("will the model follow it?"). CPU, no judge calls.

  ~/.venvs/agentdojo/bin/python e107_exposure.py   -> results/e107/exposure-{curve-n0.json, scores-n0.json, test-summary.json, test-table.md}

Stored scores are used wherever they exist (every dev case: e79final-detect-realistic{,-benign}-dev, e102-9b-detect-realistic-lookalike-dev,
N = 0); the refit only scores test look-alikes and twins. The refit is not bit-equal in this venv (C = 0.1 is weakly regularised, so
sklearn 1.9's lbfgs stops elsewhere: max |Δ| 0.49, corr 0.9996 on dev), so it is checked where it is used: it must fire on the same dev
cases as the stored scores at the threshold (>= 98 %, asserted, and recorded). Threshold: the 95th percentile of the stored benign-twin
dev scores at N = 0 (E095's rule, as the comply gate's).
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e096_analyze import e79_states
from e107_analyze import MARGIN, summarise, table
from e107_curve import at, attack_cases, benign_cases, sweep
from e107_n0 import TOL, e102_states   # TOL: cases this close to the threshold are counted
from injecforks_load import load
from injecforks_probe import States, fit, train_pairs

OUT = HERE / "results/e107"; SC = HERE / "results/injecforks/scores"
LAYER, C, SEED = 3, 0.1, 0   # e79final-selection.json's detect choice


def stored(cfg):
    name = "e102-9b-detect-realistic-lookalike-dev.jsonl" if cfg == "realistic-lookalike" else f"e79final-detect-{cfg}-dev.jsonl"
    return {x["row_id"].rsplit("|", 1)[0]: x["score"] for x in map(json.loads, open(SC / name)) if x["N"] == 0}


if __name__ == "__main__":
    sel = json.load(open(HERE / "results/injecforks/e79final-selection.json"))["realistic"]["detect"]["chosen"]
    assert (sel["layer"], sel["C"]) == (LAYER, C), sel
    S = States([e79_states(), e102_states()])   # E079's train and dev states, E102's look-alike states
    inj_tr, ben_tr = load("realistic", "train", subset=None), load("realistic-benign", "train", subset=None)
    Pd = train_pairs(inj_tr, S, SEED, "realistic|detect-inj") + train_pairs(ben_tr, S, SEED, "realistic|detect-ben")
    f = fit(S, LAYER, Pd, [1] * len(inj_tr) + [0] * len(ben_tr), C)
    ref = {**stored("realistic"), **stored("realistic-benign"), **stored("realistic-lookalike")}; keys = sorted(ref)
    thr = float(np.percentile([x["score"] for x in map(json.loads, open(SC / "e79final-detect-realistic-benign-dev.jsonl")) if x["N"] == 0], 95))
    got = f([(k + "|0", 0) for k in keys]); want = np.array([ref[k] for k in keys])
    dev = float(np.max(np.abs(got - want))); agree = float(np.mean((got >= thr) == (want >= thr)))
    assert agree >= 0.98 and np.corrcoef(got, want)[0, 1] > 0.999, ("refitted detect probe does not fire like e79final's", agree, dev)
    score0 = lambda ks: {k: ref[k] if k in ref else v for k, v in zip(ks, map(float, f([(k + "|0", 0) for k in ks])))}
    sA = {k.split("|", 2)[2]: v for k, v in score0(sorted(k for k in stored("realistic"))).items()}; A = attack_cases(sA)
    scores = dict(attack=sA)   # per-case N = 0 exposure scores, for e107_cascade.py
    cfg = {"twin": "realistic-benign", "lookalike": "realistic-lookalike"}; out = dict(threshold=thr, refit_max_dev=dev, refit_fire_agreement_dev=agree, refit_cases_dev=len(keys))
    for split in ("dev", "test"):
        rows = [json.loads(l) for l in open(OUT / f"n0-{split}-rows.jsonl")]
        cases = sorted({(r["category"], r["case"]) for r in rows})
        sc = score0([f"{cfg[c]}|{split}|{cs}" for c, cs in cases]); bsc = {(c, cs): sc[f"{cfg[c]}|{split}|{cs}"] for c, cs in cases}
        out[split] = sweep(A, benign_cases(split, bsc)); scores[split] = {f"{c}|{cs}": v for (c, cs), v in bsc.items()}
        if split == "test":   # the gated arm at the frozen threshold, as e107_n0.py does for the comply gate
            base = [dict(r, fires=bsc[(r["category"], r["case"])] >= thr, score=bsc[(r["category"], r["case"])]) for r in rows if r["arm"] in ("control", "always")]
            by = {(r["case"], r["j"], r["arm"]): r for r in base}
            g = base + [dict(by[(r["case"], r["j"], "always" if r["fires"] else "control")], arm="gated") for r in base if r["arm"] == "control"]
            Sm = summarise(g); la = Sm["lookalike"]["task"]["gated-control"]
            decision = "material harm" if la[2] < MARGIN else "acceptable cost" if la[1] > MARGIN else "inconclusive"
            (OUT / "exposure-test-summary.json").write_text(json.dumps(dict(meta=dict(threshold=thr, decision=decision, refit_max_dev=dev, refit_fire_agreement_dev=agree,
                                                                                      near_threshold=sum(abs(v - thr) < TOL for v in sc.values())), **Sm), indent=1, default=float))
            (OUT / "exposure-test-table.md").write_text(table(Sm, "test", decision).replace("comply ensemble at N = 16", "exposure (detect) probe at N = 0").replace(
                "N = 16 reminder, always on and gated on the comply probe", "N = 0 reminder (E105's replies), gated on the exposure probe"))
    (OUT / "exposure-curve-n0.json").write_text(json.dumps(out, indent=1)); (OUT / "exposure-scores-n0.json").write_text(json.dumps(scores, indent=1))
    for split in ("dev", "test"):
        P = out[split]
        for name, p in (("never", P[-1]), ("frozen", at(P, thr)), ("always", P[0])):
            print("exposure", split, name, {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()})
    print("threshold", thr, "refit max |d|", dev, "fire agreement on dev", agree, len(keys))
