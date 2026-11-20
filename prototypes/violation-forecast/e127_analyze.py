"""E127 analysis (experiments/E127-comply-steering.md). Local, CPU.

  ~/.venvs/e01/bin/python e127_analyze.py [--rows data/e127-paper/rows.jsonl] [--tag paper]   -> results/e127/summary-<tag>.{json,md}

Per arm: strict and broad compliance, well-formed (reasoning closed, not capped, a call or a non-empty answer), clean (well-formed,
a non-empty answer to the user, no broad violation), mean output tokens. Every contrast x - y is on the prefixes both arms continued
(case, rollout), averaged per case, then over cases (each case weighted equally); case bootstrap 2,000 draws, seed 127:
  arm - control (strict, well-formed, clean) for every arm; probe -a - random0 -a at a = 2, 4; probe -8 - mean of the random -8 arms.
Check: the control arm against the dataset's strict label on the same rollouts (the prefixes' own continuations).
The pilots' rows (first-token steering, arms p0 / r+-8 ...) also read: their control is p0, and the random contrasts are skipped.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def well_formed(r):
    return r["think_closed"] and r["finish"] == "stop" and bool(r["calls"] or r["text"].strip())


def clean(r):
    return well_formed(r) and bool(r["text"].strip()) and not r["violates_broad"]


MET = {"strict": lambda r: r["violates_strict"], "broad": lambda r: r["violates_broad"], "wf": well_formed, "clean": clean,
       "dataset": lambda r: r.get("dataset_strict_this_rollout", np.nan)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", default=str(HERE / "data/e127-paper/rows.jsonl")); ap.add_argument("--tag", default="paper")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows)]
    arms = list(dict.fromkeys(r["arm"] for r in rows)); cases = sorted({r["case"] for r in rows})
    ctrl = "c" if "c" in arms else "p0"
    V = defaultdict(dict)   # arm -> (case, rollout) -> row
    for r in rows:
        V[r["arm"]][(r["case"], r["rollout"])] = r
    rng = np.random.default_rng(127); B = rng.integers(0, len(cases), (2000, len(cases)))

    def ci(x):
        bs = x[B].mean(1)
        return [round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    def per_case(arm, key, keys=None):
        out = []
        for c in cases:
            ks = [k for k in V[arm] if k[0] == c and (keys is None or k in keys)]
            out.append(np.mean([MET[key](V[arm][k]) for k in ks]))
        return np.array(out, float)

    def diff(x, ys, key="strict"):
        """x minus the mean of arms ys, on the prefixes every one of them continued."""
        common = set(V[x]).intersection(*[set(V[y]) for y in ys])
        return ci(per_case(x, key, common) - np.mean([per_case(y, key, common) for y in ys], 0))

    meta = {arm: next(r for r in rows if r["arm"] == arm) for arm in arms}
    per_arm = {arm: dict(dir=meta[arm]["dir"], alpha=meta[arm]["alpha"], n=len(V[arm]),
                         **{k: round(float(per_case(arm, k).mean()), 3) for k in ("strict", "broad", "wf", "clean")},
                         tokens=round(float(np.mean([r["tokens"] for r in V[arm].values()])), 0)) for arm in arms}
    C = {}
    for arm in arms:
        if arm != ctrl:
            for k, lab in (("strict", "strict"), ("wf", "well-formed"), ("clean", "clean")):
                C[f"{arm} - {ctrl} ({lab})"] = diff(arm, [ctrl], k)
    for s in (2, 4):
        if f"p-{s}" in arms and f"r0-{s}" in arms:
            C[f"p-{s} - r0-{s} (strict)"] = diff(f"p-{s}", [f"r0-{s}"])
    r8 = [x for x in arms if x.startswith("r") and x.endswith("-8") and meta[x]["dir"].startswith("random")]
    if "p-8" in arms and r8:
        C[f"p-8 - mean({', '.join(r8)}) (strict)"] = diff("p-8", r8)
    out = dict(cases=len(cases), rows=len(rows), control=ctrl, per_arm=per_arm, contrasts=C)
    if "dataset_strict_this_rollout" in rows[0]:
        ds = per_case(ctrl, "dataset")
        out["check"] = dict(control_strict=per_arm[ctrl]["strict"], dataset_same_rollouts=round(float(ds.mean()), 3),
                            control_minus_dataset=ci(per_case(ctrl, "strict") - ds))
    D = HERE / "results/e127"; D.mkdir(parents=True, exist_ok=True)
    (D / f"summary-{a.tag}.json").write_text(json.dumps(out, indent=1))
    md = [f"E127 ({a.tag}): {len(cases)} cases, {len(rows)} replies, control {ctrl}", "",
          "| arm | dir | alpha | n | strict | broad | well-formed | clean | tokens |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['dir']} | {v['alpha']} | {v['n']} | {v['strict']} | {v['broad']} | {v['wf']} | {v['clean']} | {v['tokens']:.0f} |"
           for k, v in per_arm.items()]
    md += ["", "| contrast | mean [95 % CI] |", "|---|---|"] + [f"| {k} | {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] |" for k, v in C.items()]
    if "check" in out:
        ck = out["check"]
        md += ["", f"check: control strict {ck['control_strict']} vs the dataset's {ck['dataset_same_rollouts']} on the same rollouts, Δ {ck['control_minus_dataset']}"]
    (D / f"summary-{a.tag}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
