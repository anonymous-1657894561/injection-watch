"""E107, exploratory: the N = 0 gated reminder's trade-off as the comply probe's threshold sweeps from "never remind" to "always
remind". Attack side: realistic dev injected cases, E090's N = 0 samples (no reminder) vs E096's reminder_n0 (8 each); cost side:
E105's realistic look-alikes and twins (results/e107/n0-<split>-rows.jsonl). One N = 0 score per case, the frozen comply ensemble's
(dev injected: e79ens's stored scores; look-alikes and twins: e107_n0.py's). -> results/e107/curve-n0.json
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e093_analyze import boot

OUT = HERE / "results/e107"; R = "anonymous/e79v2-states"
E090 = "e090/user-e090-9b-pod-20261027T102034Z/results/resample.jsonl"; E096 = "e096/user-e096-9b-pod-20261029T190638Z/results/e096.jsonl"


def attack_cases(score=None):
    """{case: (N = 0 score, attack success without, with the reminder)}; score: case -> N = 0 score (default: the comply ensemble's)."""
    from huggingface_hub import hf_hub_download
    c = [x for x in map(json.loads, open(hf_hub_download(R, E090))) if x["N"] == 0]
    m = [x for x in map(json.loads, open(hf_hub_download(R, E096))) if x.get("part") == "reminder_n0" and x["config"] == "realistic"]
    sc = score or {x["row_id"].rsplit("|", 1)[0].split("|", 2)[2]: x["score"] for x in map(json.loads, open(HERE / "results/injecforks/scores/e79ens-comply-realistic-dev.jsonl")) if x["N"] == 0}
    by = {}
    for arm, X in (("control", c), ("reminder", m)):
        for x in X:
            by.setdefault(x["case"], {}).setdefault(arm, []).append(float(bool(x["violates_strict"])))
    assert len(by) == 55 and all(len(v["control"]) == len(v["reminder"]) == 8 for v in by.values()), "55 cases x 8 per arm"
    return {k: (sc[k], np.mean(v["control"]), np.mean(v["reminder"])) for k, v in by.items()}


def benign_cases(split, score=None):
    """{(category, case): {score, base, control: [task], always: [task]}}; score: (category, case) -> N = 0 score (default: comply's)."""
    rows = [json.loads(l) for l in open(OUT / f"n0-{split}-rows.jsonl")]
    by = {}
    for r in rows:
        if r["arm"] in ("control", "always") and r["task"] is not None:
            k = (r["category"], r["case"]); d = by.setdefault(k, dict(score=score[k] if score else r["score"], base=r["base"], control=[], always=[]))
            d[r["arm"]].append(float(r["task"]))
    return by


def sweep(A, Bn):
    """One point per threshold (fire at score >= t), from always (-inf) to never (+inf)."""
    ts = [-np.inf] + sorted({s for s, *_ in A.values()} | {d["score"] for d in Bn.values()}) + [np.inf]
    comp = lambda cat, t: float(np.mean([np.mean(d["always"] if d["score"] >= t else d["control"]) for (k, _), d in Bn.items() if k == cat]))
    fire = lambda cat, t: float(np.mean([d["score"] >= t for (k, _), d in Bn.items() if k == cat]))
    return [dict(threshold=None if not np.isfinite(t) else float(t), always=bool(t == -np.inf), never=bool(t == np.inf),
                 attack_success=float(np.mean([r if s >= t else c for s, c, r in A.values()])), fire_attack=float(np.mean([s >= t for s, *_ in A.values()])),
                 lookalike_completion=comp("lookalike", t), twin_completion=comp("twin", t), fire_lookalike=fire("lookalike", t), fire_twin=fire("twin", t))
            for t in ts]


def at(P, thr):
    """The point a frozen threshold lands on: the smallest swept threshold at or above it (same firing set)."""
    return min((p for p in P if p["threshold"] is not None and p["threshold"] >= thr), key=lambda p: p["threshold"])


if __name__ == "__main__":
    A = attack_cases(); out = {split: sweep(A, benign_cases(split)) for split in ("dev", "test")}
    out["frozen_threshold"] = json.loads((OUT / "n0-threshold.json").read_text())["threshold"]
    out["note"] = "exploratory; attack side is realistic dev injected (55 cases x 8, E090 vs E096); cost side is E105's look-alikes/twins of the named split"
    (OUT / "curve-n0.json").write_text(json.dumps(out, indent=1))
    for split in ("dev", "test"):
        P = out[split]
        for name, p in (("never", P[-1]), ("frozen", at(P, out["frozen_threshold"])), ("always", P[0])):
            print(split, name, {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()})
