"""E139 step 2 (experiments/E139-embedding-lookalike-negatives.md): emb_A on every run E135 / E136 scored, and its cutoffs. No outcome read.

Takes E136's results/e136/gate/scores.jsonl (E135's runs + ft; every join asserted there) and re-derives each run's source row the same
way, asserting row_id equality. Adds `emb_A` (e139_dev.fit_A). Cutoffs: e135_scores.match_thr, calibrated (E107 test look-alikes) and
pooled, 10 % / 20 %, as E110 / E135 / E136; every earlier gate's cutoff is carried over unchanged.

  uv run --no-project --python 3.12 --with sentence-transformers --with scikit-learn --with huggingface_hub python e139_scores.py
  -> results/e139/{scores.jsonl, thresholds.json}
"""
import json, subprocess
from pathlib import Path

import numpy as np

from e134_common import load, pre
from e135_scores import E110_CFG, e110_root, match_thr
from e136_common import gpu_lock
from e139_dev import N, OUT, fit_A, sc

HERE = Path(__file__).resolve().parent
recs = [json.loads(l) for l in open(HERE / "results/e136/gate/scores.jsonl")]
by_test = {(r["case"], r["rollout"]): r for r in load("realistic", "test", subset=None)}
la107 = {(r["case"], r["rollout"]): r for r in load("realistic-lookalike", "test", subset=None)}
la110 = {(r["case"], r["rollout"]): r for r in load(E110_CFG, "test", subset=None, root=e110_root())}
src = []
for x in recs:
    d = by_test if x["side"] == "attack" else (la110 if x["set"] == "e110" else la107)
    r = d[(x["case"], x["rollout"])]; assert r["row_id"] == x["row_id"], (r["row_id"], x["row_id"]); src.append(r)
assert all(len(pre(r, N)) > 0 for r in src)
with gpu_lock():
    A, cnt = fit_A()
    S = {}
    for name in ("e110-attack", "e110-lookalike", "e109-attack", "e109-lookalike"):
        st, sd = name.split("-"); idx = [i for i, x in enumerate(recs) if x["set"] == st and x["side"] == sd]
        for i, v in zip(idx, sc(A, f"e139-{name}", [src[i] for i in idx])):
            recs[i]["emb_A"] = float(v)
assert all("emb_A" in x for x in recs)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "scores.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
T = json.loads((HERE / "results/e136/gate/thresholds.json").read_text())
cal = [x for x in recs if x["set"] == "e109" and x["side"] == "lookalike"]; pool = [x for x in recs if x["set"] == "e109"]
for kind, X in (("calibrated", cal), ("pooled", pool)):
    for rr in (0.1, 0.2):
        T[kind][str(rr)]["emb_A"] = dict(zip(("threshold", "calibration_rate"), match_thr(np.array([x["emb_A"] for x in X]), np.ones(len(X)), rr)))
T["emb_A"] = f"E139 emb_A realistic N = 16, C = 0.003: {cnt}"
T["git_e139"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
(OUT / "thresholds.json").write_text(json.dumps(T, indent=1))
print(json.dumps({k: {r: T[k][r]["emb_A"] for r in T[k]} for k in ("calibrated", "pooled")}, indent=1), T["emb_A"])
