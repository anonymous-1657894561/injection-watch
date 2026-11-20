"""E136 gate replay, step 1: the frozen FT ensemble (realistic, N = 16) on every run E135 scored, and its cutoffs. No outcome read.

Takes E135's results/e135/scores.jsonl (E110's 493 attack + 702 fresh look-alike runs; E109 3b's 699 attack + E107's 704 look-alike
runs, every join already asserted there) and re-derives each run's source row the same way (InjecForks realistic test, the
realistic-lookalike test config, E110's release on the Hub), asserting row_id equality. Adds `ft` (3-seed ensemble logit
difference). Cutoffs: e135_scores.match_thr (= e109_audit.match_thr), calibrated / pooled at 10 % / 20 %, as E110 / E135.

  python e136_gate_scores.py   -> results/e136/gate/{scores.jsonl, thresholds.json}
"""
import json
import subprocess
from pathlib import Path

import numpy as np

from e135_scores import E110_CFG, e110_root, match_thr
from e136_common import OUT, SC, gpu_lock, load, score_ens

HERE = Path(__file__).resolve().parent
G = OUT / "gate"
N = 16
FZ = json.loads((OUT / "frozen.json").read_text())
recs = [json.loads(l) for l in open(HERE / "results/e135/scores.jsonl")]
by_test = {(r["case"], r["rollout"]): r for r in load("realistic", "test", subset=None)}
la107 = {(r["case"], r["rollout"]): r for r in load("realistic-lookalike", "test", subset=None)}
la110 = {(r["case"], r["rollout"]): r for r in load(E110_CFG, "test", subset=None, root=e110_root())}
src = []
for x in recs:
    d = by_test if x["side"] == "attack" else (la110 if x["set"] == "e110" else la107)
    r = d[(x["case"], x["rollout"])]; assert r["row_id"] == x["row_id"], (r["row_id"], x["row_id"]); src.append(r)
uniq = list({r["row_id"]: r for r in src}.values())
with gpu_lock():
    E = score_ens("realistic", N, FZ["seeds"], uniq)
assert len(E) == len(uniq), "a run without an N = 16 prefix"
# the gate's scores on test eval rows equal the test read's file
ref = {x["row_id"]: x["score"] for x in map(json.loads, open(SC / "e136-ft-realistic-test.jsonl")) if x["N"] == N}
common = [k for k in E if k in ref]
dmax = max(abs(E[k][0] - ref[k]) for k in common)
for x, r in zip(recs, src):
    x["ft"] = E[r["row_id"]][0]
G.mkdir(parents=True, exist_ok=True)
(G / "scores.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
T = json.loads((HERE / "results/e135/thresholds.json").read_text())
cal = [x for x in recs if x["set"] == "e109" and x["side"] == "lookalike"]; pool = [x for x in recs if x["set"] == "e109"]
for kind, X in (("calibrated", cal), ("pooled", pool)):
    for rr in (0.1, 0.2):
        T[kind][str(rr)]["ft"] = dict(zip(("threshold", "calibration_rate"), match_thr(np.array([x["ft"] for x in X]), np.ones(len(X)), rr)))
T["ft"] = (f"E136 FT ensemble realistic N = 16 (seeds {FZ['seeds']}); matches e136-ft-realistic-test.jsonl on {len(common)} shared rows, "
           f"max |d| {dmax:.2g} (batch-composition noise in fp16 only)")
T["git_e136"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
(G / "thresholds.json").write_text(json.dumps(T, indent=1))
print(json.dumps({k: {r: T[k][r]["ft"] for r in T[k]} for k in ("calibrated", "pooled")}, indent=1), T["ft"])
