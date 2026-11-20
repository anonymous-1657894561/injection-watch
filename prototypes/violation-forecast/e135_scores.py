"""E135 step 1 (experiments/E135-embedding-gate.md): embedding scores and frozen cutoffs. No outcome is read here.

E134's frozen realistic N = 16 embedding classifier (e134_common.fit, C = 0.003, refit on train; deterministic, asserted against
E134's own test score file) on the N = 16 reasoning prefix of every run E135 needs:
  - E110's 493 attack runs and 702 fresh look-alike runs (results/e110/units.jsonl; prefixes from InjecForks realistic test and from
    E110's own release on the Hub, e79v2-states/e110/injecforks_lookalike, SHA256SUMS-e110 checked);
  - E107's 704 test look-alike runs and E109 3b's 699 attack runs (results/e109/audit/units.jsonl), the calibration / pooled sets.
Gates: emb, and comb = a * comply + b * emb (E134 frozen.json realistic N = 16, comply = the run's E110 / E109 score).
Cutoffs: e109_audit.match_thr, as E110's thresholds().

  uv run --no-project --python 3.12 --with sentence-transformers --with scikit-learn --with huggingface_hub python e135_scores.py
  -> results/e135/{scores.jsonl, thresholds.json}
"""
import hashlib, json, subprocess
from pathlib import Path

import numpy as np

from e134_common import fit, load, pre, read_scores, score_rows

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/e135"
N, C = 16, 0.003
E110_REPO, E110_SUB, E110_CFG = "anonymous/e79v2-states", "e110/injecforks_lookalike", "realistic-lookalike-e110"


def match_thr(s, w, r):   # copied verbatim from e109_audit.match_thr (importing e109_audit pulls in the pod-state loaders)
    o = np.argsort(-s, kind="stable"); ss, ww = s[o], w[o]; cw = np.cumsum(ww) / ww.sum()
    last = np.r_[ss[1:] != ss[:-1], True]
    ok = np.where(last & (cw <= r + 1e-12))[0]
    return (float(ss[ok[-1]]), float(cw[ok[-1]])) if len(ok) else (np.inf, 0.0)


def e110_root():
    from huggingface_hub import snapshot_download
    R = Path(snapshot_download(E110_REPO, allow_patterns=f"{E110_SUB}/*")) / E110_SUB
    for line in (R / "SHA256SUMS-e110").read_text().splitlines():
        h, f = line.split("  ", 1)
        assert hashlib.sha256((R / f).read_bytes()).hexdigest() == h, ("sha mismatch", f)
    return R


def main():
    fr = json.loads((HERE / "results/e134/frozen.json").read_text())
    assert fr["C"] == C and fr["probe"] == "e79final-comply"
    a, b = fr["comb"]["realistic"][str(N)]["a"], fr["comb"]["realistic"][str(N)]["b"]
    clf = fit("realistic", N, C)
    # 0. the frozen model reproduces E134's test scores
    test_all = load("realistic", "test", subset=None)
    ref = {k: v for k, v in read_scores(HERE / "results/injecforks/scores/e134-emb-realistic-test.jsonl").items() if k[1] == N}
    ev = load("realistic", "test")   # E134's row order, so its embedding cache is hit: the classifier must match exactly
    assert {(r["row_id"], N) for r in ev if r["cut"].get(N) is not None} == set(ref)
    mine = score_rows(clf, "realistic-test", ev, N)
    dmax = max(abs(mine[r] - ref[(r, N)]) for r in mine); assert dmax < 1e-6, dmax
    # the same rows re-embedded in another batch order: fp16 batch-composition noise, the size of the drift on E135's own runs
    rev = score_rows(clf, "e135-reorder-check", ev[::-1], N)
    dre = max(abs(rev[r] - ref[(r, N)]) for r in rev); assert dre < 0.05, dre
    score_sd = float(np.std(list(mine.values())))
    # 1. source rows
    by_test = {(r["case"], r["rollout"]): r for r in test_all}
    la107 = {(r["case"], r["rollout"]): r for r in load("realistic-lookalike", "test", subset=None)}
    la110 = {(r["case"], r["rollout"]): r for r in load(E110_CFG, "test", subset=None, root=e110_root())}
    U110 = [json.loads(l) for l in open(HERE / "results/e110/units.jsonl")]
    U109 = [json.loads(l) for l in open(HERE / "results/e109/audit/units.jsonl")]
    items = []   # (set, side, case, rollout, comply, source row)
    for u in U110:
        src = (by_test if u["side"] == "attack" else la110)[(u["case"], u["rollout"])]   # KeyError = join failure
        items.append(("e110", u["side"], u["case"], u["rollout"], u["comply"], src))
    for u in U109:
        if u["side"] not in ("attack", "lookalike"):
            continue
        src = (by_test if u["side"] == "attack" else la107)[(u["case"], u["rollout"])]
        assert src["row_id"] == u["row_id"], (src["row_id"], u["row_id"])
        items.append(("e109", u["side"], u["case"], u["rollout"], u["comply"], src))
    from collections import Counter
    cnt = Counter((s, d) for s, d, *_ in items)
    assert cnt == {("e110", "attack"): 493, ("e110", "lookalike"): 702, ("e109", "attack"): 699, ("e109", "lookalike"): 704}, cnt
    assert all(x[5]["cut"].get(N) is not None for x in items), "a run without an N = 16 prefix"
    assert all(len(pre(x[5], N)) > 0 for x in items)
    # 2. scores
    S = {}
    for name in ("e110-attack", "e110-lookalike", "e109-attack", "e109-lookalike"):
        st, sd = name.split("-")
        rows = [x[5] for x in items if x[0] == st and x[1] == sd]
        sc = score_rows(clf, f"e135-{name}", rows, N)
        S.update({(st, rid): v for rid, v in sc.items()})
    recs = []
    for st, sd, cs, ro, comply, src in items:
        e = S[(st, src["row_id"])]
        recs.append(dict(set=st, side=sd, case=cs, rollout=ro, row_id=src["row_id"], comply=comply, emb=e, comb=a * comply + b * e))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    # 3. cutoffs (scores only)
    cal = [r for r in recs if r["set"] == "e109" and r["side"] == "lookalike"]
    pool = [r for r in recs if r["set"] == "e109"]
    out = dict(calibration_set="E107 test look-alikes (176 cases, 704 runs), scores only", pooled_set="E109 3b attack runs (699) + E107 test look-alike runs (704), scores only",
               emb=f"E134 realistic N = 16, C = 0.003, refit on train; reproduces e134-emb-realistic-test.jsonl: max |d| {dmax:.2g} with E134's cached embeddings, {dre:.3f} re-embedded in another batch order (score SD {score_sd:.2f})",
               comb=dict(a=a, b=b, formula="a * comply + b * emb"), git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
               calibrated={}, pooled={})
    for kind, X in (("calibrated", cal), ("pooled", pool)):
        for r in (0.1, 0.2):
            out[kind][str(r)] = {g: dict(zip(("threshold", "calibration_rate"), match_thr(np.array([x[g] for x in X]), np.ones(len(X)), r))) for g in ("emb", "comb", "comply")}
    # comply's cutoffs from E110's own rule must equal E110's thresholds.json
    T110 = json.loads((HERE / "results/e110/thresholds.json").read_text())
    for kind in ("calibrated", "pooled"):
        for r in ("0.1", "0.2"):
            assert abs(out[kind][r]["comply"]["threshold"] - T110[kind][r]["comply"]["threshold"]) < 1e-9, (kind, r)
    (OUT / "thresholds.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
