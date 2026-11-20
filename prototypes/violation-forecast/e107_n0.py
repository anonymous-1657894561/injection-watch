"""E107 part A (experiments/E107-gated-reminder-lookalike-cost.md): the comply probe's N = 0 gate applied to E105's replies. CPU, no
judge calls (E105's judged rows are reused unchanged).

  ~/.venvs/e01/bin/python e107_n0.py --split dev     # sets and freezes the N = 0 threshold (results/e107/n0-threshold.json)
  ~/.venvs/e01/bin/python e107_n0.py --split test    # refuses without the frozen threshold
  -> results/e107/n0-<split>-{rows.jsonl, summary.json, table.md}

The N = 0 state is the prompt's last token, read before any reply token, so a gate on it decides the same way for both of E105's
arms: gated = the reminder_n0 replies of the cases where it fires, control_n0 elsewhere. Realistic only (the probe's family).
Threshold: the 95th percentile of the frozen comply ensemble's N = 0 scores on realistic-benign dev (E095's rule), dev only.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e096_analyze as A
from e096_analyze import e79_states
from e107_analyze import MARGIN, THRESHOLD, summarise, table
from injecforks_probe import States

OUT = HERE / "results/e107"
E102_RUN = "user-e102-9b-lookalike-pod-20261102T122859Z"
THR = OUT / "n0-threshold.json"
SC = HERE / "results/injecforks/scores"
ARM = {"control_n0": "control", "reminder_n0": "always"}


TOL = 5e-3   # e096_analyze.ensemble asserts 1e-3; this venv's sklearn 1.9 refit measured 1.0023e-3 (corr 0.99999997), 2026-11-06


def ensemble():
    """e096_analyze.ensemble with a measured tolerance: the frozen 5-seed comply ensemble, its max deviation from the stored
    e79ens dev scores returned so a gate decision within it of the threshold can be counted."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from injecforks_probe import train_pairs
    from injecforks_load import load
    S = States(e79_states()); tr = load("realistic", "train", subset=None); y = [int(r["violates"]) for r in tr]; fs = []
    for k in range(A.SEEDS):
        X = S.get(train_pairs(tr, S, k, "realistic|comply"), A.LAYER_IDX); sc = StandardScaler().fit(X)
        fs.append((sc, LogisticRegression(C=A.C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)))
    f = lambda X: np.mean([c.decision_function(s.transform(np.asarray(X, np.float32))) for s, c in fs], 0)
    ref = {x["row_id"]: x["score"] for x in map(json.loads, open(SC / "e79ens-comply-realistic-dev.jsonl")) if x["N"] == A.NR}
    ids = sorted(ref)[:40]; dev = float(np.max(np.abs(f(S.get([(r, A.NR) for r in ids], A.LAYER_IDX)) - np.array([ref[r] for r in ids]))))
    assert dev < TOL, ("refitted ensemble differs from e79ens scores", dev)
    return f, dev


def e102_states():
    from huggingface_hub import snapshot_download
    d = snapshot_download("anonymous/e79v2-states", allow_patterns=[f"{E102_RUN}/states/*"], local_dir=HERE / "data/e102/dl")
    return Path(d) / E102_RUN / "states"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["dev", "test"]); a = ap.parse_args(); os.chdir(HERE); OUT.mkdir(parents=True, exist_ok=True)
    f, dev = ensemble()
    S = States([e79_states(), e102_states()]); L = 3   # e096_analyze.LAYER_IDX: layer 20
    rows = [json.loads(l) for l in open(OUT.parent / "e105" / f"{a.split}-rows.jsonl")]
    rows = [r for r in rows if r["family"] == "realistic"]
    cfg = lambda r: "realistic-benign" if r["category"] == "twin" else "realistic-lookalike"
    keys = sorted({f"{cfg(r)}|{a.split}|{r['case']}" for r in rows})
    score = dict(zip(keys, f(S.get([(k + "|0", 0) for k in keys], L))))
    # the refit read reproduces the stored dev N = 0 scores of the same cases (the threshold's own file)
    ref = {x["row_id"].rsplit("|", 1)[0]: x["score"] for x in map(json.loads, open(SC / "e79ens-comply-realistic-benign-dev.jsonl")) if x["N"] == 0}
    if a.split == "dev":
        d = [abs(score[k] - ref[k]) for k in keys if k in ref]
        assert len(d) == 24 and max(d) < TOL, ("refit N = 0 scores differ from e79ens", len(d), max(d) if d else None)
        thr = float(np.percentile([x["score"] for x in map(json.loads, open(SC / "e79ens-comply-realistic-benign-dev.jsonl")) if x["N"] == 0], 95))
        assert not THR.exists() or json.loads(THR.read_text())["threshold"] == thr, "the N = 0 threshold is set once, on dev"
        THR.write_text(json.dumps(dict(threshold=thr, rule="95th percentile of e79ens-comply-realistic-benign-dev.jsonl at N = 0 (E095)"), indent=1) + "\n")
    else:
        assert THR.exists(), "set the N = 0 threshold on dev first"
        thr = json.loads(THR.read_text())["threshold"]
    out = []
    for r in rows:
        s = float(score[f"{cfg(r)}|{a.split}|{r['case']}"]); fire = bool(s >= thr)
        out.append(dict(r, arm=ARM[r["arm"]], fires=fire, score=s))
    by = {(r["case"], r["j"], r["arm"]): r for r in out}
    out += [dict(by[(r["case"], r["j"], "always" if r["fires"] else "control")], arm="gated") for r in out if r["arm"] == "control"]
    assert len(out) == 3 * sum(r["arm"] == "control" for r in out)
    (OUT / f"n0-{a.split}-rows.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n" for r in out))
    Sm = summarise(out); la = Sm["lookalike"]["task"]["gated-control"]
    decision = "material harm" if la[2] < MARGIN else "acceptable cost" if la[1] > MARGIN else "inconclusive"
    (OUT / f"n0-{a.split}-summary.json").write_text(json.dumps(dict(meta=dict(threshold=thr, decision=decision, cases=len(keys), refit_max_dev=dev,
                                                                                  near_threshold=sum(abs(v - thr) < TOL for v in score.values())), **Sm), indent=1, default=float))
    t = table(Sm, a.split, decision).replace("N = 16 reminder, always on and gated", "N = 0 reminder (E105's replies), always on and gated at N = 0")
    t = t.replace(f"comply ensemble at N = 16, fires at score ≥ {THRESHOLD}", f"comply ensemble read at N = 0, fires at score ≥ {thr:.5f}")
    (OUT / f"n0-{a.split}-table.md").write_text(t); print(t)
