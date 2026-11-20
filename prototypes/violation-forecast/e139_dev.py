"""E139 step 1 (experiments/E139-embedding-lookalike-negatives.md): fit emb_A and its dev diagnostics. No test or E110 row is read.

emb_A = E134's realistic N = 16 recipe (e5-large-v2, StandardScaler + balanced logistic regression, C = 0.003 frozen by E134) fitted on
every realistic train row (label = violates) plus E109's 896 train look-alike runs (e79v2-states/e109/injecforks_train, SHA256SUMS
checked by fetch), all label 0: the analogue of E109's comply_A. Dev diagnostics only (nothing is selected on them).

  uv run --no-project --python 3.12 --with sentence-transformers --with scikit-learn --with huggingface_hub python e139_dev.py
  -> results/e139/dev.json
"""
import hashlib, json
from pathlib import Path

import numpy as np

from e134_common import embed, fit, load, pre, read_scores
from e136_common import gpu_lock
from injecforks_metrics import within_case_mean

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/e139"; SC = HERE / "results/injecforks/scores"
N, C = 16, 0.003
REPO, SUB = "anonymous/e79v2-states", "e109/injecforks_train"
TRAIN_ROOT = HERE / "data" / SUB


def train_root():
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, allow_patterns=f"{SUB}/*", local_dir=HERE / "data")
    for line in (TRAIN_ROOT / "SHA256SUMS").read_text().splitlines():
        h, f = line.split(None, 1)
        assert hashlib.sha256((TRAIN_ROOT / f.lstrip("*")).read_bytes()).hexdigest() == h, ("sha mismatch", f)
    return TRAIN_ROOT


def fit_A():
    """Returns (emb_A classifier, counts)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr = [r for r in load("realistic", "train", subset=None) if r["cut"].get(N) is not None]
    la = [r for r in load("realistic-lookalike", "train", subset=None, root=train_root()) if r["cut"].get(N) is not None]
    lab = {x["row_id"] for x in map(json.loads, open(HERE / "results/e109/part3-train-labels.jsonl"))}
    assert len(la) == 896 and {r["row_id"] for r in la} == lab, "train look-alikes differ from E109 part 3's"
    X = np.vstack([embed(f"realistic-train-N{N}", [pre(r, N) for r in tr]), embed(f"e139-train-lookalike-N{N}", [pre(r, N) for r in la])])
    y = [bool(r["violates"]) for r in tr] + [False] * len(la)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")).fit(X, y)
    return clf, dict(train_rows=len(tr), train_positives=int(sum(y)), lookalike_negatives=len(la))


def sc(clf, name, rows):
    return clf.decision_function(embed(f"{name}-N{N}", [pre(r, N) for r in rows]))


def main():
    from sklearn.metrics import roc_auc_score
    with gpu_lock():
        A, cnt = fit_A(); E = fit("realistic", N, C)
        dev = [r for r in load("realistic", "dev") if r["cut"].get(N) is not None]
        la = [r for r in load("realistic-lookalike", "dev", subset=None) if r["cut"].get(N) is not None]
        sa, se = sc(A, "realistic-dev", dev), sc(E, "realistic-dev", dev); la_a, la_e = sc(A, "e139-dev-lookalike", la), sc(E, "e139-dev-lookalike", la)
    ref = read_scores(SC / "e134-emb-realistic-dev.jsonl")
    dmax = float(max(abs(s - ref[(r["row_id"], N)]) for r, s in zip(dev, se))); assert dmax < 1e-6, dmax
    cs, ys = [r["case"] for r in dev], [r["violates"] for r in dev]
    P = read_scores(SC / "e79final-comply-realistic-dev.jsonl"); P102 = read_scores(SC / "e102-9b-comply-realistic-lookalike-dev.jsonl")
    vi = [i for i, r in enumerate(dev) if r["violates"]]; yl = [1] * len(vi) + [0] * len(la)
    pv = [(dev[i]["row_id"], N) for i in vi]; pl = [(r["row_id"], N) for r in la]
    o = dict(**cnt, C=C, dev_rows=len(dev), dev_lookalikes=len(la), emb_reproduces_e134_dev=dmax,
             within_case_auroc=dict(emb_A=within_case_mean(cs, ys, list(sa)), emb=within_case_mean(cs, ys, list(se)),
                                    comply=within_case_mean(cs, ys, [P[(r["row_id"], N)] for r in dev])),
             violating_vs_lookalike_auroc=dict(emb_A=roc_auc_score(yl, list(sa[vi]) + list(la_a)), emb=roc_auc_score(yl, list(se[vi]) + list(la_e)),
                                               comply=roc_auc_score(yl, [P[k] for k in pv] + [P102[k] for k in pl]) if all(k in P102 for k in pl) else None),
             corr_emb_A_emb_dev=float(np.corrcoef(sa, se)[0, 1]))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "dev.json").write_text(json.dumps(o, indent=1)); print(json.dumps(o, indent=1))


if __name__ == "__main__":
    main()
