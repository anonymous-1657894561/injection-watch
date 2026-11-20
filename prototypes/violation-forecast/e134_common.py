"""E134 shared pieces: the e5-large-v2 reasoning-prefix embedding and its train-fitted logistic regression.

Text = r["reasoning"][:r["cut"][N]] (the dataset's cut, never re-tokenised), embedded as "query: " + text, normalised, fp16 on
CUDA. Embeddings are cached per (name, N) under E134_CACHE (default: the session scratchpad), keyed by a hash of the texts.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from injecforks_load import load, load_variant  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
SC = HERE / "results/injecforks/scores"
OUT = HERE / "results/e134"
CACHE = Path(os.environ.get("E134_CACHE", HERE / "data/e134-cache"))
NS = [8, 16, 32, 64, 128]
MODEL = "intfloat/e5-large-v2"
_m = None


def pre(r, N):
    return r["reasoning"][:r["cut"][N]]


def embed(name, texts):
    global _m
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    f = CACHE / f"{name}-{h}.npy"
    if f.exists():
        return np.load(f)
    if _m is None:
        from sentence_transformers import SentenceTransformer
        _m = SentenceTransformer(MODEL, device="cuda").half()
    X = _m.encode(["query: " + t for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(f, X)
    return X


def fit(cfg, N, C):
    """The embedding classifier for (config, N): StandardScaler + balanced logistic regression on every train row with a prefix."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr = [r for r in load(cfg, "train", subset=None) if r["cut"].get(N) is not None]
    X = embed(f"{cfg}-train-N{N}", [pre(r, N) for r in tr])
    y = [bool(r["violates"]) for r in tr]
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")).fit(X, y)


def score_rows(clf, name, rows, N):
    """decision_function on the rows with a prefix at N; returns {row_id: score}."""
    rs = [r for r in rows if r["cut"].get(N) is not None]
    if not rs:
        return {}
    return dict(zip([r["row_id"] for r in rs], map(float, clf.decision_function(embed(f"{name}-N{N}", [pre(r, N) for r in rs])))))


def read_scores(path):
    return {(x["row_id"], int(x["N"])): x["score"] for x in map(__import__("json").loads, open(path))}
