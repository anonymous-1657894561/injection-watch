"""E137: does a text reader of the reasoning (E134's e5 embedding) transfer from Qwen3.5-9B to Qwen3.5-27B? CPU + a small GPU.

  python e137.py dev    -> results/e137/{dev.json, frozen.json}; selects C for the 27B-fit embedding on the 27B dev
  python e137.py test   -> results/e137/test.json, scores/e137-{emb27,emb9}-27b-test.jsonl, reverse.json; reads frozen.json

Rows: E112's own 27B rollouts of the InjecForks realistic arm (Hub repo e112-27b-rollouts, labels-v1.2 = data/e112/labels),
e112_final.rows_of's join: train = every row, dev / test = every rollout of the cases mixed under the label (E112's scored rows).
Text: e134_common's, unchanged ("query: " + reasoning[:cut[N]], e5-large-v2, fp16).
  emb9   E134's realistic classifier per N, refit from the 9B's InjecForks train (C = 0.003, results/e134/frozen.json); asserted
         to reproduce scores/e134-emb-realistic-test.jsonl.
  emb27  the same recipe fit on the 27B's train rows; C chosen on the 27B dev from E134's grid by the mean over N of the
         case-mean within-case AUROC.
Scoring: E112's estimator (e79_amended): case-mean within-case AUROC on identical rows (every method scored), two-way
bootstrap (cases x rollout numbers, 2,000 draws, seed 0), paired deltas; the case-only bootstrap beside it.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import e79_amended as E
from e134_common import SC as SC9, embed, fit as fit9, load, pre

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/e137"; SC27 = HERE / "results/e112/scores"
REPO, REV = "anonymous/e112-27b-rollouts", "a7ccbb5ee4a3b915950507b8b128cd55e2172fee"
NS = [8, 16, 32, 64, 128]; GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]; K = 8
C9 = json.loads((HERE / "results/e134/frozen.json").read_text())["C"]


def jl(p):
    return [json.loads(l) for l in open(p)]


def rows27(sp, mixed_only):
    from huggingface_hub import snapshot_download
    R = Path(snapshot_download(REPO, repo_type="dataset", revision=REV))
    lab = {r["row_id"]: r for r in jl(R / "labels-v1.2/realistic" / f"{sp}.jsonl")}
    cut = {(c["case"], c["rollout"]): c for c in jl(R / "cuts/realistic" / f"{sp}.jsonl")}
    out = []
    for x in jl(R / "realistic" / f"{sp}.jsonl"):
        rid = f'realistic|{sp}|{x["case"]}|{x["rollout"]}'; c = cut[(x["case"], x["rollout"])]
        out.append(dict(row_id=rid, case=x["case"], rollout=x["rollout"], user_tool=x["user_tool"], violates=lab[rid]["violates"],
                        reasoning=x["reasoning"], cut={int(n): o for n, o in c["cut"].items()}))
    assert len(out) == len(lab) == len(cut) == len({r["row_id"] for r in out}), sp
    by = defaultdict(list)
    for r in out:
        by[r["case"]].append(r)
    assert all(len(v) == K for v in by.values())
    if mixed_only:
        out = [r for v in by.values() if 0 < sum(x["violates"] for x in v) < K for r in v]
    return out


def clf_scores(clf, name, rows, N):
    rs = [r for r in rows if r["cut"].get(N) is not None]
    X = embed(f"{name}-N{N}", [pre(r, N) for r in rs])
    return {(r["row_id"], N): float(s) for r, s in zip(rs, clf.decision_function(X))}


def fit27(tr, N, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rs = [r for r in tr if r["cut"].get(N) is not None]
    X = embed(f"e137-27b-train-N{N}", [pre(r, N) for r in rs])
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")).fit(X, [bool(r["violates"]) for r in rs])


def table(rows, N, M, pairs):
    """E112's table at N: identical rows, two-way and case-only bootstraps, paired deltas."""
    meth = {k: v for k, v in M.items() if any(kk[1] == N for kk in v)}
    T, missing = E.table(rows, N, meth)
    out = dict(rows=len(T["rows"]), violating_share=round(float(T["y"].mean()), 3),
               mixed_cases=int(sum(0 < T["y"][T["case"] == c].sum() < (T["case"] == c).sum() for c in np.unique(T["case"]))),
               missing=missing)
    for tag, two in (("two_way", True), ("case_only", False)):
        W = E.weights(T["case"], T["roll"], two_way=two); D = E.draws(T, W); E.assert_point(T, D)
        out[tag] = dict(auroc={m: E.summ(v) for m, v in D.items()},
                        delta={f"{a} - {b}": E.summ(D[a] - D[b]) for a, b in pairs if a in D and b in D})
    return out


def write(path, S):
    path.write_text("".join(json.dumps(dict(row_id=r, N=n, score=v)) + "\n" for (r, n), v in sorted(S.items())))


def main(stage):
    OUT.mkdir(parents=True, exist_ok=True)
    tr = rows27("train", False)
    clf9 = {N: fit9("realistic", N, C9) for N in NS}
    if stage == "dev":
        # emb9 reproduces E134's realistic test scores (the 9B test, already read by E134; nothing new is read)
        ref = E.read_scores(str(SC9 / "e134-emb-realistic-test.jsonl")); t9 = load("realistic", "test")
        for N in NS:
            got = clf_scores(clf9[N], "realistic-test", t9, N)
            d = max(abs(got[k] - ref[k]) for k in got); assert len(got) and d < 1e-4, (N, d)
            print(f"emb9 reproduces E134 at N={N}: {len(got)} rows, max |diff| {d:.1e}", flush=True)
        dev = rows27("dev", True)
        probe = E.read_scores(str(SC27 / "probe-dev.jsonl")); length = E.read_scores(str(SC27 / "length-dev.jsonl"))
        assert {r["row_id"] for r in dev} == {k[0] for k in probe}, "dev rows differ from E112's"
        grid = {}
        for C in GRID:
            S = {}
            for N in NS:
                S.update(clf_scores(fit27(tr, N, C), "e137-27b-dev", dev, N))
            per = {N: E.within_case_mean(*zip(*[(r["case"], r["violates"], S[(r["row_id"], N)]) for r in dev if (r["row_id"], N) in S])) for N in NS}
            grid[C] = dict(mean=round(float(np.mean(list(per.values()))), 4), per_n={N: round(float(v), 4) for N, v in per.items()})
            print(C, grid[C], flush=True)
        Cb = max(GRID, key=lambda c: grid[c]["mean"])
        (OUT / "frozen.json").write_text(json.dumps(dict(C27=Cb, C9=C9, grid=grid, rule="max mean over N of the 27B dev case-mean within-case AUROC"), indent=1))
        M = dict(probe=probe, length=length, emb27={}, emb9={})
        for N in NS:
            M["emb27"].update(clf_scores(fit27(tr, N, Cb), "e137-27b-dev", dev, N))
            M["emb9"].update(clf_scores(clf9[N], "e137-27b-dev", dev, N))
        pairs = [("probe", "emb27"), ("emb27", "emb9"), ("probe", "emb9")]
        res = {N: table(dev, N, M, pairs) for N in NS}
        (OUT / "dev.json").write_text(json.dumps(dict(C27=Cb, per_n=res), indent=1))
        for N, r in res.items():
            print(N, r["rows"], {m: v["est"] for m, v in r["two_way"]["auroc"].items()}, {k: v for k, v in r["two_way"]["delta"].items()}, flush=True)
        return
    # test: the one pre-registered read of the embeddings on the 27B test
    Cb = json.loads((OUT / "frozen.json").read_text())["C27"]
    te = rows27("test", True)
    M = dict(probe=E.read_scores(str(SC27 / "probe-test.jsonl")), length=E.read_scores(str(SC27 / "length-test.jsonl")),
             monitor=E.read_scores(str(SC27 / "monitor-test-gpt61sol-low-v1.jsonl")), emb27={}, emb9={})
    assert {r["row_id"] for r in te} == {k[0] for k in M["probe"]} and len(te) == 2328, "test rows differ from E112's"
    clf27 = {N: fit27(tr, N, Cb) for N in NS}
    for N in NS:
        M["emb27"].update(clf_scores(clf27[N], "e137-27b-test", te, N))
        M["emb9"].update(clf_scores(clf9[N], "e137-27b-test", te, N))
    write(OUT / "e137-emb27-27b-test.jsonl", M["emb27"]); write(OUT / "e137-emb9-27b-test.jsonl", M["emb9"])
    pairs = [("probe", "emb27"), ("emb27", "emb9"), ("monitor", "emb27"), ("probe", "emb9"), ("probe", "monitor"), ("emb27", "length")]
    res = {N: table(te, N, M, pairs) for N in NS}
    # Sol-free tables at N <= 32 too, so the embedding / probe numbers on every eligible row are reported beside
    res_nomon = {N: table(te, N, {k: v for k, v in M.items() if k != "monitor"}, pairs) for N in (8, 16, 32)}
    (OUT / "test.json").write_text(json.dumps(dict(C27=Cb, per_n=res, per_n_without_monitor=res_nomon), indent=1))
    for N, r in res.items():
        print(N, r["rows"], r["missing"], {m: v["est"] for m, v in r["two_way"]["auroc"].items()}, flush=True)
        for k, v in r["two_way"]["delta"].items():
            print("   ", k, v, "| case-only", r["case_only"]["delta"][k], flush=True)
    # reverse transfer: the 27B-fit classifier on the 9B's realistic test (eval-v1.2 rows), vs the 9B's own embedding and probe,
    # E134's estimator (case-mean within-case AUROC, case bootstrap 2,000, seed 0) on identical rows
    t9 = load("realistic", "test")
    M9 = dict(emb9=E.read_scores(str(SC9 / "e134-emb-realistic-test.jsonl")), probe9=E.read_scores(str(SC9 / "e79final-comply-realistic-test.jsonl")),
              emb27={})
    for N in NS:
        M9["emb27"].update(clf_scores(clf27[N], "realistic-test", t9, N))
    write(OUT / "e137-emb27-9b-test.jsonl", M9["emb27"])
    rev = {N: table(t9, N, M9, [("emb9", "emb27"), ("probe9", "emb27")]) for N in NS}
    (OUT / "reverse.json").write_text(json.dumps(rev, indent=1))
    for N, r in rev.items():
        print("rev", N, r["rows"], {m: v["est"] for m, v in r["case_only"]["auroc"].items()}, r["case_only"]["delta"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
