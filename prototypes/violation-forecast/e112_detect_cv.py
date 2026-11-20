"""E112 follow-up: the 27B detect probe's injected-vs-benign AUROC, held out by user tool (no new rollouts).

E112 reported 0.998 on dev, the same rows it selected layer x C on, and test has no benign twins. This refits the frozen
detect recipe (layer 32, C = 0.1, one train state per row at a seeded N, e112_final's train_pairs) leaving one user tool out
at a time over train + dev (17 tools), and scores the held-out tool's rows at every N. Rows: every realistic injected rollout
(not only mixed cases) and every benign twin. N = 0 is one state per case, so it is scored once per case.
Length baseline: len(tool_response) in characters, unfitted (longer = injected). The InjecForks card says length alone
separates the benign twins at 1.000, so the probe's number means nothing beyond it unless the baseline is lower.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from e112_final import NS, REPO, STORED, jl, rows_of
from injecforks_probe import States, at_n, fit, train_pairs

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/e112/detect-cv.json"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--states", required=True); ap.add_argument("--root")
    a = ap.parse_args()
    if a.root:
        R = Path(a.root)
    else:
        from huggingface_hub import snapshot_download
        R = Path(snapshot_download(REPO, repo_type="dataset"))
    sel = json.load(open(HERE / "results/e112/selection.json"))["detect"]
    L, C = sel["layer"], sel["C"]; assert STORED[L] == 32 and C == 0.1, sel
    S = States(a.states)
    rows, chars = [], {}
    for cfg, y in (("realistic", 1), ("realistic-benign", 0)):
        for sp in ("train", "dev"):
            for r in rows_of(R, cfg, sp, False):
                rows.append(dict(r, y=y))
            for x in jl(R / cfg / f"{sp}.jsonl"):
                chars[f'{cfg}|{sp}|{x["case"]}'] = len(x["tool_response"])
    tools = sorted({r["user_tool"] for r in rows})
    assert all({r["y"] for r in rows if r["user_tool"] == t} == {0, 1} for t in tools), "a tool without both classes"
    print(f"{len(rows)} rows, {len(tools)} tools, {sum(r['y'] for r in rows)} injected", flush=True)

    score = {}   # (row_id, N) -> held-out score
    for i, t in enumerate(tools):
        tr = [r for r in rows if r["user_tool"] != t]; te = [r for r in rows if r["user_tool"] == t]
        P = train_pairs([r for r in tr if r["y"]], S, 0, "realistic|detect-inj") + \
            train_pairs([r for r in tr if not r["y"]], S, 0, "realistic|detect-ben")
        y = [1] * sum(r["y"] for r in tr) + [0] * sum(1 - r["y"] for r in tr)
        f = fit(S, L, P, y, C)
        for n in NS:
            rs = at_n(te, S, n)
            score.update({(r["row_id"], n): float(s) for r, s in zip(rs, f([(r["row_id"], n) for r in rs]))})
        print(f"[{i + 1}/{len(tools)}] {t}: {len(te)} rows held out", flush=True)

    case = lambda rid: rid.rsplit("|", 1)[0]
    res = {}
    for n in NS:
        rs = at_n(rows, S, n)
        if n == 0:
            rs = list({case(r["row_id"]): r for r in rs}.values())
        yv = [r["y"] for r in rs]; s = [score[(r["row_id"], n)] for r in rs]
        per_tool = {}
        for t in tools:
            ix = [k for k, r in enumerate(rs) if r["user_tool"] == t]
            if len({yv[k] for k in ix}) == 2:
                per_tool[t] = roc_auc_score([yv[k] for k in ix], [s[k] for k in ix])
        res[n] = dict(rows=len(rs), injected=sum(yv), probe=roc_auc_score(yv, s),
                      length=roc_auc_score(yv, [chars[case(r["row_id"])] for r in rs]),
                      probe_min_tool=min(per_tool.values()), probe_min_tool_name=min(per_tool, key=per_tool.get))
        print(f"N={n}: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res[n].items()}), flush=True)
    OUT.write_text(json.dumps(dict(layer=STORED[L], C=C, folds="leave one user tool out", tools=tools, by_n=res), indent=1))


if __name__ == "__main__":
    main()
