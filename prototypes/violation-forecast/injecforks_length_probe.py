"""Length-only baseline for E79 (A1, 2026-10-27): does the comply probe forecast anything beyond how long the rollout will reason?

A ridge probe on the same state as the comply probe (stored layer 3 = model layer 20, one random N per train row, the same
seed and pairing tag) is fitted to log(final reasoning tokens) on train and used as a comply score: complying rollouts reason
longer (E79 v2 diagnostics). Scores go out in the probe's format, so injecforks_score.py pairs them with every other method.

  ~/.venvs/e01/bin/python injecforks_length_probe.py --states ~/e79v2/<run_id>/states [--score dev|test] [--out e79v2]
  -> results/injecforks/scores/{out}-length-{config}-{split}.jsonl (injected rows, every N)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from injecforks_load import load
from injecforks_probe import NS, States, at_n, train_pairs

HERE = Path(__file__).resolve().parent
LAYER, ALPHA = 3, 1e4   # the comply probe's layer; ALPHA as the dev diagnostic (e79v2_diag/length_probe.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--root", help="a local copy of InjecForks; default: the Hub at injecforks_load's pinned tag")
    ap.add_argument("--configs", nargs="+", default=["smoke", "realistic"])
    ap.add_argument("--score", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="e79v2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    S = States(a.states); sdir = HERE / "results/injecforks/scores"; sdir.mkdir(parents=True, exist_ok=True)
    for cfg in a.configs:
        tr = load(cfg, "train", subset=None, root=a.root)
        P = train_pairs(tr, S, a.seed, f"{cfg}|comply")
        X = S.get(P, LAYER); sc = StandardScaler().fit(X)
        rg = Ridge(alpha=ALPHA).fit(sc.transform(X), np.log([r["reasoning_tokens"] for r in tr]))
        f = lambda Q: rg.predict(sc.transform(S.get(Q, LAYER)))
        ev = load(cfg, a.score, root=a.root); out = []
        for n in NS:
            rs = at_n(ev, S, n)
            if n == 0:   # one state per case: score once so every rollout of a case ties exactly
                first = {}
                for r in rs:
                    first.setdefault(r["row_id"].rsplit("|", 1)[0], r["row_id"])
                s0 = dict(zip(first, f([(rid, 0) for rid in first.values()])))
                out += [dict(row_id=r["row_id"], N=0, score=float(s0[r["row_id"].rsplit("|", 1)[0]])) for r in rs]
                continue
            out += [dict(row_id=r["row_id"], N=n, score=float(s)) for r, s in zip(rs, f([(r["row_id"], n) for r in rs]))]
        p = sdir / f"{a.out}-length-{cfg}-{a.score}.jsonl"
        p.write_text("".join(json.dumps(x) + "\n" for x in out))
        print(cfg, "wrote", len(out), "scores to", p.name, flush=True)


if __name__ == "__main__":
    main()
