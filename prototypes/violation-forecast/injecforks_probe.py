"""E79 v2 probes on InjecForks v1.2 states (injecforks_states.py). CPU. Pre-registration: experiments/E79-within-prompt-prefix-probes.md.

  python injecforks_probe.py --states data/e79v2/states [--configs smoke realistic] [--score dev|test] [--out NAME]
  -> results/injecforks/scores/<NAME>-{comply,detect}-<config>-<split>.jsonl  ({row_id, N, score}) + results/injecforks/<NAME>-selection.json

One probe per config and target, across all N: each training row contributes its state at one N drawn uniformly (seeded) from the
N it has (N = 0 is the case's last prompt token, shared by its rollouts). Standardised logistic regression, balanced class weights.
  comply  injected train rows, label = labels-v1.2 `violates`.
          Selected on dev (eval-v1.2 rows): layer x C maximising case-mean within-case AUROC averaged over N in {8, 16, 32, 64},
          with the mixed cases of every --config pooled (each case weighted equally; realistic dev alone has 55).
  detect  injected (1) vs benign-twin (0) train rows of the config.
          Selected on dev: layer x C maximising across-row AUROC (injected vs benign eval rows) averaged over N in {0, 8, 16, 32, 64}
          and over the --configs.
One layer x C per target is shared by the configs (A1, 2026-10-26); the probes themselves are fitted per config.
With --ensemble K the scored comply probe is the mean decision function of K probes at the chosen layer x C, fitted on the
train pairs of seeds seed .. seed + K - 1 (selection still uses --seed alone). E79's headline uses K = 5 (decisions/, 2026-10-28).
With --fixed SELECTION_JSON nothing is selected: the file's chosen layer x C are fitted as they are (the test read, 2026-10-29).
Both are then scored at every N on the eval rows of --score (dev, or test once, as pre-registered), injected and benign.
Scoring is injecforks_score.py's job; this script only fits, selects on dev and writes scores.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from injecforks_load import load, load_variant
from injecforks_metrics import per_case_auroc

HERE = Path(__file__).resolve().parent
NS = [0, 8, 16, 32, 64, 128]
SEL_COMPLY, SEL_DETECT = [8, 16, 32, 64], [0, 8, 16, 32, 64]
CS = [0.001, 0.01, 0.1]


class States:
    """(row_id, N) -> state [layers, d]; N = 0 is looked up by the row's case (config|split|case)."""

    def __init__(self, ds):
        self.parts, self.idx = [], {}
        ds = [ds] if isinstance(ds, (str, Path)) else ds   # several directories: e.g. E79 v2's states plus E102's look-alike states
        for i, p in enumerate(p for d in ds for p in sorted(Path(d).glob("states-*.npy"))):
            X = np.load(p, mmap_mode="r")
            meta = [json.loads(l) for l in open(p.with_name(p.name.replace("states-", "meta-").replace(".npy", ".jsonl")))]
            assert len(meta) == len(X), p
            self.parts.append(X)
            for j, m in enumerate(meta):
                key = (m["case"], 0) if m["kind"] == "c0" else (m["row_id"], m["N"])
                if m["kind"] == "c0" and key in self.idx:   # a case's prompt state read again by a later run (E113): keep the first
                    continue
                assert key not in self.idx, ("duplicate state", key)
                self.idx[key] = (i, j)
        self.n_layers = self.parts[0].shape[1]

    def key(self, row_id, n):
        return (row_id.rsplit("|", 1)[0], 0) if n == 0 else (row_id, n)

    def has(self, row_id, n):
        return self.key(row_id, n) in self.idx

    def get(self, pairs, layer):
        out = np.empty((len(pairs), self.parts[0].shape[2]), np.float32)
        for k, (r, n) in enumerate(pairs):
            i, j = self.idx[self.key(r, n)]
            out[k] = self.parts[i][j, layer]
        return out


def available(rows, S):
    return {r["row_id"]: [n for n in NS if (n == 0 or r["cut"].get(n) is not None) and S.has(r["row_id"], n)] for r in rows}


def fit(S, layer, pairs, y, C):
    X = S.get(pairs, layer)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    return lambda P: clf.decision_function(sc.transform(S.get(P, layer))) if P else np.array([])


def train_pairs(rows, S, seed, tag):
    rng = random.Random(f"{seed}|{tag}"); av = available(rows, S)
    missing = [r["row_id"] for r in rows if not av[r["row_id"]]]
    assert not missing, ("rows without any state", len(missing), missing[:3])
    return [(r["row_id"], rng.choice(av[r["row_id"]])) for r in rows]


def at_n(rows, S, n):
    el = [r for r in rows if n == 0 or r["cut"].get(n) is not None]
    miss = [r["row_id"] for r in el if not S.has(r["row_id"], n)]
    assert not miss, (f"{len(miss)} eligible rows have no state at N={n} (partial extraction?)", miss[:3])
    return el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, nargs="+")
    ap.add_argument("--lookalike", action="store_true", help="also score the E102 look-alike eval rows (InjecForks v1.4)")
    ap.add_argument("--e103", nargs="*", default=[], help="realistic only: also score E103 variant rows of --score (dev T2 T4 T5, v1.5; test T2 T4, v1.6)")
    ap.add_argument("--root", help="a local copy of InjecForks; default: the Hub at injecforks_load's pinned tag")
    ap.add_argument("--evalextra", action="store_true", help="test only: also score the eval cases' other rollouts (E113)")
    ap.add_argument("--configs", nargs="+", default=["smoke", "realistic"])
    ap.add_argument("--score", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ensemble", type=int, default=1, help="comply probes averaged at the chosen layer x C (train-pair seeds)")
    ap.add_argument("--layers", nargs="*", type=int, help="layer indices into the stored layers (default all)")
    ap.add_argument("--fixed", help="a <NAME>-selection.json whose chosen comply and detect layer x C are used as they are: no grid, "
                                    "no re-selection (E79 test read, decisions/ 2026-10-29)")
    a = ap.parse_args()
    S = States(a.states)
    fixed = json.load(open(a.fixed)) if a.fixed else None
    layers = [] if fixed else a.layers if a.layers else list(range(S.n_layers))
    sdir = HERE / "results/injecforks/scores"; sdir.mkdir(parents=True, exist_ok=True)
    report, D = {}, {}
    for cfg in a.configs:   # fit every layer x C per config; record dev per N (and the mixed-case count, for pooling)
        inj_tr = load(cfg, "train", subset=None, root=a.root); ben_tr = load(cfg + "-benign", "train", subset=None, root=a.root)
        inj_dv = load(cfg, "dev", root=a.root); ben_dv = load(cfg + "-benign", "dev", root=a.root)
        P = train_pairs(inj_tr, S, a.seed, f"{cfg}|comply"); y = [int(r["violates"]) for r in inj_tr]
        Pd = train_pairs(inj_tr, S, a.seed, f"{cfg}|detect-inj") + train_pairs(ben_tr, S, a.seed, f"{cfg}|detect-ben")
        yd = [1] * len(inj_tr) + [0] * len(ben_tr)
        D[cfg] = dict(inj_tr=inj_tr, P=P, y=y, Pd=Pd, yd=yd, inj_dv=inj_dv, ben_dv=ben_dv, comply={}, detect={},
                      n_train=dict(injected=len(inj_tr), benign=len(ben_tr), positives=sum(y)))
        for L in layers:
            for C in CS:
                f = fit(S, L, P, y, C); per_n = {}
                for n in SEL_COMPLY:
                    rs = at_n(inj_dv, S, n)
                    pc = per_case_auroc([r["case"] for r in rs], [r["violates"] for r in rs], f([(r["row_id"], n) for r in rs]))
                    per_n[n] = (float(np.mean(list(pc.values()))) if pc else float("nan"), len(pc))
                D[cfg]["comply"][(L, C)] = per_n
                print(f"{cfg} comply L{L} C={C}:", {n: round(v, 3) for n, (v, _) in per_n.items()}, flush=True)
                f = fit(S, L, Pd, yd, C); per_n = {}
                for n in SEL_DETECT:
                    ri, rb = at_n(inj_dv, S, n), at_n(ben_dv, S, n)
                    sc = f([(r["row_id"], n) for r in ri + rb])
                    per_n[n] = float(roc_auc_score([1] * len(ri) + [0] * len(rb), sc)) if ri and rb else float("nan")
                D[cfg]["detect"][(L, C)] = per_n
                print(f"{cfg} detect L{L} C={C}: {np.nanmean(list(per_n.values())):.3f}", flush=True)
    keys = [(L, C) for L in layers for C in CS]

    def pooled(k):   # case-mean over every config's mixed cases at each N, then the mean over N
        out = []
        for n in SEL_COMPLY:
            v = [(m, c) for cfg in a.configs for m, c in [D[cfg]["comply"][k][n]] if c]
            out.append(sum(m * c for m, c in v) / sum(c for _, c in v) if v else float("nan"))
        return float(np.nanmean(out))
    sel_c = {k: pooled(k) for k in keys}
    sel_d = {k: float(np.nanmean([v for cfg in a.configs for v in D[cfg]["detect"][k].values()])) for k in keys}
    if fixed:   # the frozen choice, read rather than re-made; one layer x C per target is shared by the configs
        ch = {t: {(fixed[cfg][t]["chosen"]["layer"], fixed[cfg][t]["chosen"]["C"]) for cfg in a.configs} for t in ("comply", "detect")}
        assert all(len(v) == 1 for v in ch.values()), ("configs disagree in", a.fixed, ch)
        best_c, best_d = ch["comply"].pop(), ch["detect"].pop()
        sel_c[best_c] = fixed[a.configs[0]]["comply"]["chosen"]["sel_pooled"]; sel_d[best_d] = fixed[a.configs[0]]["detect"]["chosen"]["sel_pooled"]
    else:
        best_c, best_d = max(keys, key=sel_c.get), max(keys, key=sel_d.get)
    print("chosen: comply L%d C=%g (dev %.3f) | detect L%d C=%g (dev %.3f)" % (*best_c, sel_c[best_c], *best_d, sel_d[best_d]), flush=True)
    for cfg in a.configs:
        d = D[cfg]
        cs = [fit(S, best_c[0], d["P"] if k == 0 else train_pairs(d["inj_tr"], S, a.seed + k, f"{cfg}|comply"), d["y"], best_c[1])
              for k in range(a.ensemble)]
        comply = lambda P, cs=cs: np.mean([f(P) for f in cs], axis=0)
        detect = fit(S, best_d[0], d["Pd"], d["yd"], best_d[1])
        report[cfg] = dict(comply=dict(chosen=dict(layer=best_c[0], C=best_c[1], sel_pooled=sel_c[best_c], ensemble_seeds=[a.seed + k for k in range(a.ensemble)]),
                                       grid=[dict(layer=L, C=C, sel_pooled=sel_c[(L, C)], per_n={n: m for n, (m, _) in d["comply"][(L, C)].items()},
                                                  mixed_cases={n: c for n, (_, c) in d["comply"][(L, C)].items()}) for L, C in keys]),
                           detect=dict(chosen=dict(layer=best_d[0], C=best_d[1], sel_pooled=sel_d[best_d]),
                                       grid=[dict(layer=L, C=C, sel_pooled=sel_d[(L, C)], per_n=d["detect"][(L, C)]) for L, C in keys]),
                           n_train=d["n_train"], fixed_from=a.fixed)
        # scores on the eval rows of --score, injected and benign, every N
        ev = (d["inj_dv"] + d["ben_dv"]) if a.score == "dev" else (load(cfg, "test", root=a.root) + load(cfg + "-benign", "test", root=a.root))
        if a.lookalike:   # their own Hub tag (v1.4) unless --root holds them
            ev += load(cfg + "-lookalike", a.score, root=a.root if a.root and (Path(a.root) / f"{cfg}-lookalike").exists() else None)
        if a.evalextra and cfg == "realistic":   # E113: the test eval cases' other rollouts
            assert a.score == "test"
            extra = load(cfg, "test", subset="evalextra", root=a.root)   # states at N = 0, 8, 16 only (E113's pod)
        else:
            extra = []
        if a.e103 and cfg == "realistic":   # exploratory (E103): labelled by the rollouts' own broad column
            vr = [r for v in a.e103 for r in load_variant(f"realistic-e103{v}", a.score)]
            if a.score == "test":   # the test variant pod read N = 8, 16 only (configs/e103-9b-variants-test-pod.yaml), as E113's
                extra += vr
            else:
                ev += vr
        for name, f in (("comply", comply), ("detect", detect)):
            out = []
            for n in NS:
                rs = at_n(ev, S, n) + (at_n(extra, S, n) if n in (0, 8, 16) else [])
                if n == 0:   # one state per case: score it once, so float noise from batch position cannot break the tie
                    first = {}
                    for r in rs:
                        first.setdefault(r["row_id"].rsplit("|", 1)[0], r["row_id"])
                    sc = dict(zip(first, f([(rid, 0) for rid in first.values()])))
                    out += [dict(row_id=r["row_id"], N=0, score=float(sc[r["row_id"].rsplit("|", 1)[0]])) for r in rs]
                    continue
                out += [dict(row_id=r["row_id"], N=n, score=float(s)) for r, s in zip(rs, f([(r["row_id"], n) for r in rs]))]
            by_split = defaultdict(list)
            for x in out:
                by_split[x["row_id"].split("|")[0]].append(x)
            for c2, xs in by_split.items():
                p = sdir / f"{a.out}-{name}-{c2}-{a.score}.jsonl"
                p.write_text("".join(json.dumps(x) + "\n" for x in xs))
            print(f"{cfg} {name}: wrote", len(out), "scores", flush=True)
    (HERE / "results/injecforks").mkdir(parents=True, exist_ok=True)
    (HERE / f"results/injecforks/{a.out}-selection.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
