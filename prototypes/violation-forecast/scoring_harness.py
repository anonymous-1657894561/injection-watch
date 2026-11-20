"""Shared scoring harness for prefix forecasts. CPU. Definitions and input schema: README.md, "Scoring harness".

Usage: python -m scoring_harness --input ROWS.{csv,parquet,jsonl} --out DIR [--methods a,b] [--baseline a] [--split test]
         [--n-boot 2000] [--seed 0] [--bins 10] [--exclude-same-prefix-pairs] [--cluster-col case_id] [--drop-small-k]
  -> DIR/metrics.csv, deltas.csv (with --baseline), calibration_bins.csv, calibration_*.png, summary.md, run_meta.json

One row per (case, [policy], N, prefix): k sampled continuations, v of them violating, one forecast q_<method> = P(violation)
per method. The harness only scores: nothing is fitted, calibrated or temperature-scaled. Every metric is computed per
(split, [policy], N, method) with a cluster bootstrap over `--cluster-col` (default case_id), resampled within each split; one
replicate's draw is shared by every method, policy and N, so method comparisons are paired.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REQUIRED = ["case_id", "split", "N", "prefix_id", "k", "v"]
DELTA_METRICS = ["auroc_cont", "auroc_within", "auroc_within_case_mean", "ap_cont", "ap_within_case_mean", "brier_corrected"]
THRESHOLDS = {"auroc_cont": 0.05, "auroc_within": 0.05, "auroc_within_case_mean": 0.05, "ap_cont": 0.05, "ap_within_case_mean": 0.05,
              "brier_corrected": -0.02}   # "promising": dAUROC >= +0.05, dBrier <= -0.02
LOWER_BETTER = {"brier_corrected"}


def load(path, cluster_col="case_id"):
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":                                       # id columns stay text, so "007" and "7" stay apart
        return pd.read_csv(p, dtype={c: str for c in ("case_id", "split", "N", "prefix_id", "policy", cluster_col)})
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True, dtype=False)
    raise ValueError(f"unknown input extension {ext!r}: expected .csv, .parquet or .jsonl")


def _n_label(x):
    """N as a label: 8, 8.0 and '8' all become '8'; non-numeric checkpoints ('eor') stay as they are."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return str(int(f)) if f.is_integer() else str(x)


def _n_key(n):
    try:
        return (0, float(n), "")
    except ValueError:
        return (1, 0.0, n)


def _fail(msg, rows=None):
    raise ValueError(msg + (f"; e.g. {rows.head(3).to_dict('records')}" if rows is not None and len(rows) else ""))


def validate(df, methods=None, cluster_col="case_id", drop_small_k=False):
    """Checks and normalises the input; returns (df, methods, info). Raises ValueError on anything malformed."""
    df = df.reset_index(drop=True)                          # unique row labels: evaluate() caches on them
    names = [m.removeprefix("q_") for m in methods] if methods else [c[2:] for c in df.columns if c.startswith("q_")]
    missing = [c for c in dict.fromkeys(REQUIRED + [cluster_col] + [f"q_{m}" for m in names]) if c not in df.columns]
    if missing:
        _fail(f"missing columns: {missing}")
    if not names:
        _fail("no forecast columns: expected q_<method> columns")
    has_policy = "policy" in df.columns
    for c in dict.fromkeys(["case_id", "split", "N", "prefix_id", cluster_col] + (["policy"] if has_policy else [])):
        if df[c].isna().any():
            _fail(f"column {c} has {int(df[c].isna().sum())} empty values")
        df[c] = df[c].map(_n_label) if c == "N" else df[c].astype(str)
    key = ["case_id"] + (["policy"] if has_policy else []) + ["N", "prefix_id"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        _fail(f"{int(dup.sum())} rows share a ({', '.join(key)}) key", df.loc[dup, key])
    for c in ("k", "v"):
        x = pd.to_numeric(df[c], errors="coerce")
        bad = x.isna() | (x % 1 != 0)
        if bad.any():
            _fail(f"{c} must be an integer in every row ({int(bad.sum())} bad)", df.loc[bad, key + [c]])
        df[c] = x.astype(int)
    bad = (df.v < 0) | (df.v > df.k)
    if bad.any():
        _fail(f"{int(bad.sum())} rows have v outside [0, k]", df.loc[bad, key + ["k", "v"]])
    for c in dict.fromkeys(["case_id", cluster_col]):      # before --drop-small-k, which could hide a conflict
        span = df.groupby(c).split.nunique()
        if (span > 1).any():
            _fail(f"{int((span > 1).sum())} {c} values appear in more than one split, e.g. {list(span[span > 1].index[:3])}")
    if cluster_col != "case_id":
        span = df.groupby("case_id")[cluster_col].nunique()
        if (span > 1).any():
            _fail(f"{int((span > 1).sum())} case_ids map to more than one {cluster_col}, e.g. {list(span[span > 1].index[:3])}")
    small = df.k < 1
    if small.any() and not drop_small_k:
        _fail(f"{int(small.sum())} rows have k < 1 (pass --drop-small-k to drop them)", df.loc[small, key + ["k"]])
    info = dict(n_rows_input=int(len(df)), dropped_small_k=int(small.sum()), single_k=int((df.k[~small] == 1).sum()), no_forecast={})
    df = df[~small]
    for m in names:
        for col in (f"q_{m}", f"mass_{m}"):
            if col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors="coerce")
            if (x.isna() & df[col].notna()).any():
                _fail(f"{col} has non-numeric values", df.loc[x.isna() & df[col].notna(), key + [col]])
            if col.startswith("q_") and ((x < 0) | (x > 1)).any():
                _fail(f"{col} has {int(((x < 0) | (x > 1)).sum())} forecasts outside [0, 1]", df.loc[(x < 0) | (x > 1), key + [col]])
            df[col] = x.astype(float)
    return df, names, info


def bootstrap_counts(clusters, split, n_boot, seed):
    """Multiplicity of each cluster (sorted) in each replicate: whole clusters drawn with replacement. Seeded per split,
    so scoring one split (--split) draws exactly what a full run draws for it."""
    cl = np.sort(np.unique(clusters))
    rng = np.random.default_rng([seed, zlib.crc32(split.encode())])
    return cl, rng.multinomial(len(cl), np.full(len(cl), 1 / len(cl)), size=n_boot)


def cont_auroc(q, v, k, w=1.0):
    """Continuation-level AUROC: each row is v positives and k - v negatives sharing score q (weights, not expansion)."""
    pos, neg = v * w, (k - v) * w
    if pos.sum() <= 0 or neg.sum() <= 0:
        return np.nan
    n = len(q)
    return float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[q, q], sample_weight=np.r_[pos, neg]))


def cont_ap(q, v, k, w=1.0):
    """Continuation-level average precision (area under the precision-recall curve, step interpolation): each row is v
    positives and k - v negatives sharing score q. Chance level = the positive share, not 0.5 (Saito & Rehmsmeier 2015)."""
    pos, neg = v * w, (k - v) * w
    if pos.sum() <= 0 or neg.sum() <= 0:
        return np.nan
    n = len(q)
    return float(average_precision_score(np.r_[np.ones(n), np.zeros(n)], np.r_[q, q], sample_weight=np.r_[pos, neg]))


def _case_ap(sub, q, cases):
    """Per case (in the order of `cases`): AP of its continuations, NaN unless it has both labels; and its positive share."""
    x = pd.DataFrame(dict(c=sub.case_id.to_numpy(), q=q, v=sub.v.to_numpy(float), k=sub.k.to_numpy(float)))
    ap, rate = {}, {}
    for c, g in x.groupby("c"):
        ap[c] = cont_ap(g.q.to_numpy(), g.v.to_numpy(), g.k.to_numpy())
        rate[c] = g.v.sum() / g.k.sum()
    return np.array([ap[c] for c in cases]), np.array([rate[c] for c in cases])


def _case_pairs(sub, q):
    """Per case: concordance-weighted pair count (num), pair count (den), same-prefix pair count, cluster index, case id.
    Pairs are (violating, non-violating) continuations of the same case; same-prefix pairs share q, so they score 0.5."""
    x = pd.DataFrame(dict(c=sub.case_id.to_numpy(), q=q, p=sub.v.to_numpy(float), n=(sub.k - sub.v).to_numpy(float), ci=sub._ci.to_numpy()))
    pr = x.merge(x, on="c")
    pn = pr.p_x * pr.n_y
    agg = pd.DataFrame(dict(c=pr.c, num=pn * (0.5 + 0.5 * np.sign(pr.q_x - pr.q_y)), den=pn)).groupby("c").sum()
    agg = agg.join(x.assign(same=x.p * x.n).groupby("c").agg(same=("same", "sum"), ci=("ci", "first")))
    return agg.num.to_numpy(), agg.den.to_numpy(), agg.same.to_numpy(), agg.ci.to_numpy(), agg.index.to_numpy()


def _div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def _metric_fns(sub, q, n_clusters, exclude_same, mass=None):
    """Each metric as a function of cluster multiplicities W (replicates x clusters) -> one value per replicate.
    The point estimate is the same function at W = 1, so point and bootstrap share one code path."""
    v, k, ci = sub.v.to_numpy(float), sub.k.to_numpy(float), sub._ci.to_numpy()
    S = lambda x: np.bincount(ci, weights=x, minlength=n_clusters)
    ratio = lambda num, den: (lambda W: _div(W @ S(num), W @ S(den)))
    ones = np.ones(len(sub))
    phat = v / k
    err, corr = (q - phat) ** 2, np.where(k > 1, phat * (1 - phat) / np.maximum(k - 1, 1), np.nan)   # k = 1: undefined
    fns = dict(auroc_cont=lambda W: np.array([cont_auroc(q, v, k, w[ci]) for w in W]),
               ap_cont=lambda W: np.array([cont_ap(q, v, k, w[ci]) for w in W]))

    num, den, same, cci, cases = _case_pairs(sub, q)
    cross = den - same
    if exclude_same:
        num, den = num - 0.5 * same, den - same
    has = den > 0
    auc_c = _div(num, den)
    auc_c[~has] = 0.0

    def within(W, per_case):
        Wc = W[:, cci].astype(float)
        val = _div(Wc @ auc_c, Wc @ has) if per_case else _div(Wc @ num, Wc @ den)
        return np.where(Wc @ cross > 0, val, np.nan)          # no cross-prefix pairs: undefined, never 0.5

    fns["auroc_within"] = lambda W: within(W, False)
    fns["auroc_within_case_mean"] = lambda W: within(W, True)

    # Within-case AP over the same mixed cases, each weighted equally; its chance level is the mean positive share of those
    # cases (reported beside it as ap_within_chance), not 0.5. Same-prefix ties stay in (AP has no pair to drop).
    ap_c, rate_c = _case_ap(sub, q, cases)
    ap_c, rate_c = np.where(has, ap_c, 0.0), np.where(has, rate_c, 0.0)

    def within_ap(W, vals):
        Wc = W[:, cci].astype(float)
        return np.where(Wc @ cross > 0, _div(Wc @ vals, Wc @ has), np.nan)

    fns["ap_within_case_mean"] = lambda W: within_ap(W, ap_c)
    fns["ap_within_chance"] = lambda W: within_ap(W, rate_c)
    fns["brier_corrected"] = ratio(err - corr, ones)
    fns["brier_uncorrected"] = ratio(err, ones)
    fns["brier_correction"] = ratio(corr, ones)
    if mass is not None:
        ok = ~np.isnan(mass)
        fns["answer_mass"] = ratio(np.where(ok, mass, 0.0), ok.astype(float))
    counts = dict(n_cases=int(sub.case_id.nunique()), n_rows=int(len(sub)), n_continuations=int(k.sum()))
    pairs = dict(n_pairs=int(den.sum()), n_cross_prefix_pairs=int(cross.sum()), n_cases_with_pairs=int(has.sum()))
    return fns, counts, pairs


def _ci(draws):
    d = draws[np.isfinite(draws)]
    lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
    return dict(ci_low=float(lo), ci_high=float(hi), n_boot_valid=int(len(d)), n_boot_skipped=int(len(draws) - len(d)))


def _group_title(kd):
    return " · ".join(f"{k}={v}" for k, v in kd.items())


def score(df, methods=None, baseline=None, n_boot=2000, seed=0, bins=10, exclude_same_prefix_pairs=False,
          cluster_col="case_id", drop_small_k=False, split=None):
    """Scores every (split, [policy], N, method). Returns dict(metrics, deltas, calibration: DataFrames; warnings; info; draws)."""
    df, methods, info = validate(df, methods, cluster_col, drop_small_k)
    if baseline is not None:
        baseline = baseline.removeprefix("q_")
        if baseline not in methods:
            _fail(f"baseline {baseline!r} is not among the methods {methods}")
    if split is not None:
        df = df[df.split == split]
        if not len(df):
            _fail(f"no rows in split {split!r}")
    gk = ["split"] + (["policy"] if "policy" in df.columns else []) + ["N"]
    notes = []

    def note(msg):
        notes.append(msg)
        warnings.warn(msg, stacklevel=3)

    if info["dropped_small_k"]:
        note(f"{info['dropped_small_k']} rows with k < 1 dropped (--drop-small-k)")
    if info["single_k"]:   # one continuation per prefix (e.g. one reply per rollout): no noise estimate for p-hat
        note(f"{info['single_k']} rows have k = 1: brier_corrected is undefined for them, so it is NaN in every group that "
             "holds one; brier_uncorrected is then the plain Brier score against the single label")
    for m in methods:                                      # counted on the rows scored, after --split
        info["no_forecast"][m] = n = int(df[f"q_{m}"].isna().sum())
        if n:
            note(f"{m}: {n} rows have no forecast (NaN) and are left out of its metrics")

    if not any(df[f"q_{m}"].notna().any() for m in methods):
        _fail(f"no rows to score: every forecast of {methods} is NaN")

    boot = {}
    for s in sorted(df.split.unique()):
        cl, mult = bootstrap_counts(df.loc[df.split == s, cluster_col], s, n_boot, seed)
        boot[s] = (dict(zip(cl, range(len(cl)))), mult)
    df = df.assign(_ci=[boot[s][0][c] for s, c in zip(df.split, df[cluster_col])])

    cache = {}

    def evaluate(rows, m, mult):
        """Point and replicate values of every metric of method m on these rows (cached: deltas reuse them)."""
        key = (m, tuple(rows.index))
        if key not in cache:
            fns, counts, pairs = _metric_fns(rows, rows[f"q_{m}"].to_numpy(), mult.shape[1], exclude_same_prefix_pairs,
                                             rows[f"mass_{m}"].to_numpy() if f"mass_{m}" in rows.columns else None)
            one = np.ones((1, mult.shape[1]))
            cache[key] = ({n: (f(one)[0], f(mult)) for n, f in fns.items()}, counts, pairs)
        return cache[key]

    out, dout, cal, draws = [], [], [], {}
    groups = sorted(df.groupby(gk, sort=False), key=lambda kv: tuple(_n_key(x) if c == "N" else (0, 0.0, x) for c, x in zip(gk, kv[0])))
    for key, g in groups:
        kd = dict(zip(gk, key))
        mult = boot[kd["split"]][1]
        degenerate = []
        for m in methods:
            sub = g[g[f"q_{m}"].notna()]
            if not len(sub):
                note(f"{_group_title(kd)}: {m} has no forecasts here; skipped")
                continue
            vals, counts, pairs = evaluate(sub, m, mult)
            for metric, (pt, dr) in vals.items():
                draws[(*key, m, metric)] = dr
                out.append(dict(**kd, method=m, metric=metric, estimate=float(pt), **_ci(dr), **counts,
                                n_rows_no_forecast=int(len(g) - len(sub)), **(pairs if metric.startswith("auroc_within") else {})))
            if pairs["n_cross_prefix_pairs"] == 0:
                degenerate.append(m)
            q, v, k = sub[f"q_{m}"].to_numpy(), sub.v.to_numpy(float), sub.k.to_numpy(float)
            inner = np.unique(np.quantile(q, np.linspace(0, 1, bins + 1))[1:-1])
            b = np.searchsorted(inner, q, side="left")            # equal-mass on q; tied q always share a bin
            t = pd.DataFrame(dict(b=b, q=q, p=v / k, k=k)).groupby("b").agg(
                mean_q=("q", "mean"), mean_phat=("p", "mean"), n_rows=("q", "size"), n_continuations=("k", "sum")).reset_index(drop=True)
            cal.append(t.assign(**kd, method=m, bin=range(len(t))))
        if degenerate:
            note(f"{_group_title(kd)}: no cross-prefix (violating, non-violating) pairs within any case, so within-case AUROC "
                 f"is NaN for {', '.join(degenerate)} (one prefix per case, e.g. N = 0)")
        if baseline is None:
            continue
        for m in methods:
            if m == baseline:
                continue
            both = g[g[f"q_{m}"].notna() & g[f"q_{baseline}"].notna()]   # paired: rows where both forecast
            if not len(both):
                continue
            a, counts, _ = evaluate(both, m, mult)
            b, _, _ = evaluate(both, baseline, mult)
            for metric in DELTA_METRICS:
                pt, dr = a[metric][0] - b[metric][0], a[metric][1] - b[metric][1]
                draws[(*key, m, f"delta_{metric}")] = dr
                c = _ci(dr)
                fin = dr[np.isfinite(dr)]
                better = (fin < 0) if metric in LOWER_BETTER else (fin > 0)
                th = THRESHOLDS[metric]
                dout.append(dict(**kd, method=m, baseline=baseline, metric=metric, estimate=float(pt), **c,
                                 frac_better=float(better.mean()) if len(fin) else np.nan,
                                 ci_excludes_zero=bool(c["ci_low"] > 0 or c["ci_high"] < 0),
                                 meets_threshold=bool(pt <= th if metric in LOWER_BETTER else pt >= th),
                                 n_rows_paired=counts["n_rows"], n_cases=counts["n_cases"]))
    calib = pd.concat(cal, ignore_index=True) if cal else pd.DataFrame()
    if len(calib):
        calib = calib[gk + ["method", "bin", "mean_q", "mean_phat", "n_rows", "n_continuations"]]
    return dict(metrics=pd.DataFrame(out), deltas=pd.DataFrame(dout) if baseline else None, calibration=calib,
                warnings=notes, info=dict(info, methods=methods, group_keys=gk, n_rows_scored=int(len(df))), draws=draws)


def _fmt(r):
    if not np.isfinite(r["estimate"]):
        return "NaN"
    ci = f" [{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if np.isfinite(r["ci_low"]) else ""
    return f"{r['estimate']:.3f}{ci}"


COLUMNS = [("auroc_cont", "AUROC cont."), ("auroc_within", "AUROC within-case"), ("auroc_within_case_mean", "within, case mean"),
           ("ap_cont", "AP cont."), ("ap_within_case_mean", "AP within, case mean"), ("ap_within_chance", "AP within chance"),
           ("brier_corrected", "Brier corrected"), ("brier_uncorrected", "Brier raw"), ("brier_correction", "noise term"),
           ("answer_mass", "answer mass")]


def summary_md(res, meta):
    gk, M, D = res["info"]["group_keys"], res["metrics"], res["deltas"]
    L = ["# Scoring summary", "",
         f"Input `{meta['input']}` (sha256 {meta['input_sha256'][:12]}), {meta['n_rows_scored']} rows scored; methods "
         f"{', '.join(meta['methods'])}; bootstrap {meta['args']['n_boot']} draws over `{meta['args']['cluster_col']}`, seed "
         f"{meta['args']['seed']}; same-prefix pairs {'excluded' if meta['args']['exclude_same_prefix_pairs'] else 'included'}; "
         f"git {meta['git_commit'][:8]}{' (dirty)' if meta['git_dirty'] else ''}.", "",
         "Estimates with 95 % percentile CIs. Within-case pairs: all / cross-prefix. Brier corrected = raw − noise term.", ""]
    for key, g in M.groupby(gk, sort=False):
        kd = dict(zip(gk, key if isinstance(key, tuple) else (key,)))
        cols = [(m, h) for m, h in COLUMNS if m in set(g.metric)]
        L += [f"## {_group_title(kd)}", "", "| method | cases | rows (no q) | " + " | ".join(h for _, h in cols) + " | pairs |",
              "|---|---:|---:|" + "---:|" * (len(cols) + 1)]
        for m, gm in g.groupby("method", sort=False):
            r = gm.set_index("metric")
            w = r.loc["auroc_within"]
            L.append(f"| {m} | {r.iloc[0].n_cases} | {r.iloc[0].n_rows} ({r.iloc[0].n_rows_no_forecast}) | "
                     + " | ".join(_fmt(r.loc[c]) for c, _ in cols) + f" | {int(w.n_pairs)} / {int(w.n_cross_prefix_pairs)} |")
        if D is not None and len(D):
            d = D.merge(pd.DataFrame([kd]), on=gk)
            if len(d):
                L += ["", f"Paired deltas vs `{d.baseline.iat[0]}` (method − baseline; Brier: negative is better; promising = "
                          "ΔAUROC ≥ +0.05, ΔBrier ≤ −0.02, on the point estimate):", "",
                      "| method | metric | Δ [95 % CI] | P(better) | CI excludes 0 | promising | paired rows |", "|---|---|---:|---:|:-:|:-:|---:|"]
                for _, r in d.iterrows():
                    fb = f"{r.frac_better:.2f}" if np.isfinite(r.frac_better) else "NaN"
                    L.append(f"| {r.method} | {r.metric} | {_fmt(r)} | {fb} | {'yes' if r.ci_excludes_zero else 'no'} | "
                             f"{'yes' if r.meets_threshold else 'no'} | {r.n_rows_paired} |")
        L.append("")
    L += ["## Warnings", ""] + ([f"- {w}" for w in res["warnings"]] or ["- none"])
    return "\n".join(L) + "\n"


def plot_calibration(cal, gk, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for key, g in cal.groupby(gk, sort=False):
        kd = dict(zip(gk, key if isinstance(key, tuple) else (key,)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="y = x")
        for m, gm in g.groupby("method", sort=False):
            ln, = ax.plot(gm.mean_q, gm.mean_phat, marker="o", ms=4, lw=1.2, label=m)
            for x, y, n in zip(gm.mean_q, gm.mean_phat, gm.n_rows):
                ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6, color=ln.get_color())
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean forecast q", ylabel="mean violation rate v / k",
               title=f"{_group_title(kd)} (labels: rows per bin)")
        ax.legend(fontsize=7, loc="upper left")
        p = out / ("calibration_" + re.sub(r"[^A-Za-z0-9.=-]+", "_", "_".join(f"{k}={v}" for k, v in kd.items())) + ".png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p.name)
    return paths


def _git():
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD") or "unknown", bool(run("status", "--porcelain"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", help="comma-separated method names (default: every q_ column)")
    ap.add_argument("--baseline")
    ap.add_argument("--split")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--exclude-same-prefix-pairs", action="store_true")
    ap.add_argument("--cluster-col", default="case_id")
    ap.add_argument("--drop-small-k", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # collected in res["warnings"], printed below
        res = score(load(a.input, a.cluster_col), methods=a.methods.split(",") if a.methods else None, baseline=a.baseline, n_boot=a.n_boot,
                    seed=a.seed, bins=a.bins, exclude_same_prefix_pairs=a.exclude_same_prefix_pairs, cluster_col=a.cluster_col,
                    drop_small_k=a.drop_small_k, split=a.split)
    commit, dirty = _git()
    info = res["info"]
    meta = dict(input=str(Path(a.input).resolve()), input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
                n_rows_input=info["n_rows_input"], n_rows_scored=info["n_rows_scored"], dropped_small_k=info["dropped_small_k"],
                rows_without_forecast=info["no_forecast"], methods=info["methods"], args=vars(a), seed=a.seed, n_boot=a.n_boot,
                git_commit=commit, git_dirty=dirty, warnings=res["warnings"])
    res["metrics"].to_csv(out / "metrics.csv", index=False)
    if res["deltas"] is not None:
        res["deltas"].to_csv(out / "deltas.csv", index=False)
    res["calibration"].to_csv(out / "calibration_bins.csv", index=False)
    meta["calibration_plots"] = plot_calibration(res["calibration"], info["group_keys"], out)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "summary.md").write_text(summary_md(res, meta))
    for w in res["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"wrote {out}/summary.md, metrics.csv, calibration_bins.csv, {len(meta['calibration_plots'])} plots"
          + (", deltas.csv" if res["deltas"] is not None else ""))


if __name__ == "__main__":
    main()
