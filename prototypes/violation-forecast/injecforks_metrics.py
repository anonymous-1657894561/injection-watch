"""Point estimates for model selection on dev (no bootstrap). The reported numbers come from scoring_harness.py via
injecforks_score.py, which asserts that its point estimate equals within_case_mean() here.

within_case_mean: per case with both labels, the AUROC of its rows (ties count 1/2); the mean over those cases, each case
weighted equally (scoring_harness's auroc_within_case_mean with k = 1 per row).
within_case_ap_mean: the same with average precision; its chance level (within_case_ap_chance) is the mean positive share of
those cases, not 0.5. Written without sklearn so it checks the harness independently.
"""
from collections import defaultdict

import numpy as np


def per_case_auroc(case, label, score):
    by = defaultdict(lambda: ([], []))
    for c, y, s in zip(case, label, score):
        by[c][int(bool(y))].append(s)
    out = {}
    for c, (neg, pos) in by.items():
        if pos and neg:
            p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
            out[c] = float(((p > n) + 0.5 * (p == n)).mean())
    return out


def within_case_mean(case, label, score):
    a = per_case_auroc(case, label, score)
    return float(np.mean(list(a.values()))) if a else float("nan")


def within_pairs(case, label, score):
    """Pair-weighted within-case AUROC (scoring_harness's auroc_within)."""
    by = defaultdict(lambda: ([], []))
    for c, y, s in zip(case, label, score):
        by[c][int(bool(y))].append(s)
    num = den = 0.0
    for neg, pos in by.values():
        if pos and neg:
            p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
            num += ((p > n) + 0.5 * (p == n)).sum(); den += p.size * n.size
    return num / den if den else float("nan")


def _ap(label, score):
    """Average precision with tied scores taken as one threshold (sklearn's step definition)."""
    y, s = np.asarray(label, float), np.asarray(score, float)
    out, tp, fp = 0.0, 0.0, 0.0
    for t in np.unique(s)[::-1]:
        m = s == t
        dtp = y[m].sum(); tp += dtp; fp += (1 - y[m]).sum()
        out += dtp / y.sum() * tp / (tp + fp)
    return out


def _mixed(case, label, score):
    by = defaultdict(lambda: ([], []))
    for c, y, s in zip(case, label, score):
        by[c][0].append(int(bool(y))); by[c][1].append(s)
    return {c: v for c, v in by.items() if 0 < sum(v[0]) < len(v[0])}


def within_case_ap_mean(case, label, score):
    m = _mixed(case, label, score)
    return float(np.mean([_ap(y, s) for y, s in m.values()])) if m else float("nan")


def within_case_ap_chance(case, label, score):
    m = _mixed(case, label, score)
    return float(np.mean([np.mean(y) for y, _ in m.values()])) if m else float("nan")
