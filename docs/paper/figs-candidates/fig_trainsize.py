"""Training-set-size curve (appendix). Reads prototypes/violation-forecast/results/e144/trainsize/curve.json, written by
prototypes/violation-forecast/e144/trainsize.py. Run from the repo root: python3 docs/paper/figs-candidates/fig_trainsize.py

Two panels (general, targeted arm) sharing the y axis; in each, N = 16 solid and N = 8 dashed; shaded: range over the
five random subsets at each size (the full split is a single point)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from style import *
plt = setup()

ROOT = Path(__file__).resolve().parents[3]
D = json.load(open(ROOT / "prototypes/violation-forecast/results/e144/trainsize/curve.json"))
OUT = ROOT / "docs/paper/fig_trainsize.pdf"
SIZES = [14, 54, 217, 867]                     # tick labels: every other subset size, ending at the full train split

fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, (arm, title) in zip(axes, [("realistic", "General arm"), ("smoke", "Targeted arm")]):
    for n, kw in [(8, dict(ls="--", mfc="white")), (16, {})]:
        g = defaultdict(list)
        for r in D:
            if r["arm"] == arm:
                g[r["cases"]].append(r[f"N{n}"])
        x = sorted(g); m = [np.mean(g[k]) for k in x]
        assert all(len(g[k]) == (1 if k == x[-1] else 5) for k in x), "five subsets per size; the full split once"
        band(ax, x, [min(g[k]) for k in x], [max(g[k]) for k in x])
        line(ax, x, m, "probe", ms=3.5, lw=LW_OURS if n == 16 else LW_BASE + 0.2, **kw)
        if arm == "smoke":                     # direct labels once, on the right panel; same line styles in both
            ax.text(x[-1] * 1.15, m[-1], f"$N={n}$", va="center", ha="left")
    line(ax, [10, 1100], [0.5, 0.5], "classifier")
    ax.set_xscale("log"); ax.set_xlim(11, 1000); ax.set_title(title)
    ax.set_xticks(SIZES, [str(s) for s in SIZES]); ax.xaxis.set_minor_locator(plt.NullLocator())
axes[0].set_ylim(0.45, 0.78); axes[0].set_yticks([0.5, 0.6, 0.7])
axes[0].set_ylabel("Within-case AUROC ($\\uparrow$)")
fig.supxlabel("Training cases (each with all its replies)", fontsize=FS, y=-0.1)
save(fig, OUT)
