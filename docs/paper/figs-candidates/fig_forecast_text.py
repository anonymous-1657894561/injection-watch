"""Fig. forecast with the text classifiers, replacing Table 2 (round 26). Run from the repo root:

    python docs/paper/figs-candidates/fig_forecast_text.py

Every method on the same rows: the 299 evaluation cases x 8 replies (eval8), general arm, test, case-bootstrap CIs.
  probe, monitor, self-report   E79 test   <RES>/injecforks/analysis/e79-final-test-vs-monitor/summary.md (realistic/all)
  sentence embedding            E134 test  <RES>/injecforks/analysis/e134-test/summary.md
  fine-tuned DeBERTa            E136 test  <RES>/injecforks/analysis/e136-test/summary.md
These are Table 2's numbers. The text's N <= 16 comparisons with the monitor use all 16 replies (E113); the caption says so.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "prototypes/violation-forecast/results/injecforks/analysis"
OUT = ROOT / "docs/paper/fig_forecast.pdf"
NS = [0, 8, 16, 32, 64, 128]; X = list(range(len(NS)))


def table(path, cut="realistic/all"):
    """First 'Case-mean within-case AUROC' table in a summary.md -> {method: {N: (est, lo, hi)}}."""
    txt = path.read_text().split("## Case-mean within-case AUROC")[1]
    rows = []
    for l in txt.splitlines()[1:]:
        if l.startswith("|"):
            rows.append(l)
        elif rows:
            break  # end of the first table
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    out = {}
    for l in rows[2:]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        if cells[0] != cut:
            continue
        n = int(cells[1])
        for h, c in zip(head[2:], cells[2:]):
            m = re.match(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", c)
            if m:
                out.setdefault(h, {})[n] = tuple(map(float, m.groups()))
    return out


e79, e134, e136 = table(A / "e79-final-test-vs-monitor/summary.md"), table(A / "e134-test/summary.md"), table(A / "e136-test/summary.md")
assert all(e79["probe"][n][0] == e134["probe"][n][0] == e136["probe"][n][0] for n in NS), "probe estimates differ between sources"
R = {"probe": e79["probe"], "monitor": e79["monitor"], "selfreport": e79["selfreport"], "emb": e134["emb"], "ft": e136["ft"]}
for m, v in R.items():
    print(f"{m:10s}", [round(v[n][0], 3) for n in NS])

import sys; sys.path.insert(0, str(Path(__file__).parent))
from style import *
plt = setup()
LAB = {"probe": "Probe (ours), 0.05 ms", "monitor": "LLM monitor, 2 s", "selfreport": "Self-report, 200 ms",
       "ft": "Fine-tuned DeBERTa, 65 ms", "emb": "Sentence embedding, 31 ms"}

# One panel, every method with its 95% case-bootstrap band (baselines faint, the probe's on top).
fig, ax = plt.subplots(figsize=(COL_W, 2.3))
ax.set_ylim(0.44, 0.96); ax.set_xlim(-0.2, 5.2)
highlight_x(ax, 2, "gate acts here", y=0.955)
order = ("selfreport", "emb", "ft", "monitor", "probe")
for m in order:
    band(ax, X, [R[m][n][1] for n in NS], [R[m][n][2] for n in NS], m)
for m in order:
    line(ax, X, [R[m][n][0] for n in NS], m, LAB[m], ms=3.8 if m != "ft" else 3.4)
line(ax, [-0.2, 5.2], [0.5, 0.5], "classifier")
ax.text(5.15, 0.508, "Input classifiers", color=INK2, ha="right", va="bottom")
ax.set_xticks(X, [str(n) for n in NS]); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_xlabel("Reasoning tokens read, $N$"); ax.set_ylabel("Within-case AUROC ($\\uparrow$)")
h, l = ax.get_legend_handles_labels()
idx = [l.index(LAB[m]) for m in ("probe", "monitor", "selfreport", "ft", "emb")]   # columns fill first: ours + LLM readers | text classifiers
ax.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper center", bbox_to_anchor=(0.42, -0.24), ncol=2)
save(fig, OUT)
