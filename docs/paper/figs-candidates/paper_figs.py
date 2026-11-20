"""Paper figures 2a, 2b and 3, written straight to docs/paper/*.pdf.  Run from the repo root:

    python docs/paper/figs-candidates/paper_figs.py

Fig. 2 (test, general arm, 299 cases) uses the same sources as the text:
  N = 8, 16      E113 all16 (16 rollouts, two-way bootstrap)   <RES>/injecforks/analysis/e113.json
  N = 32..128    E79 test, original rules (case bootstrap)     <RES>/injecforks/analysis/e79-final-test-verdict.json
  timings        E104                                          <RES>/e104-timing/summary.md
Fig. 3 (test) is the E110 fresh-input run                      <RES>/e110/{units.jsonl,summary.json}
"""
import csv, json, subprocess
from pathlib import Path
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[3]
RES = "prototypes/violation-forecast/results"
OUT = ROOT / "docs/paper"

import sys; sys.path.insert(0, str(Path(__file__).parent))
from style import *
plt = setup()
K, CLS = INK, GREY_M
STY = {m: {k: v for k, v in METHOD[m].items() if k != "zorder"} for m in ("probe", "monitor", "selfreport")}
LAB = {"probe": "Probe (ours)", "monitor": "LLM monitor", "selfreport": "Self-report"}
M = ("probe", "monitor", "selfreport")
NS = [0, 8, 16, 32, 64, 128]; X = list(range(len(NS)))

e113 = json.load(open(ROOT / RES / "injecforks/analysis/e113.json"))["all16"]
e79 = json.load(open(ROOT / RES / "injecforks/analysis/e79-final-test-verdict.json"))["per_n"]
R = {(m, 0): dict(est=0.5, ci=[0.5, 0.5]) for m in M}          # one shared input state: 0.5 by construction
for n in (8, 16):
    assert e113[str(n)]["cases"] == 299 and not any(e113[str(n)]["missing"].values())
    for m in M: R[m, n] = e113[str(n)]["auroc"][m]
for n in (32, 64, 128):
    for m in M: R[m, n] = e79[str(n)][m]
for m in M: print(m, [round(R[m, n]["est"], 3) for n in NS])

# ---- Fig. 2a: accuracy vs tokens read ----
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.set_ylim(0.45, 1.0); highlight_x(ax, 2, "gate acts here", y=0.995)
band(ax, X, [R["probe", n]["ci"][0] for n in NS], [R["probe", n]["ci"][1] for n in NS])
for m in ("selfreport", "monitor", "probe"):
    ax.plot(X, [R[m, n]["est"] for n in NS], ms=4.5, mew=0.9, zorder=3, label=LAB[m], **STY[m])
line(ax, [-0.2, 5.2], [0.5, 0.5], "classifier")
ax.text(5.15, 0.508, "Input classifiers", color=INK2, ha="right", va="bottom")
ax.set_xticks(X, [str(n) for n in NS]); ax.set_xlim(-0.2, 5.2)
ax.set_ylim(0.45, 1.0); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Reasoning tokens read, $N$"); ax.set_ylabel("Within-case AUROC ($\\uparrow$)")
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], frameon=False, loc="upper left", handlelength=2.2, labelspacing=0.25, borderaxespad=0.1)
# no panel title: Fig. 2 is a single panel since round 20 (fig_forecast_b.pdf is no longer included in main.tex)
fig.savefig(OUT / "fig_forecast_a.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---- Fig. 2b: accuracy vs time per prediction (log), at the gate's N = 16 and at N = 8 ----
T = {"probe": 5e-5, "classifier": 0.018, "selfreport": 0.223, "monitor": 1.986}   # E104, cached
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for m in M:
    a8, a16 = R[m, 8]["est"], R[m, 16]["est"]; lo, hi = R[m, 16]["ci"]; s = STY[m]
    ax.plot([T[m]] * 2, [lo, hi], color=s["color"], lw=0.9, zorder=1)
    ax.plot(T[m], a16, marker=s["marker"], color=s["color"], mfc=s["color"], ms=5, ls="none", zorder=3)
    ax.plot(T[m], a8, marker=s["marker"], color=s["color"], mfc="white", ms=5, mew=1.0, ls="none", zorder=3)
    ax.text(T[m], hi + 0.012, LAB[m], color=INK, va="bottom", ha="left" if m == "probe" else "center")
ax.plot(T["classifier"], 0.5, marker="D", color=CLS, mfc=CLS, ms=5.5, ls="none", zorder=3)
ax.text(T["classifier"] * 1.5, 0.5, "Input classifiers", color=INK2, va="center")
ax.annotate("", xy=(T["probe"], 0.905), xytext=(T["monitor"], 0.905),
            arrowprops=dict(arrowstyle="<->", color=K, lw=0.8, shrinkA=0, shrinkB=0))
ax.text((T["probe"] * T["monitor"]) ** 0.5, 0.915, r"$\approx$40,000$\times$ faster", ha="center", va="bottom")
ax.plot([], [], marker="o", color=K, mfc=K, ls="none", label="$N=16$")
ax.plot([], [], marker="o", color=K, mfc="white", ls="none", label="$N=8$")
ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.38, 0.33), handletextpad=0.2, labelspacing=0.25, borderaxespad=0.1)
ax.set_xscale("log"); ax.set_xlim(2e-5, 40); ax.set_ylim(0.45, 1.0); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10], ["0.1 ms", "1 ms", "10 ms", "0.1 s", "1 s", "10 s"])
ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(2, 10), numticks=100))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.tick_params(axis="x", which="minor", width=0.5, length=1.8)
ax.set_xlabel("Time per prediction (log scale)"); ax.set_ylabel("Within-case AUROC ($\\uparrow$)")
ax.set_title("(b) Accuracy against cost", fontsize=FS, loc="left")
fig.savefig(OUT / "fig_forecast_b.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---- Fig. 3: compliance against reminder budget (test, E110 fresh inputs; sweep as in e110_figure.py) ----
import numpy as np
show = lambda f: (ROOT / RES / "e110" / f).read_text()   # identical to the e105 branch copy (checked 2026-11-20)
U = [json.loads(l) for l in show("units.jsonl").splitlines()]
S110 = json.loads(show("summary.json"))
A = [u for u in U if u["side"] == "attack"]; L = [u for u in U if u["side"] == "lookalike"]
assert (len(A), len(L)) == (493, 702)
vc, vr = np.array([u["viol_control"] for u in A], float), np.array([u["viol_reminder"] for u in A], float)
assert abs(vc.mean() - S110["none"]["call_rate"]) < 1e-9 and abs(vr.mean() - S110["always"]["call_rate"]) < 1e-9
def at(g, t):
    sa, sl = np.array([u[g] for u in A]), np.array([u[g] for u in L]); f = sa >= t
    return (f.sum() + (sl >= t).sum()) / len(U), float(np.mean(np.where(f, vr, vc)))
def sweep(g):
    ts = np.r_[np.inf, np.unique([u[g] for u in U])[::-1]]
    return zip(*[at(g, t) for t in ts])
fig, ax = plt.subplots(figsize=(COL_W, 2.3))
line(ax, [0, 1], [vc.mean(), vr.mean()], "random", label="Random gate")
TR = [("pg2", "PromptGuard 2"), ("piguard", "PIGuard"), ("exposure_n0", "Detection probe"), ("comply", "Compliance probe (ours)")]
KEY = {"pg2": "pg2", "piguard": "piguard", "exposure_n0": "detection", "comply": "probe"}
for g, lab in TR:
    st = {k: v for k, v in METHOD[KEY[g]].items() if k not in ("marker", "mfc", "ms")}
    xs, ys = sweep(g); ax.step(xs, ys, where="post", label=lab, **st)
    c = S110["cut"]["calibrated|0.1"][g]; x, y = at(g, c["threshold"])     # the threshold used in the text
    assert abs(y - c["call_rate"]) < 1e-9
    lo, hi = c["ci"]["call_rate"]                                          # its 95% case-bootstrap interval (E110)
    ax.plot([x, x], [lo, hi], color=st["color"], lw=0.8, zorder=6, solid_capstyle="butt")
    mk = METHOD[KEY[g]]
    ax.plot(x, y, marker=mk["marker"], color=mk["color"], mfc=mk["color"], mec="white", mew=0.8, ms=6.5, ls="none", zorder=7)
    print(g, round(x, 3), round(y, 3), [round(lo, 3), round(hi, 3)])
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.165); ax.set_yticks([0, 0.04, 0.08, 0.12, 0.16], ["0%", "4%", "8%", "12%", "16%"])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_xlabel("Replies reminded (injected and look-alike)"); ax.set_ylabel("Strict compliance ($\\downarrow$)")
save(fig, OUT / "fig_budget.pdf"); plt.close(fig)
print("ok")
