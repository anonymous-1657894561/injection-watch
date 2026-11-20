"""Shared look for every paper figure. The rules are in docs/paper/figure-style.md; this file is their single source.

Plots:     from style import *; setup()          then METHOD["probe"] etc. for each line
Diagrams:  from style import INK, BLUE, ...       (diagrams.py writes SVG with the same tokens)
"""
from pathlib import Path

# ---- colour tokens (contrast on white in brackets; text needs >= 4.5) ----
INK = "#1f2328"          # text, axes, our tokens in diagrams                        (15.8)
INK2 = "#4b5563"         # secondary text: annotations, axis notes                   (7.6)
BLUE = "#1f5fa8"         # PRIMARY: our method, and only our method                  (6.4)
BLUE_TINT = "#d6e3f3"    # fills behind blue marks: CI band of the probe, layer read
BLUE_WASH = "#eef3fa"    # panel background for the probe step in diagrams
ACCENT = "#f2b277"       # ACCENT (pale orange): the injected instruction, and nothing else; fill only
ACCENT_TINT = "#fde6cf"  # accent as a band or highlighter behind text
ACCENT_INK = "#a3540f"   # stroke or hatch drawn on an accent fill                   (5.5)
GREY_D = "#374151"       # baselines, first family (LLM readers)                     (10.3)
GREY_M = "#6b7280"       # baselines, second family (classifiers)                    (4.8)
GREY_L = "#c3c8cf"       # reference lines (chance, random gate), neutral bars in diagrams
GREY_W = "#f1f3f5"       # neutral fill (card headers)
GREY_BAND = "#e6e9ed"    # the operating point (N = 16) in plots
# Subtle hues for the forecast comparisons, one family per kind of baseline (dark = the stronger one).
# Validated with dataviz validate_palette.js --pairs all against BLUE: CVD and normal-vision checks pass;
# the light pair is under 3:1 contrast, so dash and marker must carry identity too.
ROSE, ROSE_L = "#a8456c", "#cc7f9d"     # LLM readers: monitor, self-report                      (6.0, 2.9)
OCHRE, OCHRE_L = "#8a6d10", "#bc9a33"   # text classifiers: fine-tuned DeBERTa, sentence embedding (5.0, 2.6)

# ---- sizes (points). AAAI: \columnwidth = 3.3 in, \textwidth = 7.0 in ----
FS = 9                   # every piece of text in every figure; never smaller
COL_W, TEXT_W = 3.3, 7.0
LW_OURS, LW_BASE, LW_REF, LW_AXIS = 1.6, 1.0, 0.8, 0.6
MS = 4.5

# ---- one look per method, the same in every figure. Markers are unique per method. ----
# Ours: blue, solid, filled circle, thickest line, drawn on top.
# Baselines: grey, dashed or dotted, hollow markers. Reference lines: light grey, no marker.
METHOD = {
    "probe":      dict(color=BLUE, ls="-", marker="o", mfc=BLUE, lw=LW_OURS, zorder=5),
    "monitor":    dict(color=ROSE, ls="--", marker="s", mfc="white", lw=LW_BASE, zorder=3),
    "selfreport": dict(color=ROSE_L, ls="-.", marker="^", mfc="white", lw=LW_BASE, zorder=3),
    "ft":         dict(color=OCHRE, ls="--", marker="D", mfc="white", lw=LW_BASE, zorder=3, ms=3.8),
    "emb":        dict(color=OCHRE_L, ls="-.", marker="v", mfc=OCHRE_L, lw=LW_BASE, zorder=3),
    "detection":  dict(color=GREY_D, ls="--", marker="h", mfc=GREY_D, lw=LW_BASE, zorder=3),
    "piguard":    dict(color=GREY_M, ls="-.", marker="P", mfc=GREY_M, lw=LW_BASE, zorder=3),
    "pg2":        dict(color=GREY_M, ls=":", marker="X", mfc=GREY_M, lw=LW_BASE + 0.2, zorder=3),
    "classifier": dict(color=GREY_M, ls=(0, (1, 1.5)), lw=LW_BASE, zorder=2),          # input classifiers at 0.5
    "random":     dict(color=GREY_L, ls=(0, (3, 2)), lw=LW_REF, zorder=1),
}


def line(ax, x, y, m, label=None, **kw):
    """Plot one method's line with its fixed look."""
    st = {**METHOD[m], **kw}
    ms = st.pop("ms", MS if "marker" in st else None)
    if ms is not None:
        st.update(ms=ms, mew=0.9)
    return ax.plot(x, y, label=label, **st)


def band(ax, x, lo, hi, m="probe"):
    """Interval band: blue tint for ours, near-transparent grey for a baseline."""
    if METHOD[m]["color"] == BLUE:
        return ax.fill_between(x, lo, hi, color=BLUE_TINT, alpha=0.85, lw=0, zorder=2)
    return ax.fill_between(x, lo, hi, color=METHOD[m]["color"], alpha=0.10, lw=0, zorder=1)


def highlight_x(ax, x, label, width=0.32, y=None):
    """Light grey band marking the operating point (where the gate acts). Not orange: orange means injection."""
    ax.axvspan(x - width / 2, x + width / 2, color=GREY_BAND, lw=0, zorder=0)
    if label:
        lo, hi = ax.get_ylim()
        ax.text(x + width / 2 + 0.05, hi if y is None else y, label, color=INK2, ha="left", va="top")


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes"], "mathtext.fontset": "stix",
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
        "legend.fontsize": FS, "legend.frameon": False, "legend.handlelength": 2.4, "legend.labelspacing": 0.25,
        "legend.columnspacing": 1.0, "legend.borderaxespad": 0.1, "legend.handletextpad": 0.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": LW_AXIS, "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titlelocation": "left", "axes.titlepad": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    return plt


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    print("wrote", path)
