# -*- coding: utf-8 -*-
"""
Supplementary Figure 2 — Ablation Study
════════════════════════════════════════

目的：比较不同特征选择策略 (PI-only / PVC-only / LFC-only / Consensus / SkipCV)
      在 K = 2 / 4 / 6 下的性能，验证 MCFS 共识框架的相对优势。

数据来源:
  消融实验/ablation_comparison.csv

输出:
  figS2_ablation.png / .pdf
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figures.fig_utils import set_style, save_figure

ABLATION_CSV = ROOT / "消融实验/ablation_comparison.csv"
OUT_NAME     = "figS2_ablation"

# ── Palette ─────────────────────────────────────────────────────────
# Consensus is the chosen one and gets the only warm colour so the eye
# lands on it. Other strategies are cool tones — visually "alternatives".
STRAT_COLORS = {
    "PI-only":   "#5B7B99",   # slate blue
    "PVC-only":  "#7A9CC6",   # muted blue
    "LFC-only":  "#A9C0DA",   # light blue
    "Consensus": "#C77B30",   # terracotta — the chosen (matches Fig 2/3/4)
    "SkipCV":    "#9BA6AA",   # neutral grey
}
STRATEGIES = list(STRAT_COLORS.keys())
KS         = [2, 4, 6]
C_GRID     = "#D9E1E5"

# Publication-oriented palette override, placed below legacy comments to avoid
# issues from older non-UTF-8 text in this file.
STRAT_COLORS.update({
    "PI-only":   "#6F879B",
    "PVC-only":  "#7FA6C8",
    "LFC-only":  "#B7CDE0",
    "Consensus": "#1B7F79",
    "SkipCV":    "#A7B0B4",
})

# Each panel: (column in csv, y-axis label, panel title, y-range fitted to data)
# CRITICAL: per-panel y-range. The original code used a single 0.3-0.7 for all
# three metrics, which clipped AUROC (values ≈ 0.75-0.85) into a useless strip.
PANELS = [
    ("VAL_AUPRC", "VAL AUPRC", "VAL AUPRC", (0.20, 0.62)),
    ("VAL_AUROC", "VAL AUROC", "VAL AUROC", (0.20, 0.92)),
    ("DEV_AUPRC",   "DEV AUPRC",   "DEV AUPRC",   (0.20, 0.62)),
]

# ── Load ─────────────────────────────────────────────────────────────
df = pd.read_csv(ABLATION_CSV)
df = df.rename(columns={
    "INDEP_AUPRC": "VAL_AUPRC",
    "INDEP_AUROC": "VAL_AUROC",
})
df["Strategy"] = df["Strategy"].replace({
    "PI-only":   "PI-only",
    "PVC-only":  "PVC-only",
    "LFC-only":  "LFC-only",
    "Consensus": "Consensus",
    "SkipCV":    "SkipCV",
})

# ── Figure ───────────────────────────────────────────────────────────
set_style()
plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
fig, axes = plt.subplots(
    1, 3, figsize=(13.8, 4.95),
    gridspec_kw=dict(wspace=0.23),
    constrained_layout=False,
)
fig.subplots_adjust(left=0.055, right=0.990, bottom=0.165, top=0.805, wspace=0.23)


def add_panel_label(ax, label):
    ax.text(-0.16, 1.13, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold",
            color="#1f2933")

for ax, (col, ylabel, title, ylim) in zip(axes, PANELS):
    x     = np.arange(len(KS))
    width = 0.165

    for si, strat in enumerate(STRATEGIES):
        vals = []
        for k in KS:
            row = df[(df["Strategy"] == strat) & (df["K"] == k)]
            vals.append(row[col].values[0] if not row.empty else np.nan)

        offset  = (si - (len(STRATEGIES) - 1) / 2) * width
        is_main = strat == "Consensus"
        bars = ax.bar(
            x + offset, vals, width,
            color=STRAT_COLORS[strat],
            edgecolor="white", linewidth=0.7,
            zorder=3 if is_main else 2,
            label=strat,
        )

        # Value labels — rotated vertical so they don't clash with neighbours;
        # bar-coloured so each label is visually tied to its bar.
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + (ylim[1] - ylim[0]) * 0.012,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=7.7,
                color=STRAT_COLORS[strat],
                fontweight="bold" if is_main else "normal",
                rotation=90,
            )

    # Per-metric y-range
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in KS])
    ax.set_xlabel("Number of features (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=8)

    # Cosmetics — consistent with Fig 1-5
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"  ].set_color("#9AA3A8")
    ax.spines["bottom"].set_color("#9AA3A8")
    ax.tick_params(colors="#4b5560", labelsize=10)
    ax.grid(True, axis="y", linestyle=":", color=C_GRID,
            alpha=0.75, linewidth=0.65)
    ax.set_axisbelow(True)

for label, ax in zip(["A", "B", "C"], axes):
    add_panel_label(ax, label)

# ── Single shared legend below all panels ───────────────────────────
handles = [
    plt.Rectangle((0, 0), 1, 1, color=STRAT_COLORS[s], ec="white", lw=0.5)
    for s in STRATEGIES
]
labels = [
    f"{s}" if s == "Consensus" else s
    for s in STRATEGIES
]
fig.legend(
    handles, labels,
    loc="upper center", bbox_to_anchor=(0.5, 0.998),
    ncol=len(STRATEGIES),
    frameon=False, handletextpad=0.55, columnspacing=1.8,
)

save_figure(fig, OUT_NAME)
fig.savefig(Path(__file__).resolve().parent / f"{OUT_NAME}_600dpi.png", dpi=600)
fig.savefig(Path(__file__).resolve().parent / f"{OUT_NAME}.svg", dpi=600)
plt.close()
print("[DONE] Supplementary Figure 2 — Ablation plot generated.")
