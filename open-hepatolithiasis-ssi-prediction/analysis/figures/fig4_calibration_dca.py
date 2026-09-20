# -*- coding: utf-8 -*-
"""
Figure 4 — Calibration Curve + Decision Curve Analysis (DCA)
═══════════════════════════════════════════════════════════════

目的：评估 3 种校准方案 (Uncalibrated / Platt / Isotonic) 的概率校准质量,
      并通过 DCA 评估不同阈值下的净收益.

数据来源:
  最终训练/Final_Predictions_CatBoost+S0_Top4.csv

输出:
  fig4_calibration_dca.png / .pdf
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from figures.fig_utils import set_style, save_figure

PRED_CSV = ROOT / "最终训练/Final_Predictions_CatBoost+S0_Top4.csv"
OUT_NAME = "fig4_calibration_dca"

# ── Palette — matches Fig 1/2/3 ──────────────────────────
C_UNCALIB = "#737373"   # neutral gray: uncalibrated reference
C_PLATT   = "#0072B2"   # Okabe-Ito blue
C_ISO     = "#009E73"   # Okabe-Ito bluish green: selected output strategy
C_OPT     = C_ISO        # optional operating-threshold highlight
C_REF     = "#A7B0B7"   # low-contrast reference lines
C_GRID    = "#D9E1E5"
SHOW_FIGURE_TITLE = False

# (column, label, color, linestyle, linewidth, alpha, marker)
MODELS = [
    ("prob_uncalib", "Uncalibrated", C_UNCALIB, ":",  1.7, 1.00, "o"),
    ("prob_platt",   "Platt",        C_PLATT,   "--", 1.8, 1.00, "s"),
    ("prob_iso",     "Isotonic",    C_ISO,     "-",  2.1, 1.00, "D"),
]

# Set to a float (e.g. 0.295) to highlight a fixed operating threshold,
# or leave as None if no single threshold is being committed to.
THRESHOLD_USED = None

N_BINS         = 7              # quantile bins for reliability diagram
CAL_AXIS_MAX   = 0.60           # focus calibration panel on the observed prediction range
PT_MIN, PT_MAX = 0.01, 0.45     # clinically meaningful threshold range


# ── Helpers ──────────────────────────────────────────
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Standard ECE — weighted average of |observed − predicted| per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, n_tot = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else \
               (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n_tot) * abs(y_true[mask].mean()
                                          - y_prob[mask].mean())
    return ece


def net_benefit(y_true, p, thresholds):
    """Vanderbilt-style net benefit: NB = TP/n − FP/n · pt/(1−pt)."""
    n = len(y_true)
    out = np.zeros_like(thresholds, dtype=float)
    for i, pt in enumerate(thresholds):
        yp = (p >= pt).astype(int)
        tp = ((yp == 1) & (y_true == 1)).sum()
        fp = ((yp == 1) & (y_true == 0)).sum()
        out[i] = tp / n - fp / n * pt / max(1 - pt, 1e-6)
    return out


# ── Load ──────────────────────────────────────────────
pred_df = pd.read_csv(PRED_CSV)
y_true  = pred_df["y_true"].values
prev    = y_true.mean()

# Precompute calibration metrics
brier = {col: brier_score_loss(y_true, pred_df[col].values)
         for col, *_ in MODELS}
ece   = {col: expected_calibration_error(y_true, pred_df[col].values, 10)
         for col, *_ in MODELS}

# ── Figure ────────────────────────────────────────────
set_style()
plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
fig = plt.figure(figsize=(11.8, 5.15), dpi=300)
axL = fig.add_axes([0.075, 0.145, 0.405, 0.760])
axR = fig.add_axes([0.565, 0.145, 0.405, 0.760])


def add_panel_label(ax, label):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold",
            color="#1f2933")

# ════════════════════════════════════════════════
# LEFT: Calibration curve
# ════════════════════════════════════════════════
axL.plot([0, 1], [0, 1], color=C_REF, linestyle=":", linewidth=1.2,
         alpha=0.7, zorder=1, label="Perfect calibration")

for col, label, color, ls, lw, alpha, mk in MODELS:
    p = pred_df[col].values
    prob_true, prob_pred = calibration_curve(y_true, p,
                                              n_bins=N_BINS,
                                              strategy="quantile")
    axL.plot(prob_pred, prob_true, marker=mk, color=color, linestyle=ls,
             linewidth=lw, markersize=6.2, alpha=alpha,
             markeredgecolor="white", markeredgewidth=0.8,
             label=label, zorder=4)

# Brier / ECE readout in upper-left — no header, compact spacing
box_left = 0.022
box_top  = 0.945
line_h   = 0.044
readout_lines = []
for col, label, color, *_ in MODELS:
    short = label
    readout_lines.append(
        (f"{short:<13}  Brier {brier[col]:.3f}  ECE {ece[col]:.3f}",
         color, 8.8))

pad_v_top, pad_v_bottom = 0.012, 0.030
pad_h = 0.010
frame_left = box_left - pad_h
frame_top = box_top + pad_v_top
frame_bottom = box_top - (len(readout_lines) - 1) * line_h - pad_v_bottom

axL.add_patch(FancyBboxPatch(
    (frame_left, frame_bottom), 0.750, frame_top - frame_bottom,
    boxstyle="round,pad=0.002,rounding_size=0.008",
    transform=axL.transAxes, linewidth=0.65, edgecolor="#AEB7BE",
    facecolor="white", alpha=0.94, zorder=3, clip_on=False))

for i, (text, color, fs) in enumerate(readout_lines):
    axL.text(box_left, box_top - i * line_h, text,
             transform=axL.transAxes, ha="left", va="top",
             fontsize=fs, fontfamily="DejaVu Sans Mono",
             fontweight="bold", color=color, zorder=4)

# Focus the reliability diagram on the probability range where the validation
# predictions and observed fractions actually lie; this avoids an empty upper
# right quadrant while preserving the identity line.
axL.set_xlim(-0.01, CAL_AXIS_MAX)
axL.set_ylim(-0.01, CAL_AXIS_MAX)
axL.set_xticks(np.arange(0, CAL_AXIS_MAX + 0.001, 0.10))
axL.set_yticks(np.arange(0, CAL_AXIS_MAX + 0.001, 0.10))
axL.set_aspect("equal")

axL.set_xlabel("Predicted probability")
axL.set_ylabel("Observed fraction of positives")
axL.set_title("",
              fontsize=11.5, fontweight="bold", loc="left", pad=8)
axL.legend(loc="lower right", frameon=True,
           edgecolor="#cccccc", framealpha=0.95,
           handletextpad=0.5, borderpad=0.5,
           handlelength=2.6, markerscale=0.9, labelspacing=0.35)

# ════════════════════════════════════════════════
# RIGHT: Decision Curve Analysis
# ════════════════════════════════════════════════
thresholds = np.linspace(PT_MIN, PT_MAX, 200)

# Reference: Treat-all
nb_treat_all = prev - (1 - prev) * thresholds / np.maximum(1 - thresholds, 1e-6)
axR.plot(thresholds, nb_treat_all, color=C_REF, linestyle="--",
         linewidth=1.2, alpha=0.85, label="Treat all", zorder=2)
# Reference: Treat-none
axR.axhline(0, color=C_REF, linestyle=":", linewidth=1.0, alpha=0.8,
            label="Treat none", zorder=2)

# Model curves
all_nb = [nb_treat_all]
for col, label, color, ls, lw, alpha, _ in MODELS:
    nb = net_benefit(y_true, pred_df[col].values, thresholds)
    all_nb.append(nb)
    axR.plot(thresholds, nb, color=color, linestyle=ls, linewidth=lw,
             alpha=alpha, label=label, zorder=4)

# Operating threshold from Fig 3 — only drawn when THRESHOLD_USED is set
if THRESHOLD_USED is not None:
    axR.axvline(THRESHOLD_USED, color=C_OPT, linestyle=(0, (2, 2)),
                linewidth=1.0, alpha=0.55, zorder=1)
    axR.axvspan(THRESHOLD_USED - 0.005, THRESHOLD_USED + 0.005,
                color=C_OPT, alpha=0.10, zorder=1)
    axR.text(THRESHOLD_USED, 0.97, f" thr = {THRESHOLD_USED:.3f}",
             color=C_OPT, fontsize=8.5, fontweight="bold", fontstyle="italic",
             ha="left", va="top",
             transform=axR.get_xaxis_transform(),
             bbox=dict(boxstyle="round,pad=0.25", fc="white",
                       ec=C_OPT, lw=0.6, alpha=0.95))

# y-limits: clip to clinically meaningful range, but extend bottom enough
# to include where model curves actually dip (treat-all blow-up excluded).
model_min = min(np.nanmin(nb) for nb in all_nb[1:])   # skip treat_all
y_top     = max(np.max(nb) for nb in all_nb)
y_bottom  = min(-0.04, model_min - 0.015)
axR.set_ylim(y_bottom, y_top * 1.10)
axR.set_xlim(0, PT_MAX)

axR.set_xlabel("Threshold probability")
axR.set_ylabel("Net benefit")
axR.set_title("", fontsize=11.5,
              fontweight="bold", loc="left", pad=8)
axR.legend(loc="upper right", frameon=True,
           edgecolor="#cccccc", framealpha=0.95,
           handletextpad=0.6, borderpad=0.55, labelspacing=0.40)

# ── Shared cosmetics ──
for ax in (axL, axR):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"  ].set_color("#9AA3A8")
    ax.spines["bottom"].set_color("#9AA3A8")
    ax.grid(True, linestyle=":", color=C_GRID, alpha=0.75, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4b5560", labelsize=10)

add_panel_label(axL, "A")
add_panel_label(axR, "B")

if SHOW_FIGURE_TITLE:
    fig.suptitle("Probability calibration and clinical utility",
                 fontsize=14, fontweight="bold", y=0.98)

save_figure(fig, OUT_NAME)
fig.savefig(Path(__file__).resolve().parent / f"{OUT_NAME}_600dpi.png", dpi=600)
fig.savefig(Path(__file__).resolve().parent / f"{OUT_NAME}.svg", dpi=600)
plt.close()
print("[DONE] Figure 4 — Calibration & DCA plot generated.")
