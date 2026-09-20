# -*- coding: utf-8 -*-
"""
Figure 3 — Model Performance: ROC / PR Curves + Confusion Matrix
══════════════════════════════════════════════════════════════════

目的：展示最终模型 (CatBoost+S0 Top4) 在验证集上的 ROC、PR 曲线
      以及 Isotonic 校准方案的混淆矩阵和关键性能指标。

数据来源：
  最终训练/Final_Predictions_CatBoost+S0_Top4.csv
  最终训练/Final_Metrics_with_CI_CatBoost+S0_Top4.csv

输出：
  fig3_model_performance.png / .pdf
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             confusion_matrix, average_precision_score)
from figures.fig_utils import set_style, save_figure

PRED_CSV   = ROOT / "最终训练/Final_Predictions_CatBoost+S0_Top4.csv"
METRIC_CSV = ROOT / "最终训练/Final_Metrics_with_CI_CatBoost+S0_Top4.csv"
OUT_NAME   = "fig3_model_performance"

# ── Palette — colorblind-friendly and stable in grayscale ────────────────
# Curve identity uses both color and line style: dotted (uncalibrated),
# dashed (Platt), and solid (isotonic).
C_UNCALIB = "#737373"  # neutral gray: uncalibrated reference
C_PLATT = "#0072B2"   # Okabe-Ito blue
C_ISO = "#009E73"     # Okabe-Ito bluish green: selected output strategy
C_GRID = "#D9E1E5"
CM_CMAP = LinearSegmentedColormap.from_list(
    "cm_cmap", ["#FFFFFF", "#E8F4F0", "#A9D8C9", "#57AE95", C_ISO], N=256)
SHOW_FIGURE_TITLE = False

# Each entry: (column, label, color, linestyle, linewidth, alpha)
# Isotonic gets a thicker, fully-opaque line so it reads as "the one we use".
MODELS = [
    ("prob_uncalib", "Uncalibrated", C_UNCALIB, ":",  1.7, 1.00),
    ("prob_platt",   "Platt",        C_PLATT,   "--", 1.8, 1.00),
    ("prob_iso",     "Isotonic",    C_ISO,     "-",  2.1, 1.00),
]

# ── Load ──────────────────────────────────────────────
pred_df   = pd.read_csv(PRED_CSV)
metric_df = pd.read_csv(METRIC_CSV)
y_true    = pred_df["y_true"].values

iso_row = metric_df[metric_df["Model_Status"] == "Isotonic Calibrated"].iloc[0]
thr_iso = float(iso_row["Thr_used"])

def _point(s):
    """Extract the point estimate from a 'value (low-high)' string."""
    return float(str(s).split()[0])

iso_metrics = {
    "Sens": _point(iso_row["Sensitivity"]),
    "Spec": _point(iso_row["Specificity"]),
    "PPV":  _point(iso_row["PPV"]),
    "NPV":  _point(iso_row["NPV"]),
    "F1":   _point(iso_row["F1"]),
    "Acc":  _point(iso_row["Accuracy"]),
}

# Pre-compute curves and operating point on Isotonic
curve_data = {}
for col, label, color, ls, lw, alpha in MODELS:
    p = pred_df[col].values
    fpr, tpr, _ = roc_curve(y_true, p)
    prec, rec, _ = precision_recall_curve(y_true, p)
    curve_data[col] = dict(fpr=fpr, tpr=tpr, auroc=auc(fpr, tpr),
                           prec=prec, rec=rec,
                           auprc=average_precision_score(y_true, p))

p_iso  = pred_df["prob_iso"].values
y_pred = (p_iso >= thr_iso).astype(int)
cm     = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
op_fpr  = FP / (FP + TN)
op_tpr  = TP / (TP + FN)
op_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
op_rec  = TP / (TP + FN)
pos_ratio = y_true.mean()

# ── Figure ────────────────────────────────────────────
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
fig = plt.figure(figsize=(15.2, 4.65), dpi=300)
axR = fig.add_axes([0.055, 0.155, 0.285, 0.735])
axP = fig.add_axes([0.372, 0.155, 0.285, 0.735])
axC = fig.add_axes([0.695, 0.155, 0.285, 0.735])


def add_panel_label(ax, label):
    ax.text(-0.14, 1.06, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold",
            color="#1f2933")

# ─ PANEL 1: ROC ─────────────────────────────────────
for col, label, color, ls, lw, alpha in MODELS:
    d = curve_data[col]
    axR.plot(d["fpr"], d["tpr"], color=color, linestyle=ls,
             linewidth=lw, alpha=alpha,
             label=f"{label}  AUROC = {d['auroc']:.3f}")
axR.plot([0, 1], [0, 1], color="#888888", linestyle=":",
         linewidth=0.8, alpha=0.7, zorder=1)
# Operating point — only on the chosen (Isotonic) curve
axR.scatter([op_fpr], [op_tpr], s=85, facecolor="white",
            edgecolor=C_ISO, linewidth=1.7, zorder=10)
axR.annotate(f"Isotonic operating point\n(threshold = {thr_iso:.3f})",
             xy=(op_fpr, op_tpr),
             xytext=(34, -25), textcoords="offset points",
             fontsize=9.5, color=C_ISO, fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="-", color=C_ISO, lw=0.9, alpha=0.80))

axR.set_xlabel("1 - Specificity (FPR)")
axR.set_ylabel("Sensitivity (TPR)")
axR.set_xlim(-0.02, 1.02);  axR.set_ylim(-0.02, 1.02)
axR.set_box_aspect(0.82)
axR.legend(loc="lower right", frameon=True,
           edgecolor="#cccccc", framealpha=0.95, handletextpad=0.6,
           borderpad=0.55, labelspacing=0.45)
axR.set_title("", fontsize=11.5, fontweight="bold", loc="left", pad=8)

# ─ PANEL 2: Precision–Recall ────────────────────────
for col, label, color, ls, lw, alpha in MODELS:
    d = curve_data[col]
    axP.plot(d["rec"], d["prec"], color=color, linestyle=ls,
             linewidth=lw, alpha=alpha,
             label=f"{label}  AUPRC = {d['auprc']:.3f}")
axP.axhline(y=pos_ratio, color="#888888", linestyle=":",
            linewidth=0.8, alpha=0.7, zorder=1)
axP.text(1.0, pos_ratio + 0.02, f"Prevalence = {pos_ratio:.2f}",
         ha="right", va="bottom", fontsize=9.3, color="#666666",
         fontstyle="italic")
axP.scatter([op_rec], [op_prec], s=85, facecolor="white",
            edgecolor=C_ISO, linewidth=1.7, zorder=10)
axP.annotate("Isotonic operating point",
             xy=(op_rec, op_prec),
             xytext=(-16, 18), textcoords="offset points",
             fontsize=9.5, color=C_ISO, fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="-", color=C_ISO, lw=0.9, alpha=0.80))

axP.set_xlabel("Recall (Sensitivity)")
axP.set_ylabel("Precision (PPV)")
axP.set_xlim(-0.02, 1.02);  axP.set_ylim(-0.02, 1.02)
axP.set_box_aspect(0.82)
axP.legend(loc="upper right", frameon=True,
           edgecolor="#cccccc", framealpha=0.95, handletextpad=0.6,
           borderpad=0.55, labelspacing=0.45)
axP.set_title("", fontsize=11.5, fontweight="bold",
              loc="left", pad=8)

# ─ PANEL 3: Confusion Matrix ────────────────────────
cm_norm = cm.astype(float) / cm.sum()
vmax    = cm.max() * 1.15
axC.imshow(cm, cmap=CM_CMAP, vmin=0, vmax=vmax, aspect="equal")

# Cell annotations: count (big) + % (small) + TN/FP/FN/TP corner tag
corners = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        v   = cm[i, j]
        pct = cm_norm[i, j] * 100
        is_dark = v > vmax * 0.55
        main_color = "white"    if is_dark else "#1a1a1a"
        sub_color  = "#E7F2F1"  if is_dark else "#667783"
        axC.text(j, i, f"{v}", ha="center", va="center",
                 fontsize=26, fontweight="bold", color=main_color)
        axC.text(j, i + 0.28, f"{pct:.1f}%", ha="center", va="center",
                 fontsize=10.5, color=sub_color)
        axC.text(j - 0.45, i - 0.42, corners[i][j],
                 ha="left", va="top", fontsize=10.5, fontweight="bold",
                 color=sub_color, fontstyle="italic")

axC.set_xticks([0, 1]); axC.set_yticks([0, 1])
axC.set_xticklabels(["Predicted\nNo SSI", "Predicted\nSSI"], fontsize=11)
axC.set_yticklabels(["Actual\nNo SSI",    "Actual\nSSI"   ], fontsize=11)
axC.tick_params(axis="both", length=0)
axC.tick_params(axis="x", pad=3)
axC.tick_params(axis="y", pad=6)

# Thin frame and white cell separators
for spine in axC.spines.values():
    spine.set_color("#888888")
    spine.set_linewidth(0.8)
axC.axhline(0.5, color="white", linewidth=2.5)
axC.axvline(0.5, color="white", linewidth=2.5)

axC.set_title("", fontsize=11.5,
              fontweight="bold", loc="left", pad=8)
axC.set_aspect("equal", adjustable="box")


# ── Shared cosmetics for the curve panels ──
for ax in (axR, axP):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"  ].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(True, linestyle=":", color=C_GRID, alpha=0.6, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#444444", labelsize=10)

add_panel_label(axR, "A")
add_panel_label(axP, "B")
add_panel_label(axC, "C")

if SHOW_FIGURE_TITLE:
    fig.suptitle(
    "Final model performance — CatBoost+S0 (Top 4) on validation set (n = 70)",
        fontsize=14, fontweight="bold", y=0.98)

save_figure(fig, OUT_NAME)
fig.savefig(Path(__file__).resolve().parent / f"{OUT_NAME}_600dpi.png", dpi=600)
fig.savefig(Path(__file__).resolve().parent / f"{OUT_NAME}.svg", dpi=600)
plt.close()
print("[DONE] Figure 3 — Model Performance plot generated.")
