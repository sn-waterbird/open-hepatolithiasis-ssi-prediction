# -*- coding: utf-8 -*-
"""
fig_utils.py — 通用绘图样式与配置

为所有 Figure 提供统一的颜色、字体、字号、图例风格，
确保论文 7 张图视觉风格一致。
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── 输出目录 ──────────────────────────────────────────────
FIGURE_DIR = Path(__file__).resolve().parent

# ── 配色方案 ──────────────────────────────────────────────
#   主色：CatBoost+S0 Top4 最终模型
COLOR_UNCALIB  = "#4C72B0"   # Uncalibrated (Base)   — 蓝色
COLOR_PLATT    = "#DD8452"   # Platt (Sigmoid)       — 橙褐色
COLOR_ISOTONIC = "#55A868"   # Isotonic Calibrated   — 绿色
COLOR_FULL     = "#C44E52"   # Full model            — 红色
COLOR_REDUCED  = "#8172B3"   # Reduced (Top4) model  — 紫色

# 消融实验各策略
COLOR_STRATEGIES = {
    "PI-only":    "#4C72B0",
    "PVC-only":   "#DD8452",
    "LFC-only":   "#55A868",
    "Consensus":  "#C44E52",
    "SkipCV":     "#8172B3",
}

# 通用学术配色
COLOR_POS = "#C44E52"   # 阳性事件 (红色)
COLOR_NEG = "#4C72B0"   # 阴性事件 (蓝色)

# ── 字体设置 ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.pad_inches": 0.15,
})

# ── 通用样式 ──────────────────────────────────────────────
def set_style():
    """应用统一的 matplotlib 风格（类 seaborn whitegrid）"""
    plt.rcParams.update({
        "axes.facecolor":      "white",
        "axes.edgecolor":      ".8",
        "axes.grid":           True,
        "grid.color":          ".9",
        "grid.linestyle":      "-",
        "grid.alpha":          0.6,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
    })

def save_figure(fig: plt.Figure, name: str, dpi: int = 300):
    """以 PNG + PDF 格式保存至 figures/ 目录"""
    path_png = FIGURE_DIR / f"{name}.png"
    path_pdf = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(path_png, dpi=dpi)
    fig.savefig(path_pdf, dpi=dpi)
    print(f"  [SAVED] {path_png.name}")
    print(f"  [SAVED] {path_pdf.name}")

# ── 快捷提取 Bootstrap CI ──────────────────────────────
def parse_ci(value_str: str) -> tuple:
    """
    将 '0.8467 (0.7292-0.9425)' 解析为 (mean, lower, upper)
    """
    import re
    m = re.match(r"([\d.]+)\s*\(([\d.]+)-([\d.]+)\)", str(value_str))
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return float(value_str), None, None
