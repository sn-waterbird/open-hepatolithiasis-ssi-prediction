# -*- coding: utf-8 -*-
"""
Generate the integrated manuscript Figure 2.

The script redraws all panels from source CSV files:
  A. Full-feature base model and imbalance-strategy screening heatmap
  B. Top-K traversal curves for DEV CV and held-out VAL AUPRC

Outputs:
  fig2_integrated_mcfs.png
  fig2_integrated_mcfs_600dpi.png
  fig2_integrated_mcfs.pdf
  fig2_integrated_mcfs.svg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


MODEL_ORDER = ["LR", "CatBoost", "XGBoost", "ExtraTrees"]
MODEL_LABELS = {
    "LR": "Logistic regression",
    "CatBoost": "CatBoost",
    "XGBoost": "XGBoost",
    "ExtraTrees": "Extra Trees",
}
STRATEGY_ORDER = ["S0", "S1", "S2", "S3"]

K_OPT = 4
N_FEATURES_PANEL_B = 12

INK = "#1f2933"
NAVY = "#12334d"
BLUE = "#17496d"
TEAL = "#2b8790"
WARM = "#b76a4f"
MUTED = "#6b7d88"
LIGHT = "#e8eef2"
LIGHTER = "#f5f8fa"
GRID = "#dfe7eb"


def default_paths() -> tuple[Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    model_csv = project_root / "训练全特征输出" / "multi_scheme_summary_dev_indep.csv"
    contrib_csv = project_root / "特征筛选共识" / "CatBoost+S0" / "contribution_scheme2.csv"
    topk_csv = project_root / "特征筛选共识" / "CatBoost+S0" / "topk_metrics_scheme2.csv"
    out_prefix = script_dir / "fig2_integrated_mcfs"
    return model_csv, contrib_csv, topk_csv, out_prefix


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_heatmap(path: Path) -> np.ndarray:
    rows = read_csv(path)
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        model = row.get("Model", "")
        strategy = row.get("Strategy", "")
        if model in MODEL_ORDER and strategy in STRATEGY_ORDER:
            values[(model, strategy)] = float(row["DEV_AUPRC"])

    missing = [(m, s) for m in MODEL_ORDER for s in STRATEGY_ORDER if (m, s) not in values]
    if missing:
        raise ValueError(f"Missing model-strategy cells: {missing}")
    return np.array([[values[(m, s)] for s in STRATEGY_ORDER] for m in MODEL_ORDER])


def load_contribution(path: Path) -> list[dict[str, float | str]]:
    rows = read_csv(path)
    rows.sort(key=lambda r: int(float(r["Rank"])))
    rows = rows[:N_FEATURES_PANEL_B]
    return [
        {
            "rank": int(float(row["Rank"])),
            "feature": row["feature"],
            "score": float(row["Consensus_score"]),
            "contribution": float(row["Contribution_pct"]) * 100.0,
        }
        for row in rows
    ]


def load_topk(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    rows.sort(key=lambda r: int(float(r["TopK"])))
    return {
        "k": np.array([int(float(row["TopK"])) for row in rows]),
        "dev": np.array([float(row["DEV_AUPRC"]) for row in rows]),
        "val": np.array([float(row["INDEP_AUPRC"]) for row in rows]),
    }


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.0,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_letter(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
    )


def lighten(hex_color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * amount)


def draw_panel_a(ax: plt.Axes, matrix: np.ndarray) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "journal_blue",
        ["#f7fbfd", "#d8ecf2", "#94c7d9", "#2376a6"],
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0.285, vmax=0.465, aspect="equal")

    ax.set_xticks(np.arange(len(STRATEGY_ORDER)), labels=STRATEGY_ORDER)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), labels=[MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=5, colors=INK)
    ax.tick_params(axis="y", left=False, colors=INK)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_color(INK)

    ax.set_xticks(np.arange(-0.5, len(STRATEGY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value >= 0.405 else INK
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=11.0, fontweight="bold", color=color)

    for spine in ax.spines.values():
        spine.set_color("#cbd7dd")
        spine.set_linewidth(0.8)


def draw_panel_b(ax: plt.Axes, rows: list[dict[str, float | str]]) -> None:
    y = np.arange(len(rows))
    scores = np.array([float(row["score"]) for row in rows])
    contrib = np.array([float(row["contribution"]) for row in rows])
    labels = [str(row["feature"]) for row in rows]
    ranks = [int(row["rank"]) for row in rows]

    colors = [BLUE, "#1f6675", "#2f7d84", "#4b9694"] + [lighten("#cad9df", 0.30 + i * 0.035) for i in range(len(rows) - K_OPT)]
    ax.barh(y, scores, height=0.62, color=colors, edgecolor="white", linewidth=0.6, zorder=2)

    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.10, 1.23)
    ax.set_xlabel("Consensus score", color=MUTED, labelpad=6)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.grid(axis="x", color=GRID, linestyle="-", linewidth=0.55, alpha=0.85, zorder=0)
    ax.tick_params(axis="y", length=0, colors=INK, pad=3)
    ax.tick_params(axis="x", colors=MUTED, length=3)

    for tick, row_index in zip(ax.get_yticklabels(), range(len(rows))):
        tick.set_fontweight("bold" if row_index < K_OPT else "normal")
        tick.set_color(INK if row_index < K_OPT else MUTED)

    for i, (rank, score, cp) in enumerate(zip(ranks, scores, contrib)):
        score_label = "<0.01" if 0 <= score < 0.01 else f"{score:.3f}"
        contribution_label = "<0.01%" if 0 <= cp < 0.01 else f"{cp:.1f}%"
        ax.text(-0.06, i, str(rank), ha="center", va="center", fontsize=7.8, color="#9aa8b0")
        ax.text(score + 0.015, i, score_label, ha="left", va="center", fontsize=8.0, color=INK if i < K_OPT else MUTED)
        ax.text(1.17, i, contribution_label, ha="center", va="center", fontsize=8.0, color=INK if i < K_OPT else MUTED, fontweight="bold" if i < K_OPT else "normal")

    ax.text(-0.06, -0.90, "Rank", ha="center", va="center", fontsize=8.0, color=MUTED)
    ax.text(1.17, -0.90, "Contribution", ha="center", va="center", fontsize=8.0, color=MUTED)
    ax.axhline(K_OPT - 0.5, color=GRID, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cfd9de")
    ax.spines["bottom"].set_linewidth(0.7)


def draw_topk_axis(
    ax: plt.Axes,
    k: np.ndarray,
    dev_values: np.ndarray,
    val_values: np.ndarray,
) -> None:
    ax.plot(
        k, dev_values, color=BLUE, linewidth=1.9, marker="o",
        markersize=4.0, markerfacecolor=BLUE, markeredgecolor="white",
        markeredgewidth=0.6, zorder=3, label="DEV set CV"
    )
    ax.plot(
        k, val_values, color=TEAL, linewidth=1.9, marker="o",
        markersize=4.0, markerfacecolor=TEAL, markeredgecolor="white",
        markeredgewidth=0.6, zorder=3, label="VAL set"
    )
    ax.axvline(K_OPT, color=WARM, linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.75, zorder=1)

    idx = int(np.where(k == K_OPT)[0][0])
    ax.scatter([K_OPT], [dev_values[idx]], s=62, facecolor="white", edgecolor=BLUE, linewidth=1.5, zorder=4)
    ax.scatter([K_OPT], [val_values[idx]], s=62, facecolor="white", edgecolor=TEAL, linewidth=1.5, zorder=4)
    ax.annotate(
        f"K = {K_OPT}",
        xy=(K_OPT, max(dev_values[idx], val_values[idx])),
        xytext=(8, 18),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )
    ax.annotate(
        f"DEV {dev_values[idx]:.3f}",
        xy=(K_OPT, dev_values[idx]), xytext=(10, 2),
        textcoords="offset points", ha="left", va="center",
        fontsize=9.5, color=BLUE,
    )
    ax.annotate(
        f"VAL {val_values[idx]:.3f}",
        xy=(K_OPT, val_values[idx]), xytext=(10, -12),
        textcoords="offset points", ha="left", va="center",
        fontsize=9.5, color=TEAL,
    )

    ax.set_ylim(0.30, 0.62)
    ax.set_yticks([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    ax.set_xticks(k)
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.9)
    ax.grid(axis="x", visible=False)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("Number of top-ranked features (K)", color=INK, labelpad=7)
    ax.set_ylabel("AUPRC", color=INK, labelpad=8)
    ax.legend(
        loc="upper right", frameon=True, framealpha=0.96,
        edgecolor="#d2dde3", fontsize=10.2, handlelength=2.1,
        borderpad=0.50, labelspacing=0.40
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cfd9de")
    ax.spines["bottom"].set_color("#cfd9de")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def draw_figure(matrix: np.ndarray, topk: dict[str, np.ndarray]) -> plt.Figure:
    set_style()
    fig = plt.figure(figsize=(11.2, 4.80), dpi=300)

    # Detailed MCFS ranking is now reported in Table 2. Figure 2 focuses on
    # the two decisions made during model development: model/strategy selection
    # and final feature-number selection.
    ax_a = fig.add_axes([0.095, 0.150, 0.385, 0.720])
    ax_b = fig.add_axes([0.550, 0.180, 0.405, 0.660])

    draw_panel_a(ax_a, matrix)
    draw_topk_axis(ax_b, topk["k"], topk["dev"], topk["val"])

    fig.text(0.045, 0.880, "A", ha="left", va="center", fontsize=15, fontweight="bold", color=NAVY)
    fig.text(0.510, 0.880, "B", ha="left", va="center", fontsize=15, fontweight="bold", color=NAVY)
    return fig


def save_figure(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        (out_prefix.with_suffix(".png"), 300),
        (out_prefix.with_name(out_prefix.name + "_600dpi").with_suffix(".png"), 600),
        (out_prefix.with_suffix(".pdf"), 600),
        (out_prefix.with_suffix(".svg"), 600),
    ]
    for path, dpi in outputs:
        fig.savefig(path, dpi=dpi, facecolor="white")
        print(f"Wrote: {path}")


def main(argv: Iterable[str] | None = None) -> None:
    default_model_csv, default_contrib_csv, default_topk_csv, default_out_prefix = default_paths()
    parser = argparse.ArgumentParser(description="Generate integrated Figure 2.")
    parser.add_argument("--model-csv", type=Path, default=default_model_csv)
    parser.add_argument("--contrib-csv", type=Path, default=default_contrib_csv)
    parser.add_argument("--topk-csv", type=Path, default=default_topk_csv)
    parser.add_argument("--out-prefix", type=Path, default=default_out_prefix)
    args = parser.parse_args(argv)

    matrix = load_heatmap(args.model_csv)
    topk = load_topk(args.topk_csv)

    fig = draw_figure(matrix, topk)
    save_figure(fig, args.out_prefix)
    plt.close(fig)


if __name__ == "__main__":
    main()
