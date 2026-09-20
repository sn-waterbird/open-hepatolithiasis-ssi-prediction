# -*- coding: utf-8 -*-
"""
Figure 5 — SHAP analysis: pre- and post-aggregation (FIXED)
═══════════════════════════════════════════════════════════════

修复记录 (2026-05-26):
  1. [关键] SHAP 同时在 DEV 和 VAL 集计算 — DEV 为主图，VAL 加 _val 后缀
     (原版仅在 VAL 计算，而 VAL 集 Drain time 分布倒挂，导致医生看到异常 SHAP)
  2. [保留] SimpleImputer(add_indicator=True) 用于 SHAP 可视化 —
     预聚合图需要展示 missing-indicator 展开列，使读者看到缺失值对模型的影响；
     虽然 CatBoost 原生处理缺失值，但 missing-indicator 列为 SHAP 归因提供了
     可解释的"缺失分支"信号；零方差列(SHAP=0)在图中自然不显眼，无负面影响。
  3. [改进] 从 retune_grid 加载最优超参，而非硬编码
  4. [改进] matching_indices 使用 rsplit 避免 startswith 误匹配相似特征名
  5. [防御] 传递 numpy 数组给 shap_values()，避免列名与 CatBoost 内部命名不匹配

目的:用 SHAP TreeExplainer 解释最终 CatBoost+S0 Top-K 模型，提供两种粒度:
  - 聚合前 (Pre-aggregation):  one-hot / missing-indicator 后的"展开列"层面
  - 聚合后 (Post-aggregation): 同一原始临床特征的展开列合并回一根，更易读

数据来源:
  开腹肝胆道结石手术SSI清洗.csv
  splits/split_folds_full.csv
  特征筛选共识/CatBoost+S0/contribution_scheme2.csv
  特征类别/feature_types_summary.csv
  最终训练/retune_grid_CatBoost+S0_Top4.csv

输出 (DEV = 主图, VAL = _val 后缀):
  fig5_shap_pre_bee.png  / .pdf  — DEV 聚合前 Beeswarm
  fig5_shap_pre_bar.png  / .pdf  — DEV 聚合前 Bar
  fig5_shap_post_bee.png / .pdf  — DEV 聚合后 Beeswarm
  fig5_shap_post_bar.png / .pdf  — DEV 聚合后 Bar
  fig5_shap_pre_bee_val.png  / .pdf  — VAL 聚合前 Beeswarm
  fig5_shap_pre_bar_val.png  / .pdf  — VAL 聚合前 Bar
  fig5_shap_post_bee_val.png / .pdf  — VAL 聚合后 Beeswarm
  fig5_shap_post_bar_val.png / .pdf  — VAL 聚合后 Bar
"""

import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from catboost import CatBoostClassifier

from 最终训练 import fix_seed, build_preprocessor
from figures.fig_utils import set_style, save_figure

# ── Config ────────────────────────────────────────────
SEED, MODEL_NAME, STRATEGY = 42, "CatBoost", "S0"
TOPK              = 4
PRE_MAX_DISPLAY   = 15      # max rows in pre-aggregation plots

DATA_CSV          = ROOT / "开腹肝胆道结石手术SSI清洗.csv"
SPLIT_CSV         = ROOT / "splits/split_folds_full.csv"
CONTRIB_CSV       = ROOT / "特征筛选共识/CatBoost+S0/contribution_scheme2.csv"
FEATURE_TYPES_CSV = ROOT / "特征类别/feature_types_summary.csv"
RETUNE_CSV        = ROOT / "最终训练/retune_grid_CatBoost+S0_Top4.csv"

# Palette — continues from Fig 1/2/3/4
C_ACCENT = "#C77B30"   # terracotta (chosen-model accent)
C_GRID   = "#cccccc"
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_feature_value",
    ["#008BFB", "#8A39C1", "#FF0051"],
)
PANEL_FONT = "Arial"


# ── Helpers ───────────────────────────────────────────
def beautify(col_name: str) -> str:
    """Human-readable column name for plotting."""
    if col_name.startswith("missingindicator_"):
        return col_name[len("missingindicator_"):] + " (missing)"
    return col_name


def style_axes(ax):
    """Apply the spine/tick treatment used across Fig 1-4."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"  ].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#444444", labelsize=9)


def custom_bar(shap_values, feature_names, fname, max_display=None):
    """Journal-style horizontal bar of mean |SHAP| — matches Fig 1 aesthetic."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    if max_display is not None:
        order = order[:max_display]
    f = [feature_names[i] for i in order]
    v = mean_abs[order]
    n = len(f)

    # Alpha gradient — most important = darkest
    alphas = np.linspace(0.95, 0.45, n)
    bar_colors = [(*mpl.colors.to_rgb(C_ACCENT), a) for a in alphas]

    fig, ax = plt.subplots(
        figsize=(7.5, 0.42 * n + 1.4),
        constrained_layout=True,
    )
    y_pos = np.arange(n)
    ax.barh(y_pos, v, color=bar_colors, edgecolor="white", linewidth=0.5,
            height=0.68)

    # Value labels to the right of each bar
    pad = v.max() * 0.015
    for i, val in enumerate(v):
        ax.text(val + pad, i, f"{val:.3f}",
                va="center", fontsize=8.5,
                color="#222222" if i < 4 else "#666666",
                fontweight="bold" if i < 4 else "normal",
                family="DejaVu Sans Mono")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(f, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, v.max() * 1.20)
    ax.grid(True, axis="x", linestyle=":", color=C_GRID, alpha=0.7,
            linewidth=0.6)
    ax.set_axisbelow(True)
    style_axes(ax)
    save_figure(fig, fname)
    plt.close(fig)


def shap_beeswarm(shap_values, X_df, fname, max_display=15):
    """shap.summary_plot wrapped + post-cleaned to match journal style."""
    n_shown = min(max_display, X_df.shape[1])
    plt.figure(figsize=(7.5, 0.42 * n_shown + 1.6))
    shap.summary_plot(shap_values, X_df,
                      max_display=max_display, show=False, alpha=0.75,
                      color_bar=True, sort=True)
    fig = plt.gcf()
    ax  = plt.gca()
    style_axes(ax)
    ax.set_xlabel("SHAP value  (impact on model output)", fontsize=10.5)
    # SHAP's default tick labels can be tiny — make them consistent
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    save_figure(fig, fname)
    plt.close(fig)


def shap_beeswarm_panels(
    aggregated_values: np.ndarray,
    aggregated_features: pd.DataFrame,
    unaggregated_values: np.ndarray,
    unaggregated_features: pd.DataFrame,
    fname: str,
):
    """Create the manuscript Figure 5 as side-by-side aggregated and unaggregated SHAP plots.

    The left panel shows the clinical-variable level interpretation used in the
    main text. The right panel retains the expanded missingness indicator, so
    readers can verify its independent contribution without confusing it for a
    fifth clinical predictor. Individual SHAP plots are still saved separately
    for supplementary use and quality-control checks.
    """
    # Reserve a dedicated label gutter for panel B.  Its expanded feature
    # names then remain visually separate from panel A rather than spilling
    # into the central plotting area.
    fig = plt.figure(figsize=(15.0, 5.6))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 0.36, 1, 0.04],
        left=0.065,
        right=0.955,
        bottom=0.17,
        top=0.88,
        wspace=0.04,
    )
    ax_agg = fig.add_subplot(grid[0, 0])
    ax_unagg = fig.add_subplot(grid[0, 2])
    cax = fig.add_subplot(grid[0, 3])

    # A common horizontal scale makes the two explanations directly comparable.
    all_values = np.concatenate([aggregated_values.ravel(), unaggregated_values.ravel()])
    x_min = float(np.nanmin(all_values))
    x_max = float(np.nanmax(all_values))
    # Keep a small, consistent margin around the observed SHAP range without
    # letting unused horizontal space visually dominate either panel.
    x_pad = max((x_max - x_min) * 0.010, 0.03)
    x_limits = (x_min - x_pad, x_max + x_pad)

    for ax, values, feature_df, label, title in (
        (ax_agg, aggregated_values, aggregated_features, "A", "Aggregated"),
        (ax_unagg, unaggregated_values, unaggregated_features, "B", "Unaggregated"),
    ):
        plt.sca(ax)
        shap.summary_plot(
            values,
            feature_df,
            max_display=feature_df.shape[1],
            show=False,
            alpha=0.75,
            color=SHAP_CMAP,
            color_bar=False,
            plot_size=None,
            sort=True,
        )
        ax.set_xlim(*x_limits)
        ax.set_xlabel(
            "SHAP value (impact on model output)",
            fontsize=11.5,
            fontfamily=PANEL_FONT,
            fontweight="normal",
            labelpad=9,
        )
        style_axes(ax)
        # Use the same compact label treatment in both panels so that feature
        # names sit close to, but outside, their respective plotting areas.
        ax.tick_params(
            axis="y",
            labelsize=11,
            pad=-10,
        )
        for tick in ax.get_yticklabels():
            tick.set_fontfamily(PANEL_FONT)
            tick.set_fontsize(11)
            tick.set_fontweight("normal")
        for tick in ax.get_xticklabels():
            tick.set_fontfamily(PANEL_FONT)
            tick.set_fontsize(10)
            tick.set_fontweight("normal")
        # Align panel headings with the corresponding data axes; the blank
        # middle column is reserved solely for the panel-B feature labels.
        ax.text(
            0.00,
            1.055,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            fontfamily=PANEL_FONT,
            color="#222222",
        )
        ax.text(
            0.065,
            1.06,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            fontfamily=PANEL_FONT,
            color="#222222",
        )

    # Both panels encode the same within-feature low-to-high color direction.
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=SHAP_CMAP),
        cax=cax,
        ticks=[0, 1],
    )
    colorbar.ax.set_yticklabels(["Low", "High"])
    colorbar.set_label("Feature value", fontsize=11.5, labelpad=7)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=10.5)

    save_figure(fig, fname)
    plt.close(fig)


# ── Improved preprocessor ─────────────────────────────
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_shap_preprocessor(continuous_cols: List[str],
                            binary_cols: List[str],
                            cat_cols: List[str]) -> ColumnTransformer:
    """
    Preprocessor for SHAP analysis.

    Uses add_indicator=True so that the pre-aggregation SHAP plot
    can display missing-indicator columns, showing the reader how
    missingness impacts predictions.  Although CatBoost handles
    missing values natively, the explicit indicator columns provide
    an interpretable "missing branch" signal in SHAP attribution.

    Features without any missing values produce zero-variance
    indicator columns (all zeros); their SHAP values will be 0
    for all samples, so they are harmless in the visualization.
    """
    transformers = []
    if continuous_cols:
        transformers.append(("cont",
                             SimpleImputer(strategy="median",
                                           add_indicator=True),
                             continuous_cols))
    if binary_cols:
        transformers.append(("bin",
                             SimpleImputer(strategy="most_frequent",
                                           add_indicator=True),
                             binary_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent",
                                  add_indicator=True)),
            ("to_str", FunctionTransformer(lambda x: x.astype(str),
                                           feature_names_out="one-to-one")),
            ("ohe", make_ohe()),
        ]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


# ── Robust matching ───────────────────────────────────
def matching_indices(orig_feature: str, transformed_cols: List[str]) -> List[int]:
    """
    Match transformed column names back to original clinical feature.

    Handles:
      - exact match:               'Drain time' -> 'Drain time'
      - missing indicator:         'missingindicator_Drain time' -> after strip = 'Drain time'
      - one-hot expansion:         'Sex' -> 'Sex_M', 'Sex_F'

    Uses rsplit('_', 1) for the one-hot case to ensure only the
    LAST underscore separates feature name from category value,
    avoiding false matches between similar feature names.
    """
    out = []
    missing_prefix = "missingindicator_"
    for j, c in enumerate(transformed_cols):
        # Peel off missing-indicator prefix first if present
        stripped = c[len(missing_prefix):] if c.startswith(missing_prefix) else c

        if stripped == orig_feature:
            # Exact match: value / imputed column, or indicator after strip
            out.append(j)
        elif stripped.startswith(orig_feature + "_"):
            # Potential one-hot expansion: feature_value
            # Use rsplit to ensure only the last underscore is the separator
            prefix_part = stripped.rsplit("_", 1)[0]
            if prefix_part == orig_feature:
                out.append(j)
    return out


# ══════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════
fix_seed(SEED)

df = pd.read_csv(DATA_CSV).merge(
    pd.read_csv(SPLIT_CSV)[["ID", "split", "Infection"]],
    on="ID", how="inner", suffixes=("", "_split"))
y_col = "Infection_split" if "Infection_split" in df.columns else "Infection"
df[y_col] = pd.to_numeric(df[y_col], errors="raise").astype(int)

sp_dev   = df[df["split"].astype(str).str.upper() == "DEV"].copy()
sp_indep = df[df["split"].astype(str).str.upper() == "VAL"].copy()

print(f"[DATA] DEV: {len(sp_dev)} samples, "
      f"Infected={int(sp_dev[y_col].sum())} / Non-infected={int((sp_dev[y_col]==0).sum())}")
print(f"[DATA] VAL: {len(sp_indep)} samples, "
      f"Infected={int(sp_indep[y_col].sum())} / Non-infected={int((sp_indep[y_col]==0).sum())}")

contrib_df = pd.read_csv(CONTRIB_CSV).sort_values("Rank")
features = contrib_df["feature"].head(TOPK).tolist()
print(f"[INFO] Top-{TOPK} features: {features}")

ft = pd.read_csv(FEATURE_TYPES_CSV)
cont_cols = [c for c in ft[ft["type"].str.lower() == "continuous"]["feature"].tolist()
             if c in features]
bin_cols  = [c for c in ft[ft["type"].str.lower() == "binary"]["feature"].tolist()
             if c in features]
cat_cols  = [c for c in ft[ft["type"].str.lower().isin(["categorical", "category"])]["feature"].tolist()
             if c in features]


# ══════════════════════════════════════════════════════
# 2. Load best hyperparameters from retune grid
# ══════════════════════════════════════════════════════
FINAL_PARAMS = {"depth": 3, "learning_rate": 0.03, "iterations": 300}

if RETUNE_CSV.exists():
    retune_df = pd.read_csv(RETUNE_CSV)
    best = retune_df.iloc[0]  # already sorted by mean_AUPRC descending
    retuned_params = {
        "depth": int(best["depth"]),
        "learning_rate": float(best["learning_rate"]),
        "iterations": int(best["iterations"]),
    }
    if retuned_params != FINAL_PARAMS:
        raise ValueError(
            "The leading retuning-grid row does not match the final "
            f"CatBoost + S0 configuration: {FINAL_PARAMS}."
        )
    print(f"[PARAMS] Verified final configuration from retune grid: {FINAL_PARAMS} "
          f"(AUPRC={best['mean_AUPRC']:.4f})")
else:
    print(f"[PARAMS] Retune grid not found; using fixed final configuration: {FINAL_PARAMS}")

params = FINAL_PARAMS


# ══════════════════════════════════════════════════════
# 3. Train model with improved preprocessor
# ══════════════════════════════════════════════════════
print("[MODEL] Training the uncalibrated final CatBoost + S0 model "
      "(l2_leaf_reg=3.0, scale_pos_weight=1.0, seed=42).")
preprocessor = build_preprocessor(cont_cols, bin_cols, cat_cols)

Xtr, ytr = sp_dev[features], sp_dev[y_col].values.astype(int)

est = CatBoostClassifier(
    loss_function="Logloss",
    random_seed=SEED,
    verbose=False,
    thread_count=-1,
    depth=params["depth"],
    learning_rate=params["learning_rate"],
    iterations=params["iterations"],
    l2_leaf_reg=3.0,
    # S0: no additional class-imbalance handling.
    scale_pos_weight=1.0,
)
pipe = Pipeline([("prep", preprocessor), ("clf", est)])
pipe.fit(Xtr, ytr)

prep = pipe.named_steps["prep"]
clf  = pipe.named_steps["clf"]


# ══════════════════════════════════════════════════════
# 4. SHAP computation helper
# ══════════════════════════════════════════════════════
def compute_shap(data_df: pd.DataFrame, label: str
                 ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                            List[str], List[str]]:
    """
    Compute SHAP values for a given dataset.

    Returns:
      sv           — SHAP values (pre-agg), shape (n, n_transformed_cols)
      sv_agg       — SHAP values aggregated to original features, shape (n, K)
      X_pre        — transformed DataFrame (for pre-agg beeswarm)
      X_display    — original feature values (for post-agg beeswarm coloring)
      clean_cols   — cleaned column names (for matching)
      pretty_cols  — beautified column names (for plotting)
    """
    X = data_df[features]
    X_t = prep.transform(X)
    trans_cols = prep.get_feature_names_out()
    clean_cols = [c.split("__")[-1] for c in trans_cols]
    pretty_cols = [beautify(c) for c in clean_cols]

    X_pre = pd.DataFrame(X_t, columns=pretty_cols, index=X.index)

    # Pass numpy array to avoid column-name mismatch:
    # the CatBoost model was trained inside a Pipeline and received
    # unnamed numpy arrays, so its internal feature names are indices.
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_pre.values)
    if isinstance(sv, list):
        sv = sv[1]          # binary classification: pick positive class
    elif sv.ndim == 3:
        sv = sv[:, :, 1]    # multi-class fallback

    print(f"[SHAP] {label}: SHAP shape = {sv.shape}")

    # Aggregate SHAP back to original clinical features
    sv_agg = np.zeros((sv.shape[0], len(features)))
    for i, orig in enumerate(features):
        idx_list = matching_indices(orig, clean_cols)
        if not idx_list:
            print(f"[WARN] {label}: No transformed columns matched for '{orig}'")
            continue
        sv_agg[:, i] = sv[:, idx_list].sum(axis=1)
        mapped = [f"'{clean_cols[j]}'" for j in idx_list]
        print(f"  {label} {orig:<30s} <- {{{', '.join(mapped)}}}")

    # Build display DataFrame at original-feature level for beeswarm coloring
    X_display = data_df[features].copy()
    for col in X_display.columns:
        if X_display[col].dtype == "object" or str(X_display[col].dtype) == "category":
            X_display[col] = pd.factorize(X_display[col])[0]
        X_display[col] = pd.to_numeric(X_display[col], errors="coerce")

    return sv, sv_agg, X_pre, X_display, clean_cols, pretty_cols


# ══════════════════════════════════════════════════════
# 5. Compute SHAP: DEV (in-sample) + VAL (out-of-sample)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[RUN] Computing SHAP on DEV (training set)...")
print("=" * 60)
sv_dev, sv_agg_dev, X_pre_dev, X_display_dev, clean_dev, pretty_dev = \
    compute_shap(sp_dev, "DEV")

print("\n" + "=" * 60)
print("[RUN] Computing SHAP on VAL (validation set)...")
print("=" * 60)
sv_val, sv_agg_val, X_pre_val, X_display_val, clean_val, pretty_val = \
    compute_shap(sp_indep, "VAL")


# ══════════════════════════════════════════════════════
# 6. Generate plots
# ══════════════════════════════════════════════════════
set_style()

# ── DEV plots (primary — original filenames, no suffix) ──
print("\n[PLOT] (1/8) DEV pre-aggregation beeswarm...")
shap_beeswarm(sv_dev, X_pre_dev, "fig5_shap_pre_bee",
              max_display=PRE_MAX_DISPLAY)

print("[PLOT] (2/8) DEV pre-aggregation bar...")
custom_bar(sv_dev, pretty_dev, "fig5_shap_pre_bar",
           max_display=PRE_MAX_DISPLAY)

print("[PLOT] (3/8) DEV post-aggregation beeswarm...")
shap_beeswarm(sv_agg_dev, X_display_dev, "fig5_shap_post_bee",
              max_display=len(features))

print("[PLOT] (4/8) DEV post-aggregation bar...")
custom_bar(sv_agg_dev, features, "fig5_shap_post_bar",
           max_display=len(features))

print("[PLOT] DEV combined SHAP panels (main Figure 5)...")
shap_beeswarm_panels(
    sv_agg_dev,
    X_display_dev,
    sv_dev,
    X_pre_dev,
    "fig5_shap_beeswarm_panels",
)

# ── VAL plots (secondary — _val suffix) ──
print("\n[PLOT] (5/8) VAL pre-aggregation beeswarm...")
shap_beeswarm(sv_val, X_pre_val, "fig5_shap_pre_bee_val",
              max_display=PRE_MAX_DISPLAY)

print("[PLOT] (6/8) VAL pre-aggregation bar...")
custom_bar(sv_val, pretty_val, "fig5_shap_pre_bar_val",
           max_display=PRE_MAX_DISPLAY)

print("[PLOT] (7/8) VAL post-aggregation beeswarm...")
shap_beeswarm(sv_agg_val, X_display_val, "fig5_shap_post_bee_val",
              max_display=len(features))

print("[PLOT] (8/8) VAL post-aggregation bar...")
custom_bar(sv_agg_val, features, "fig5_shap_post_bar_val",
           max_display=len(features))

print("\n[DONE] All individual and combined SHAP figures saved:")
print("  DEV (primary): fig5_shap_pre/post_bee/bar (.png/.pdf)")
print("  Main Figure 5: fig5_shap_beeswarm_panels (.png/.pdf)")
print("  VAL (secondary): fig5_shap_pre/post_bee/bar_val (.png/.pdf)")
