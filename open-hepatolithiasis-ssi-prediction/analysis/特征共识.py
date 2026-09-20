# -*- coding: utf-8 -*-
"""
feature_contribution_consensus.py

作用：
1) 遍历“特征筛选/”下所有方案文件夹
2) 分别读取：
   - selected_features_scheme1.csv
   - selected_features_scheme2.csv
3) 仅保留：
   - scheme1: keep_scheme1 == True
   - scheme2: keep_scheme2 == True
4) 在“入选特征集合内部”做 Min-Max normalization
5) 采用几何均值融合得到 Consensus_score
6) 再标准化为 Contribution_pct
7) 输出每个方案、每种筛选方案下的特征贡献表

输出：
- 特征筛选/<scheme>/contribution_scheme1.csv
- 特征筛选/<scheme>/contribution_scheme2.csv
- 特征筛选/all_schemes_contribution_scheme1.csv
- 特征筛选/all_schemes_contribution_scheme2.csv

融合规则：
------------------------------------------------
CatBoost:
    Consensus_score = (PI_norm * PVC_norm * LFC_norm)^(1/3)

XGBoost:
    Consensus_score = (PI_norm * GAIN_norm)^(1/2)

ExtraTrees:
    Consensus_score = (PI_norm * ET_IMP_norm)^(1/2)
------------------------------------------------
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# Config (EDIT HERE ONLY)
# =========================
INPUT_ROOT = Path("特征筛选共识")
OUTPUT_ROOT = INPUT_ROOT   # 输出也放回各方案目录
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================
def minmax_norm_in_selected_set(x: pd.Series) -> pd.Series:
    """
    在当前入选特征集合内部做 Min-Max normalization
    特殊情况：
    - 如果全为空，返回全 NaN
    - 如果 max == min，返回全 1.0（表示都一样重要，不人为压成 0）
    """
    x = pd.to_numeric(x, errors="coerce")

    valid = x.dropna()
    if len(valid) == 0:
        return pd.Series([np.nan] * len(x), index=x.index)

    xmin = valid.min()
    xmax = valid.max()

    if xmax == xmin:
        return pd.Series([1.0 if pd.notna(v) else np.nan for v in x], index=x.index)

    return (x - xmin) / (xmax - xmin)


def geometric_mean_safe(arrs):
    """
    arrs: list of 1D numpy arrays / Series
    要求各路都 >= 0
    如果有 NaN，则结果为 NaN
    """
    stacked = np.vstack(arrs).astype(float)  # shape = (k, n)
    with np.errstate(invalid="ignore"):
        out = np.prod(stacked, axis=0) ** (1.0 / stacked.shape[0])
    return out


def normalize_contribution(consensus_score: pd.Series) -> pd.Series:
    s = pd.to_numeric(consensus_score, errors="coerce")
    denom = s.sum(skipna=True)
    if pd.isna(denom) or denom <= 0:
        return pd.Series([np.nan] * len(s), index=s.index)
    return s / denom


def load_selected_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path.resolve()}")
    df = pd.read_csv(path)
    return df


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    s2 = s.astype(str).str.strip().str.lower()
    return s2.isin(["true", "1", "yes", "y", "t"])


def process_one_scheme_file(df: pd.DataFrame, scheme_type: str) -> pd.DataFrame:
    """
    scheme_type: "scheme1" or "scheme2"
    """
    if "Scheme" not in df.columns or "Model" not in df.columns or "feature" not in df.columns:
        raise ValueError("Input csv missing one of required columns: Scheme / Model / feature")

    keep_col = "keep_scheme1" if scheme_type == "scheme1" else "keep_scheme2"
    if keep_col not in df.columns:
        raise ValueError(f"Input csv missing keep column: {keep_col}")

    df = df.copy()
    df[keep_col] = to_bool_series(df[keep_col])

    sel = df[df[keep_col]].copy().reset_index(drop=True)
    if len(sel) == 0:
        # 返回空表，但保留列结构
        return pd.DataFrame(columns=[
            "Scheme", "Model", "Strategy", "feature",
            "PI_used", "Internal1_used", "Internal2_used",
            "PI_norm", "Internal1_norm", "Internal2_norm",
            "Consensus_score", "Contribution_pct", "Rank"
        ])

    model = str(sel["Model"].iloc[0])

    # -------------------------
    # 1) 确定 PI 用哪一列
    # -------------------------
    # selected_features_scheme1.csv 对应 median PI
    # selected_features_scheme2.csv 对应 mean PI
    if scheme_type == "scheme1":
        pi_col = "PI_median"
    else:
        pi_col = "PI_mean"

    if pi_col not in sel.columns:
        raise ValueError(f"{model} selected csv missing PI column: {pi_col}")

    # -------------------------
    # 2) 模型内重要性列
    # -------------------------
    if model == "CatBoost":
        need_cols = ["PVC_mean", "LFC_mean"]
        for c in need_cols:
            if c not in sel.columns:
                raise ValueError(f"{model} selected csv missing column: {c}")

        sel["PI_used"] = pd.to_numeric(sel[pi_col], errors="coerce")
        sel["Internal1_used"] = pd.to_numeric(sel["PVC_mean"], errors="coerce")
        sel["Internal2_used"] = pd.to_numeric(sel["LFC_mean"], errors="coerce")

        sel["PI_norm"] = minmax_norm_in_selected_set(sel["PI_used"])
        sel["Internal1_norm"] = minmax_norm_in_selected_set(sel["Internal1_used"])
        sel["Internal2_norm"] = minmax_norm_in_selected_set(sel["Internal2_used"])

        sel["Consensus_score"] = geometric_mean_safe([
            sel["PI_norm"].values,
            sel["Internal1_norm"].values,
            sel["Internal2_norm"].values,
        ])

    elif model == "XGBoost":
        need_cols = ["GAIN_mean"]
        for c in need_cols:
            if c not in sel.columns:
                raise ValueError(f"{model} selected csv missing column: {c}")

        sel["PI_used"] = pd.to_numeric(sel[pi_col], errors="coerce")
        sel["Internal1_used"] = pd.to_numeric(sel["GAIN_mean"], errors="coerce")
        sel["Internal2_used"] = np.nan

        sel["PI_norm"] = minmax_norm_in_selected_set(sel["PI_used"])
        sel["Internal1_norm"] = minmax_norm_in_selected_set(sel["Internal1_used"])
        sel["Internal2_norm"] = np.nan

        sel["Consensus_score"] = geometric_mean_safe([
            sel["PI_norm"].values,
            sel["Internal1_norm"].values,
        ])

    elif model == "ExtraTrees":
        need_cols = ["ET_IMP_mean"]
        for c in need_cols:
            if c not in sel.columns:
                raise ValueError(f"{model} selected csv missing column: {c}")

        sel["PI_used"] = pd.to_numeric(sel[pi_col], errors="coerce")
        sel["Internal1_used"] = pd.to_numeric(sel["ET_IMP_mean"], errors="coerce")
        sel["Internal2_used"] = np.nan

        sel["PI_norm"] = minmax_norm_in_selected_set(sel["PI_used"])
        sel["Internal1_norm"] = minmax_norm_in_selected_set(sel["Internal1_used"])
        sel["Internal2_norm"] = np.nan

        sel["Consensus_score"] = geometric_mean_safe([
            sel["PI_norm"].values,
            sel["Internal1_norm"].values,
        ])

    else:
        raise ValueError(f"Unsupported model: {model}")

    # -------------------------
    # 3) Contribution_pct
    # -------------------------
    sel["Contribution_pct"] = normalize_contribution(sel["Consensus_score"])

    # -------------------------
    # 4) 排序
    # -------------------------
    sel = sel.sort_values(
        by=["Contribution_pct", "Consensus_score", "feature"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    sel["Rank"] = np.arange(1, len(sel) + 1)

    # 美化
    keep_cols = [
        "Scheme", "Model", "Strategy", "feature",
        "PI_used", "Internal1_used", "Internal2_used",
        "PI_norm", "Internal1_norm", "Internal2_norm",
        "Consensus_score", "Contribution_pct", "Rank"
    ]
    out = sel[keep_cols].copy()

    num_cols = [c for c in out.columns if c not in ["Scheme", "Model", "Strategy", "feature", "Rank"]]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)

    return out


# =========================
# Main
# =========================
def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT.resolve()}")

    scheme_dirs = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]
    scheme_dirs = sorted(scheme_dirs, key=lambda x: x.name)

    all_scheme1_outputs = []
    all_scheme2_outputs = []

    print("[INFO] Found scheme dirs:", [p.name for p in scheme_dirs])

    for scheme_dir in scheme_dirs:
        print(f"\n[RUN] Processing: {scheme_dir.name}")

        path1 = scheme_dir / "selected_features_scheme1.csv"
        path2 = scheme_dir / "selected_features_scheme2.csv"

        # -------- scheme1 --------
        if path1.exists():
            df1 = load_selected_csv(path1)
            out1 = process_one_scheme_file(df1, scheme_type="scheme1")
            save1 = scheme_dir / "contribution_scheme1.csv"
            out1.to_csv(save1, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {save1} | n_selected={len(out1)}")
            all_scheme1_outputs.append(out1)
        else:
            print(f"[WARN] Missing: {path1}")

        # -------- scheme2 --------
        if path2.exists():
            df2 = load_selected_csv(path2)
            out2 = process_one_scheme_file(df2, scheme_type="scheme2")
            save2 = scheme_dir / "contribution_scheme2.csv"
            out2.to_csv(save2, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {save2} | n_selected={len(out2)}")
            all_scheme2_outputs.append(out2)
        else:
            print(f"[WARN] Missing: {path2}")

    # 全部方案汇总
    if len(all_scheme1_outputs) > 0:
        all1 = pd.concat(all_scheme1_outputs, axis=0, ignore_index=True)
        save_all1 = OUTPUT_ROOT / "all_schemes_contribution_scheme1.csv"
        all1.to_csv(save_all1, index=False, encoding="utf-8-sig")
        print(f"\n[SAVE] {save_all1}")

    if len(all_scheme2_outputs) > 0:
        all2 = pd.concat(all_scheme2_outputs, axis=0, ignore_index=True)
        save_all2 = OUTPUT_ROOT / "all_schemes_contribution_scheme2.csv"
        all2.to_csv(save_all2, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {save_all2}")

    print("\n[FINISHED] Done.")


if __name__ == "__main__":
    main()