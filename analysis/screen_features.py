"""Screen predictors using contribution direction and cross-validation stability.

Scheme2 is the manuscript's primary mean-based screening procedure.
Scheme1 uses median-based positive-contribution screening for sensitivity
analysis. Both retain the original coefficient-of-variation thresholds."""

from pathlib import Path
import numpy as np
import pandas as pd

IMPORTANCE_ROOT = Path("outputs/feature_importance")
ALL_SUMMARY_CSV = IMPORTANCE_ROOT / "all_schemes_feature_importance_summary.csv"
SCHEME_DIR_ROOT = Path("outputs/feature_selection")
SCHEME_DIR_ROOT.mkdir(parents=True, exist_ok=True)
PI_CV_THRESH = 5.0
PVC_CV_THRESH = 1.0
LFC_CV_THRESH = 5.0
GAIN_CV_THRESH = 2.0
ET_CV_THRESH = 2.0
EPS = 1e-08


def to_num_safe(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_required_base_cols(df: pd.DataFrame):
    required = [
        "Scheme",
        "Model",
        "Strategy",
        "feature",
        "PI_mean",
        "PI_median",
        "PI_std",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_cv(std_s: pd.Series, mean_s: pd.Series, eps: float = 1e-08) -> pd.Series:
    """
    CV = std / (abs(mean) + eps)
    """
    std_s = pd.to_numeric(std_s, errors="coerce")
    mean_s = pd.to_numeric(mean_s, errors="coerce")
    return std_s / (mean_s.abs() + float(eps))


def add_cv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"PI_mean", "PI_std"}.issubset(out.columns):
        out["PI_CV"] = compute_cv(out["PI_std"], out["PI_mean"], EPS)
    if {"PVC_mean", "PVC_std"}.issubset(out.columns):
        out["PVC_CV"] = compute_cv(out["PVC_std"], out["PVC_mean"], EPS)
    if {"LFC_mean", "LFC_std"}.issubset(out.columns):
        out["LFC_CV"] = compute_cv(out["LFC_std"], out["LFC_mean"], EPS)
    if {"GAIN_mean", "GAIN_std"}.issubset(out.columns):
        out["GAIN_CV"] = compute_cv(out["GAIN_std"], out["GAIN_mean"], EPS)
    if {"ET_IMP_mean", "ET_IMP_std"}.issubset(out.columns):
        out["ET_IMP_CV"] = compute_cv(out["ET_IMP_std"], out["ET_IMP_mean"], EPS)
    return out


def apply_scheme1(df: pd.DataFrame) -> pd.DataFrame:
    """Apply median-based contribution screening and the original stability thresholds for sensitivity analysis."""
    out = df.copy()
    model = str(out["Model"].iloc[0])
    out["pass_PI_center"] = out["PI_median"] > 0
    out["pass_PI_stability"] = out["PI_CV"] <= PI_CV_THRESH
    for c in [
        "pass_PVC_center",
        "pass_LFC_center",
        "pass_PVC_stability",
        "pass_LFC_stability",
        "pass_GAIN_center",
        "pass_GAIN_stability",
        "pass_ET_center",
        "pass_ET_stability",
    ]:
        out[c] = np.nan
    if model == "CatBoost":
        need_cols = ["PVC_median", "LFC_median", "PVC_CV", "LFC_CV"]
        for c in need_cols:
            if c not in out.columns:
                raise ValueError(f"{model} summary missing column: {c}")
        out["pass_PVC_center"] = out["PVC_median"] > 0
        out["pass_LFC_center"] = out["LFC_median"] > 0
        out["pass_PVC_stability"] = out["PVC_CV"] <= PVC_CV_THRESH
        out["pass_LFC_stability"] = out["LFC_CV"] <= LFC_CV_THRESH
        out["pass_model_internal"] = (
            out["pass_PVC_center"]
            & out["pass_LFC_center"]
            & out["pass_PVC_stability"]
            & out["pass_LFC_stability"]
        )
    elif model == "XGBoost":
        need_cols = ["GAIN_median", "GAIN_CV"]
        for c in need_cols:
            if c not in out.columns:
                raise ValueError(f"{model} summary missing column: {c}")
        out["pass_GAIN_center"] = out["GAIN_median"] > 0
        out["pass_GAIN_stability"] = out["GAIN_CV"] <= GAIN_CV_THRESH
        out["pass_model_internal"] = (
            out["pass_GAIN_center"] & out["pass_GAIN_stability"]
        )
    elif model == "ExtraTrees":
        need_cols = ["ET_IMP_median", "ET_IMP_CV"]
        for c in need_cols:
            if c not in out.columns:
                raise ValueError(f"{model} summary missing column: {c}")
        out["pass_ET_center"] = out["ET_IMP_median"] > 0
        out["pass_ET_stability"] = out["ET_IMP_CV"] <= ET_CV_THRESH
        out["pass_model_internal"] = out["pass_ET_center"] & out["pass_ET_stability"]
    else:
        raise ValueError(f"Unsupported model for current rule set: {model}")
    out["PI_CV_THRESH_USED"] = PI_CV_THRESH
    out["PVC_CV_THRESH_USED"] = PVC_CV_THRESH
    out["LFC_CV_THRESH_USED"] = LFC_CV_THRESH
    out["GAIN_CV_THRESH_USED"] = GAIN_CV_THRESH
    out["ET_CV_THRESH_USED"] = ET_CV_THRESH
    out["keep_scheme1"] = (
        out["pass_PI_center"] & out["pass_PI_stability"] & out["pass_model_internal"]
    )
    out = out.sort_values(
        by=["keep_scheme1", "PI_median", "PI_mean", "feature"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return out


def apply_scheme2(df: pd.DataFrame) -> pd.DataFrame:
    """Apply mean-based contribution screening and the original stability thresholds for the primary analysis."""
    out = df.copy()
    model = str(out["Model"].iloc[0])
    out["pass_PI_center"] = out["PI_mean"] > 0
    out["pass_PI_stability"] = out["PI_CV"] <= PI_CV_THRESH
    for c in [
        "pass_PVC_center",
        "pass_LFC_center",
        "pass_PVC_stability",
        "pass_LFC_stability",
        "pass_GAIN_center",
        "pass_GAIN_stability",
        "pass_ET_center",
        "pass_ET_stability",
    ]:
        out[c] = np.nan
    if model == "CatBoost":
        need_cols = ["PVC_mean", "LFC_mean", "PVC_CV", "LFC_CV"]
        for c in need_cols:
            if c not in out.columns:
                raise ValueError(f"{model} summary missing column: {c}")
        out["pass_PVC_center"] = out["PVC_mean"] > 0
        out["pass_LFC_center"] = out["LFC_mean"] > 0
        out["pass_PVC_stability"] = out["PVC_CV"] <= PVC_CV_THRESH
        out["pass_LFC_stability"] = out["LFC_CV"] <= LFC_CV_THRESH
        out["pass_model_internal"] = (
            out["pass_PVC_center"]
            & out["pass_LFC_center"]
            & out["pass_PVC_stability"]
            & out["pass_LFC_stability"]
        )
    elif model == "XGBoost":
        need_cols = ["GAIN_mean", "GAIN_CV"]
        for c in need_cols:
            if c not in out.columns:
                raise ValueError(f"{model} summary missing column: {c}")
        out["pass_GAIN_center"] = out["GAIN_mean"] > 0
        out["pass_GAIN_stability"] = out["GAIN_CV"] <= GAIN_CV_THRESH
        out["pass_model_internal"] = (
            out["pass_GAIN_center"] & out["pass_GAIN_stability"]
        )
    elif model == "ExtraTrees":
        need_cols = ["ET_IMP_mean", "ET_IMP_CV"]
        for c in need_cols:
            if c not in out.columns:
                raise ValueError(f"{model} summary missing column: {c}")
        out["pass_ET_center"] = out["ET_IMP_mean"] > 0
        out["pass_ET_stability"] = out["ET_IMP_CV"] <= ET_CV_THRESH
        out["pass_model_internal"] = out["pass_ET_center"] & out["pass_ET_stability"]
    else:
        raise ValueError(f"Unsupported model for current rule set: {model}")
    out["PI_CV_THRESH_USED"] = PI_CV_THRESH
    out["PVC_CV_THRESH_USED"] = PVC_CV_THRESH
    out["LFC_CV_THRESH_USED"] = LFC_CV_THRESH
    out["GAIN_CV_THRESH_USED"] = GAIN_CV_THRESH
    out["ET_CV_THRESH_USED"] = ET_CV_THRESH
    out["keep_scheme2"] = (
        out["pass_PI_center"] & out["pass_PI_stability"] & out["pass_model_internal"]
    )
    out = out.sort_values(
        by=["keep_scheme2", "PI_mean", "PI_median", "feature"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return out


def select_output_columns_scheme1(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "Scheme",
        "Model",
        "Strategy",
        "feature",
        "PI_mean",
        "PI_median",
        "PI_std",
        "PI_CV",
    ]
    extra_cols = []
    model = str(df["Model"].iloc[0])
    if model == "CatBoost":
        extra_cols += [
            "PVC_mean",
            "PVC_median",
            "PVC_std",
            "PVC_CV",
            "LFC_mean",
            "LFC_median",
            "LFC_std",
            "LFC_CV",
        ]
    elif model == "XGBoost":
        extra_cols += ["GAIN_mean", "GAIN_median", "GAIN_std", "GAIN_CV"]
    elif model == "ExtraTrees":
        extra_cols += ["ET_IMP_mean", "ET_IMP_median", "ET_IMP_std", "ET_IMP_CV"]
    flag_cols = [
        "pass_PI_center",
        "pass_PI_stability",
        "pass_model_internal",
        "keep_scheme1",
        "pass_PVC_center",
        "pass_LFC_center",
        "pass_PVC_stability",
        "pass_LFC_stability",
        "pass_GAIN_center",
        "pass_GAIN_stability",
        "pass_ET_center",
        "pass_ET_stability",
        "PI_CV_THRESH_USED",
        "PVC_CV_THRESH_USED",
        "LFC_CV_THRESH_USED",
        "GAIN_CV_THRESH_USED",
        "ET_CV_THRESH_USED",
    ]
    cols = [c for c in base_cols + extra_cols + flag_cols if c in df.columns]
    out = df[cols].copy()
    num_cols = [
        c
        for c in out.columns
        if c
        not in [
            "Scheme",
            "Model",
            "Strategy",
            "feature",
            "pass_PI_center",
            "pass_PI_stability",
            "pass_model_internal",
            "keep_scheme1",
            "pass_PVC_center",
            "pass_LFC_center",
            "pass_PVC_stability",
            "pass_LFC_stability",
            "pass_GAIN_center",
            "pass_GAIN_stability",
            "pass_ET_center",
            "pass_ET_stability",
        ]
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def select_output_columns_scheme2(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "Scheme",
        "Model",
        "Strategy",
        "feature",
        "PI_mean",
        "PI_median",
        "PI_std",
        "PI_CV",
    ]
    extra_cols = []
    model = str(df["Model"].iloc[0])
    if model == "CatBoost":
        extra_cols += [
            "PVC_mean",
            "PVC_median",
            "PVC_std",
            "PVC_CV",
            "LFC_mean",
            "LFC_median",
            "LFC_std",
            "LFC_CV",
        ]
    elif model == "XGBoost":
        extra_cols += ["GAIN_mean", "GAIN_median", "GAIN_std", "GAIN_CV"]
    elif model == "ExtraTrees":
        extra_cols += ["ET_IMP_mean", "ET_IMP_median", "ET_IMP_std", "ET_IMP_CV"]
    flag_cols = [
        "pass_PI_center",
        "pass_PI_stability",
        "pass_model_internal",
        "keep_scheme2",
        "pass_PVC_center",
        "pass_LFC_center",
        "pass_PVC_stability",
        "pass_LFC_stability",
        "pass_GAIN_center",
        "pass_GAIN_stability",
        "pass_ET_center",
        "pass_ET_stability",
        "PI_CV_THRESH_USED",
        "PVC_CV_THRESH_USED",
        "LFC_CV_THRESH_USED",
        "GAIN_CV_THRESH_USED",
        "ET_CV_THRESH_USED",
    ]
    cols = [c for c in base_cols + extra_cols + flag_cols if c in df.columns]
    out = df[cols].copy()
    num_cols = [
        c
        for c in out.columns
        if c
        not in [
            "Scheme",
            "Model",
            "Strategy",
            "feature",
            "pass_PI_center",
            "pass_PI_stability",
            "pass_model_internal",
            "keep_scheme2",
            "pass_PVC_center",
            "pass_LFC_center",
            "pass_PVC_stability",
            "pass_LFC_stability",
            "pass_GAIN_center",
            "pass_GAIN_stability",
            "pass_ET_center",
            "pass_ET_stability",
        ]
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def build_kept_only_summary(
    df: pd.DataFrame, keep_col: str, center_col: str
) -> pd.DataFrame:
    """Return retained predictors for downstream evaluation."""
    kept = df[df[keep_col] == True].copy()
    if kept.empty:
        return kept
    sort_cols = [center_col, "feature"]
    ascending = [False, True]
    kept = kept.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)
    return kept


def build_scheme_summary(
    scheme: str, model: str, strategy: str, out1: pd.DataFrame, out2: pd.DataFrame
) -> pd.DataFrame:
    row = {
        "Scheme": scheme,
        "Model": model,
        "Strategy": strategy,
        "n_total_features": int(len(out1)),
        "n_keep_scheme1": (
            int(out1["keep_scheme1"].sum()) if "keep_scheme1" in out1.columns else 0
        ),
        "n_keep_scheme2": (
            int(out2["keep_scheme2"].sum()) if "keep_scheme2" in out2.columns else 0
        ),
        "PI_CV_THRESH_USED": float(PI_CV_THRESH),
        "PVC_CV_THRESH_USED": float(PVC_CV_THRESH),
        "LFC_CV_THRESH_USED": float(LFC_CV_THRESH),
        "GAIN_CV_THRESH_USED": float(GAIN_CV_THRESH),
        "ET_CV_THRESH_USED": float(ET_CV_THRESH),
    }
    return pd.DataFrame([row])


def main():
    print("[DEBUG] RUNNING FILE =", __file__)
    print("[DEBUG] PI_CV_THRESH =", PI_CV_THRESH)
    print("[DEBUG] PVC_CV_THRESH =", PVC_CV_THRESH)
    print("[DEBUG] LFC_CV_THRESH =", LFC_CV_THRESH)
    print("[DEBUG] GAIN_CV_THRESH =", GAIN_CV_THRESH)
    print("[DEBUG] ET_CV_THRESH =", ET_CV_THRESH)
    if not ALL_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Not found: {ALL_SUMMARY_CSV.resolve()}")
    all_df = pd.read_csv(ALL_SUMMARY_CSV)
    ensure_required_base_cols(all_df)
    numeric_candidates = [
        "PI_mean",
        "PI_median",
        "PI_std",
        "PVC_mean",
        "PVC_median",
        "PVC_std",
        "LFC_mean",
        "LFC_median",
        "LFC_std",
        "GAIN_mean",
        "GAIN_median",
        "GAIN_std",
        "ET_IMP_mean",
        "ET_IMP_median",
        "ET_IMP_std",
    ]
    all_df = to_num_safe(all_df, numeric_candidates)
    all_df = add_cv_columns(all_df)
    scheme_list = all_df["Scheme"].astype(str).unique().tolist()
    all_scheme1_outputs = []
    all_scheme2_outputs = []
    all_scheme1_kept = []
    all_scheme2_kept = []
    all_scheme_summary = []
    print("[INFO] Found schemes:", scheme_list)
    print(
        f"[INFO] CV thresholds | PI={PI_CV_THRESH}, PVC={PVC_CV_THRESH}, LFC={LFC_CV_THRESH}, GAIN={GAIN_CV_THRESH}, ET={ET_CV_THRESH}, EPS={EPS}"
    )
    for scheme in scheme_list:
        sub = (
            all_df[all_df["Scheme"].astype(str) == scheme].copy().reset_index(drop=True)
        )
        if sub.empty:
            continue
        model = str(sub["Model"].iloc[0])
        strategy = str(sub["Strategy"].iloc[0])
        print(
            f"\n[RUN] Scheme={scheme} | Model={model} | Strategy={strategy} | n_features={len(sub)}"
        )
        scheme_dir = SCHEME_DIR_ROOT / scheme
        if not scheme_dir.exists():
            print(f"[WARN] Scheme directory not found, creating: {scheme_dir}")
            scheme_dir.mkdir(parents=True, exist_ok=True)
        res1 = apply_scheme1(sub)
        out1 = select_output_columns_scheme1(res1)
        path1 = scheme_dir / "selected_features_scheme1.csv"
        print(f"[DEBUG] Writing: {path1}")
        out1.to_csv(path1, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {path1} | kept={int(out1['keep_scheme1'].sum())}")
        kept1 = build_kept_only_summary(
            out1, keep_col="keep_scheme1", center_col="PI_median"
        )
        kept1_path = scheme_dir / "selected_features_scheme1_kept_only.csv"
        print(f"[DEBUG] Writing: {kept1_path}")
        kept1.to_csv(kept1_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {kept1_path}")
        all_scheme1_outputs.append(out1)
        all_scheme1_kept.append(kept1)
        res2 = apply_scheme2(sub)
        out2 = select_output_columns_scheme2(res2)
        path2 = scheme_dir / "selected_features_scheme2.csv"
        print(f"[DEBUG] Writing: {path2}")
        out2.to_csv(path2, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {path2} | kept={int(out2['keep_scheme2'].sum())}")
        kept2 = build_kept_only_summary(
            out2, keep_col="keep_scheme2", center_col="PI_mean"
        )
        kept2_path = scheme_dir / "selected_features_scheme2_kept_only.csv"
        print(f"[DEBUG] Writing: {kept2_path}")
        kept2.to_csv(kept2_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {kept2_path}")
        all_scheme2_outputs.append(out2)
        all_scheme2_kept.append(kept2)
        scheme_summary = build_scheme_summary(scheme, model, strategy, out1, out2)
        scheme_summary_path = scheme_dir / "selection_summary.csv"
        print(f"[DEBUG] Writing: {scheme_summary_path}")
        scheme_summary.to_csv(scheme_summary_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {scheme_summary_path}")
        all_scheme_summary.append(scheme_summary)
    if len(all_scheme1_outputs) > 0:
        all_scheme1_df = pd.concat(all_scheme1_outputs, axis=0, ignore_index=True)
        path_all_1 = IMPORTANCE_ROOT / "all_schemes_selected_features_scheme1.csv"
        all_scheme1_df.to_csv(path_all_1, index=False, encoding="utf-8-sig")
        print(f"\n[SAVE] {path_all_1}")
    if len(all_scheme2_outputs) > 0:
        all_scheme2_df = pd.concat(all_scheme2_outputs, axis=0, ignore_index=True)
        path_all_2 = IMPORTANCE_ROOT / "all_schemes_selected_features_scheme2.csv"
        all_scheme2_df.to_csv(path_all_2, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {path_all_2}")
    if len(all_scheme1_kept) > 0:
        all_scheme1_kept_df = pd.concat(all_scheme1_kept, axis=0, ignore_index=True)
        path_all_1_kept = (
            IMPORTANCE_ROOT / "all_schemes_selected_features_scheme1_kept_only.csv"
        )
        all_scheme1_kept_df.to_csv(path_all_1_kept, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {path_all_1_kept}")
    if len(all_scheme2_kept) > 0:
        all_scheme2_kept_df = pd.concat(all_scheme2_kept, axis=0, ignore_index=True)
        path_all_2_kept = (
            IMPORTANCE_ROOT / "all_schemes_selected_features_scheme2_kept_only.csv"
        )
        all_scheme2_kept_df.to_csv(path_all_2_kept, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {path_all_2_kept}")
    if len(all_scheme_summary) > 0:
        all_scheme_summary_df = pd.concat(all_scheme_summary, axis=0, ignore_index=True)
        path_all_summary = IMPORTANCE_ROOT / "all_schemes_selection_summary.csv"
        all_scheme_summary_df.to_csv(
            path_all_summary, index=False, encoding="utf-8-sig"
        )
        print(f"[SAVE] {path_all_summary}")
    print("\n[FINISHED] Done.")


if __name__ == "__main__":
    main()
