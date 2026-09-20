"""Retune, calibrate, and evaluate the selected four-predictor CatBoost model.

Retune hyperparameters using DEV cross-validation and refit on DEV.
The uncalibrated threshold uses DEV out-of-fold predictions. Platt and
isotonic thresholds use fitted DEV ensemble probabilities, which are not
strictly out-of-fold. Preserve this study procedure and its limitation.
Bootstrap intervals describe performance in the fixed held-out sample."""

import os
import json
import ast
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

TARGET_SCHEME = "CatBoost+S0"
TARGET_TOPK = 4
SCHEME_TYPE = "scheme2"
SEED = 42
N_FOLDS = 5
N_BOOTSTRAPS = 1000
RETUNE_HYPERPARAMS = True
CAT_DEPTHS = [3, 4, 6, 8]
CAT_LRS = [0.03, 0.1]
CAT_ITERS = [300, 600, 1000]
DATA_CSV = Path("data/clinical_data.csv")
SPLIT_CSV = Path("data/split_folds.csv")
FEATURE_TYPES_CSV = Path("feature_schema.csv")
SCHEME_PARAMS_CSV = Path("outputs/model_selection/scheme_to_best_params.csv")
CONTRIBUTION_CSV = Path(
    f"outputs/feature_selection/{TARGET_SCHEME}/contribution_{SCHEME_TYPE}.csv"
)
OUTPUT_ROOT = Path("outputs/final_model")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
EPS = 1e-15
EXCLUDE_COLS = ["ID", "Infection", "Infection_split"]
SMOTE_K_NEIGHBORS = 3


def fix_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def predict_proba_pos(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    z = model.decision_function(X)
    return 1 / (1 + np.exp(-z))


def safe_basic_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    out = {}
    out["AUROC"] = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    out["AUPRC"] = average_precision_score(y_true, p)
    out["Brier_Score"] = brier_score_loss(y_true, p)
    p_clip = np.clip(p, EPS, 1 - EPS)
    out["LogLoss"] = log_loss(y_true, p_clip)
    return out


def confusion_metrics(
    y_true: np.ndarray, p: np.ndarray, thr: float
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    y_hat = (p >= float(thr)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn > 0 else np.nan
    spec = tn / (tn + fp) if tn + fp > 0 else np.nan
    ppv = tp / (tp + fp) if tp + fp > 0 else np.nan
    npv = tn / (tn + fn) if tn + fn > 0 else np.nan
    f1 = (
        2 * ppv * sens / (ppv + sens)
        if ppv == ppv and sens == sens and (ppv + sens > 0)
        else np.nan
    )
    acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else np.nan
    bal_acc = (sens + spec) / 2 if sens == sens and spec == spec else np.nan
    youden = sens + spec - 1 if sens == sens and spec == spec else np.nan
    fpr = fp / (fp + tn) if fp + tn > 0 else np.nan
    fnr = fn / (fn + tp) if fn + tp > 0 else np.nan
    return {
        "Sensitivity": float(sens) if sens == sens else np.nan,
        "Specificity": float(spec) if spec == spec else np.nan,
        "PPV": float(ppv) if ppv == ppv else np.nan,
        "NPV": float(npv) if npv == npv else np.nan,
        "F1": float(f1) if f1 == f1 else np.nan,
        "Accuracy": float(acc) if acc == acc else np.nan,
        "BalancedAcc": float(bal_acc) if bal_acc == bal_acc else np.nan,
        "YoudenJ": float(youden) if youden == youden else np.nan,
        "FPR": float(fpr) if fpr == fpr else np.nan,
        "FNR": float(fnr) if fnr == fnr else np.nan,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def find_best_threshold_youden(
    y_true: np.ndarray, p: np.ndarray, n_thresh: int = 201
) -> Tuple[float, Dict[str, float]]:
    """Select the threshold maximizing Youden J on the supplied predictions."""
    thresholds = np.linspace(0.0, 1.0, int(n_thresh))
    rows = []
    for t in thresholds:
        cm = confusion_metrics(y_true, p, t)
        rows.append({"Threshold": float(t), **cm})
    df = (
        pd.DataFrame(rows)
        .sort_values(
            by=["YoudenJ", "Sensitivity", "PPV", "Threshold"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    best_thr = float(df.loc[0, "Threshold"])
    best_row = df.loc[0].to_dict()
    best_row.pop("Threshold", None)
    return (best_thr, best_row)


def generate_ci_report(
    y_true: np.ndarray,
    p: np.ndarray,
    thr: float,
    n_bootstraps: int = 1000,
    ci: int = 95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Format original-sample metrics with percentile bootstrap confidence intervals."""
    point_basic = safe_basic_metrics(y_true, p)
    point_cm = confusion_metrics(y_true, p, thr)
    point_all = {**point_basic, **point_cm}
    np.random.seed(seed)
    n_samples = len(y_true)
    boot_results = {k: [] for k in point_all.keys()}
    print(
        f"      [INFO] Computing {n_bootstraps} bootstrap draws for 95% confidence intervals..."
    )
    for _ in range(n_bootstraps):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_b = y_true[idx]
        p_b = p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        b_basic = safe_basic_metrics(y_b, p_b)
        b_cm = confusion_metrics(y_b, p_b, thr)
        for k, v in b_basic.items():
            boot_results[k].append(v)
        for k, v in b_cm.items():
            boot_results[k].append(v)
    lower_p = (100 - ci) / 2.0
    upper_p = 100 - lower_p
    final_row = {}
    for k, v in point_all.items():
        if k in ["TP", "FP", "TN", "FN"]:
            final_row[k] = int(v)
        elif k in boot_results and len(boot_results[k]) > 0:
            valid_boots = [x for x in boot_results[k] if not np.isnan(x)]
            if len(valid_boots) > 0:
                lower = np.percentile(valid_boots, lower_p)
                upper = np.percentile(valid_boots, upper_p)
                final_row[k] = f"{v:.4f} ({lower:.4f}-{upper:.4f})"
            else:
                final_row[k] = f"{v:.4f} (NaN-NaN)"
        else:
            final_row[k] = f"{v:.4f}"
    final_row["Thr_used"] = f"{thr:.4f}"
    return final_row


def detect_indep_split(sp: pd.DataFrame) -> str:
    splits = sp["split"].astype(str).str.upper().unique().tolist()
    for cand in ["VAL", "INDEP_TEST", "TEST"]:
        if cand in splits:
            return cand
    raise ValueError(f"Cannot find independent split among {splits}.")


def parse_params_string(params_str: str) -> Dict[str, Any]:
    s = str(params_str).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    raise ValueError(f"Cannot parse params string: {s}")


def load_target_scheme_config(csv_path: Path, target_scheme: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    hit = df[df["scheme"].astype(str) == str(target_scheme)].copy()
    if hit.empty:
        raise ValueError(f"Scheme={target_scheme} not found in {csv_path}.")
    row = hit.iloc[0]
    return {
        "scheme": str(row["scheme"]),
        "model": str(row["model"]),
        "strategy": str(row["strategy"]),
        "params": parse_params_string(row["params"]),
    }


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    continuous_cols: List[str], binary_cols: List[str], cat_cols: List[str]
) -> ColumnTransformer:
    transformers = []
    if continuous_cols:
        transformers.append(
            (
                "cont",
                Pipeline(
                    [("imp", SimpleImputer(strategy="median", add_indicator=True))]
                ),
                continuous_cols,
            )
        )
    if binary_cols:
        transformers.append(
            (
                "bin",
                Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]),
                binary_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "to_str",
                            FunctionTransformer(
                                lambda x: x.astype(str), feature_names_out="one-to-one"
                            ),
                        ),
                        ("ohe", make_ohe()),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def build_estimator(
    model_name: str, params: Dict[str, Any], strategy: str, pos_weight: float, seed: int
):
    if model_name == "CatBoost":
        spw = float(pos_weight) if strategy == "S1" else 1.0
        return CatBoostClassifier(
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            thread_count=-1,
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            iterations=int(params["iterations"]),
            l2_leaf_reg=3.0,
            scale_pos_weight=spw,
        )
    raise ValueError(
        f"Only CatBoost configured in this simplified version. Modify if needed."
    )


def build_pipeline(
    preprocessor: ColumnTransformer, estimator, strategy: str, seed: int
):
    if strategy in ("S2", "S3"):
        ratio = 1.0 / 3.0 if strategy == "S2" else 1.0 / 2.0
        smote = SMOTE(
            sampling_strategy=ratio, k_neighbors=SMOTE_K_NEIGHBORS, random_state=seed
        )
        return IMBPipeline(
            [("prep", preprocessor), ("smote", smote), ("clf", estimator)]
        )
    return Pipeline([("prep", preprocessor), ("clf", estimator)])


def build_catboost_grid():
    """Construct the CatBoost hyperparameter grid for the reduced feature set."""
    grid = []
    for depth in CAT_DEPTHS:
        for lr in CAT_LRS:
            for it in CAT_ITERS:
                grid.append({"depth": depth, "learning_rate": lr, "iterations": it})
    return grid


def cv_evaluate_params(
    model_name: str,
    params: Dict[str, Any],
    strategy: str,
    preprocessor: ColumnTransformer,
    feature_cols: List[str],
    df_idx: pd.DataFrame,
    sp_dev: pd.DataFrame,
    y_col: str,
    folds: List[int],
    seed: int,
) -> Dict[str, float]:
    """Evaluate a parameter configuration by mean five-fold DEV performance."""
    aurocs, auprcs, briers, loglosses = ([], [], [], [])
    for f in folds:
        tr_ids = sp_dev.loc[sp_dev["fold_id"] != f, "ID"].tolist()
        va_ids = sp_dev.loc[sp_dev["fold_id"] == f, "ID"].tolist()
        tr = df_idx.loc[tr_ids]
        va = df_idx.loc[va_ids]
        Xtr = tr[feature_cols]
        ytr = tr[y_col].values.astype(int)
        Xva = va[feature_cols]
        yva = va[y_col].values.astype(int)
        n_pos = int(ytr.sum())
        n_neg = int((ytr == 0).sum())
        pos_weight = n_neg / max(1, n_pos)
        est = build_estimator(model_name, params, strategy, pos_weight, seed)
        pipe = build_pipeline(preprocessor, est, strategy, seed)
        try:
            pipe.fit(Xtr, ytr)
            p = predict_proba_pos(pipe, Xva)
            m = safe_basic_metrics(yva, p)
            aurocs.append(m["AUROC"])
            auprcs.append(m["AUPRC"])
            briers.append(m["Brier_Score"])
            loglosses.append(m["LogLoss"])
        except Exception as e:
            continue
    if len(auprcs) == 0:
        return {
            "mean_AUPRC": -1.0,
            "mean_AUROC": -1.0,
            "mean_Brier": 1.0,
            "mean_LogLoss": 1.0,
        }
    return {
        "mean_AUPRC": float(np.mean(auprcs)),
        "mean_AUROC": float(np.mean(aurocs)),
        "mean_Brier": float(np.mean(briers)),
        "mean_LogLoss": float(np.mean(loglosses)),
    }


def retune_hyperparams(
    model_name: str,
    strategy: str,
    preprocessor: ColumnTransformer,
    feature_cols: List[str],
    df_idx: pd.DataFrame,
    sp_dev: pd.DataFrame,
    y_col: str,
    folds: List[int],
    seed: int,
) -> Dict[str, Any]:
    """Select reduced-model hyperparameters using DEV grid search."""
    print("\n" + "=" * 60)
    print(
        f"[RETUNE] Grid search on the reduced feature set ({len(feature_cols)} predictors)..."
    )
    print("=" * 60)
    if model_name == "CatBoost":
        grid = build_catboost_grid()
    else:
        raise ValueError(f"Retuning not implemented for model: {model_name}")
    results = []
    for params in grid:
        metrics = cv_evaluate_params(
            model_name,
            params,
            strategy,
            preprocessor,
            feature_cols,
            df_idx,
            sp_dev,
            y_col,
            folds,
            seed,
        )
        results.append({**params, **metrics})
        print(f"  [CV] {params} -> AUPRC={metrics['mean_AUPRC']:.4f}")
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by=["mean_AUPRC", "mean_AUROC", "mean_Brier"], ascending=[False, False, True]
    ).reset_index(drop=True)
    best = result_df.iloc[0]
    best_params = {k: best[k] for k in grid[0].keys()}
    print(f"\n[RETUNE] Selected parameters: {best_params}")
    print(f"[RETUNE] Best mean CV average precision: {best['mean_AUPRC']:.4f}")
    tune_path = OUTPUT_ROOT / f"retune_grid_{TARGET_SCHEME}_Top{TARGET_TOPK}.csv"
    result_df.to_csv(tune_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Hyperparameter search results: {tune_path}")
    return best_params


def cv_get_base_oof(
    model_name: str,
    params: Dict[str, Any],
    strategy: str,
    preprocessor: ColumnTransformer,
    feature_cols: List[str],
    df_idx: pd.DataFrame,
    sp_dev: pd.DataFrame,
    y_col: str,
    folds: List[int],
    seed: int,
) -> np.ndarray:
    """Generate five-fold DEV out-of-fold probabilities for base-model threshold selection."""
    n = len(sp_dev)
    oof = np.zeros(n, dtype=float)
    for f in folds:
        tr_ids = sp_dev.loc[sp_dev["fold_id"] != f, "ID"].tolist()
        va_ids = sp_dev.loc[sp_dev["fold_id"] == f, "ID"].tolist()
        va_pos = np.where(sp_dev["fold_id"].values == f)[0]
        tr = df_idx.loc[tr_ids]
        va = df_idx.loc[va_ids]
        Xtr_f = tr[feature_cols]
        ytr_f = tr[y_col].values.astype(int)
        Xva_f = va[feature_cols]
        n_pos = int(ytr_f.sum())
        n_neg = int((ytr_f == 0).sum())
        pos_weight = n_neg / max(1, n_pos)
        est = build_estimator(model_name, params, strategy, pos_weight, seed)
        pipe = build_pipeline(preprocessor, est, strategy, seed)
        pipe.fit(Xtr_f, ytr_f)
        oof[va_pos] = predict_proba_pos(pipe, Xva_f)
    return oof


def main():
    fix_seed(SEED)
    print(f"Final held-out evaluation with bootstrap confidence intervals")
    print(f"Model configuration: {TARGET_SCHEME} | TopK: {TARGET_TOPK}")
    if not CONTRIBUTION_CSV.exists():
        raise FileNotFoundError(f"Missing: {CONTRIBUTION_CSV}")
    contrib_df = pd.read_csv(CONTRIBUTION_CSV)
    contrib_df["Rank"] = pd.to_numeric(contrib_df["Rank"], errors="coerce")
    contrib_df = contrib_df.sort_values("Rank")
    features = contrib_df["feature"].head(TARGET_TOPK).tolist()
    print(f"[INFO] Selected clinical predictors ({len(features)}): {features}")
    cfg = load_target_scheme_config(SCHEME_PARAMS_CSV, TARGET_SCHEME)
    model_name, strategy, params_old = (cfg["model"], cfg["strategy"], cfg["params"])
    df = pd.read_csv(DATA_CSV)
    sp = pd.read_csv(SPLIT_CSV)
    indep_split = detect_indep_split(sp)
    df = df.merge(
        sp[["ID", "split", "Infection"]], on="ID", how="inner", suffixes=("", "_split")
    )
    y_col = "Infection_split" if "Infection_split" in df.columns else "Infection"
    df[y_col] = pd.to_numeric(df[y_col], errors="raise").astype(int)
    sp_dev = df[df["split"].astype(str).str.upper() == "DEV"].copy()
    sp_indep = df[df["split"].astype(str).str.upper() == indep_split].copy()
    exclude_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    ft = pd.read_csv(FEATURE_TYPES_CSV)
    cont_cols = ft[ft["type"].str.lower() == "continuous"]["feature"].tolist()
    bin_cols = ft[ft["type"].str.lower() == "binary"]["feature"].tolist()
    cat_cols = ft[ft["type"].str.lower().isin(["categorical", "category"])][
        "feature"
    ].tolist()
    cont_cols = [c for c in cont_cols if c in features]
    bin_cols = [c for c in bin_cols if c in features]
    cat_cols = [c for c in cat_cols if c in features]
    preprocessor = build_preprocessor(cont_cols, bin_cols, cat_cols)
    Xtr = sp_dev[features]
    ytr = sp_dev[y_col].values.astype(int)
    Xte = sp_indep[features]
    yte = sp_indep[y_col].values.astype(int)
    n_pos, n_neg = (int(ytr.sum()), int((ytr == 0).sum()))
    pos_weight = n_neg / max(1, n_pos)
    if RETUNE_HYPERPARAMS:
        sp_dev_with_fold = sp_dev.copy()
        sp_dev_with_fold["fold_id"] = sp_dev_with_fold["ID"].map(
            sp[sp["split"].astype(str).str.upper() == "DEV"].set_index("ID")["fold_id"]
        )
        folds = sorted(sp_dev_with_fold["fold_id"].unique().tolist())
        df_idx = df.set_index("ID", drop=False)
        best_params = retune_hyperparams(
            model_name=model_name,
            strategy=strategy,
            preprocessor=preprocessor,
            feature_cols=features,
            df_idx=df_idx,
            sp_dev=sp_dev_with_fold,
            y_col=y_col,
            folds=folds,
            seed=SEED,
        )
        print(f"\n[INFO] Selected reduced-model parameters: {best_params}")
        print(f"[INFO] Original full-model parameters: {params_old}")
        params = best_params
    else:
        params = params_old
        print(f"[INFO] Reusing full-model parameters: {params}")
    print(
        "\n[RUN] Generating base-model DEV OOF predictions for threshold selection..."
    )
    p_base_oof = cv_get_base_oof(
        model_name=model_name,
        params=params,
        strategy=strategy,
        preprocessor=preprocessor,
        feature_cols=features,
        df_idx=df_idx,
        sp_dev=sp_dev_with_fold,
        y_col=y_col,
        folds=folds,
        seed=SEED,
    )
    best_thr_base, _ = find_best_threshold_youden(ytr, p_base_oof)
    print(f"  [INFO] Base-model DEV OOF threshold: {best_thr_base:.6f}")
    print("[RUN] Fitting final uncalibrated model...")
    est_base = build_estimator(model_name, params, strategy, pos_weight, SEED)
    pipe_base = build_pipeline(preprocessor, est_base, strategy, SEED)
    pipe_base.fit(Xtr, ytr)
    p_uncalib = predict_proba_pos(pipe_base, Xte)
    print("[RUN] Fitting Platt-calibrated model...")
    est_platt = build_estimator(model_name, params, strategy, pos_weight, SEED)
    pipe_platt = build_pipeline(preprocessor, est_platt, strategy, SEED)
    calib_platt = CalibratedClassifierCV(pipe_platt, method="sigmoid", cv=5)
    calib_platt.fit(Xtr, ytr)
    p_platt_oof = predict_proba_pos(calib_platt, Xtr)
    best_thr_platt, _ = find_best_threshold_youden(ytr, p_platt_oof)
    print(f"  [INFO] Platt fitted-DEV ensemble threshold: {best_thr_platt:.6f}")
    p_platt = predict_proba_pos(calib_platt, Xte)
    print("[RUN] Fitting isotonic-calibrated model...")
    est_iso = build_estimator(model_name, params, strategy, pos_weight, SEED)
    pipe_iso = build_pipeline(preprocessor, est_iso, strategy, SEED)
    calib_iso = CalibratedClassifierCV(pipe_iso, method="isotonic", cv=5)
    calib_iso.fit(Xtr, ytr)
    p_iso_oof = predict_proba_pos(calib_iso, Xtr)
    best_thr_iso, _ = find_best_threshold_youden(ytr, p_iso_oof)
    print(f"  [INFO] Isotonic fitted-DEV ensemble threshold: {best_thr_iso:.6f}")
    p_iso = predict_proba_pos(calib_iso, Xte)
    print("\n========== Held-out performance with 95% confidence intervals ==========")
    report_rows = []
    print("\n[EVALUATE] Uncalibrated (Base) ...")
    report_rows.append(
        {
            "Model_Status": "Uncalibrated (Base)",
            **generate_ci_report(yte, p_uncalib, best_thr_base, N_BOOTSTRAPS, 95, SEED),
        }
    )
    print("\n[EVALUATE] Platt (Sigmoid) Calibrated ...")
    report_rows.append(
        {
            "Model_Status": "Platt (Sigmoid) Calibrated",
            **generate_ci_report(yte, p_platt, best_thr_platt, N_BOOTSTRAPS, 95, SEED),
        }
    )
    print("\n[EVALUATE] Isotonic Calibrated ...")
    report_rows.append(
        {
            "Model_Status": "Isotonic Calibrated",
            **generate_ci_report(yte, p_iso, best_thr_iso, N_BOOTSTRAPS, 95, SEED),
        }
    )
    report_df = pd.DataFrame(report_rows)
    print_cols = [
        "Model_Status",
        "AUROC",
        "AUPRC",
        "Brier_Score",
        "Thr_used",
        "Sensitivity",
        "Specificity",
    ]
    print("\n" + report_df[print_cols].to_string(index=False))
    report_path = (
        OUTPUT_ROOT / f"Final_Metrics_with_CI_{TARGET_SCHEME}_Top{TARGET_TOPK}.csv"
    )
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    pred_df = pd.DataFrame(
        {
            "ID": sp_indep["ID"].values,
            "y_true": yte,
            "prob_uncalib": p_uncalib,
            "prob_platt": p_platt,
            "prob_iso": p_iso,
        }
    )
    pred_path = OUTPUT_ROOT / f"Final_Predictions_{TARGET_SCHEME}_Top{TARGET_TOPK}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"\n[SUCCESS] Analysis complete.")
    print(
        f"[SAVE] Final performance metrics with bootstrap confidence intervals: {report_path}"
    )
    print(f"[SAVE] Calibration and decision-curve predictions: {pred_path}")


if __name__ == "__main__":
    main()
