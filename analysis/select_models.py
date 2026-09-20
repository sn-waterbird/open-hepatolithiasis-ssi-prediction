"""Select model and imbalance-strategy configurations using DEV cross-validation.

Compare logistic regression, CatBoost, XGBoost, and Extra Trees across S0-S3.
Select hyperparameters by mean fold average precision, with the original
tie-breaking rules. Fit preprocessing within each training fold.
Run from the analysis directory; restricted inputs are located under data/."""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

SEED = 42
N_FOLDS = 5
DATA_CSV = Path("data/clinical_data.csv")
SPLIT_CSV = Path("data/split_folds.csv")
FEATURE_TYPES_CSV = Path("feature_schema.csv")
OUT_DIR = Path("outputs/model_selection")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDE_COLS = ["ID", "Infection", "Infection_split"]
SMOTE_K_NEIGHBORS = 3
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_REG_LAMBDA = 1.0


def fix_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def safe_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    out = {}
    out["AUROC"] = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    out["AUPRC"] = average_precision_score(y_true, p)
    out["Brier"] = brier_score_loss(y_true, p)
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)
    out["LogLoss"] = log_loss(y_true, p_clip)
    return out


def predict_proba_pos(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    z = model.decision_function(X)
    return 1 / (1 + np.exp(-z))


def load_feature_types(
    ft_path: Path, df_columns: List[str], exclude: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    ft = pd.read_csv(ft_path)
    if "feature" not in ft.columns or "type" not in ft.columns:
        raise ValueError("feature_schema.csv must contain columns: feature, type")
    feature_set = set(df_columns)
    exclude_set = set(exclude)
    cont, binary, cat = ([], [], [])
    for _, r in ft.iterrows():
        f = str(r["feature"])
        t = str(r["type"]).strip().lower()
        if f in exclude_set:
            continue
        if f not in feature_set:
            continue
        if t == "continuous":
            cont.append(f)
        elif t == "binary":
            binary.append(f)
        elif t in ("categorical", "category"):
            cat.append(f)
        else:
            cont.append(f)
    known = set(cont) | set(binary) | set(cat) | exclude_set
    leftovers = [c for c in df_columns if c not in known]
    if leftovers:
        cont += leftovers

    def uniq(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return (uniq(cont), uniq(binary), uniq(cat))


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
        cont_pipe = Pipeline(
            [("imp", SimpleImputer(strategy="median", add_indicator=True))]
        )
        transformers.append(("cont", cont_pipe, continuous_cols))
    if binary_cols:
        bin_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
        transformers.append(("bin", bin_pipe, binary_cols))
    if cat_cols:
        cat_pipe = Pipeline(
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
        )
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def lr_grid():
    return [{"C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]]


def cat_grid():
    grid = []
    for depth in [3, 4, 6, 8]:
        for lr in [0.03, 0.1]:
            for it in [300, 600, 1000]:
                grid.append({"depth": depth, "learning_rate": lr, "iterations": it})
    return grid


def xgb_grid():
    grid = []
    for md in [3, 4, 5, 6]:
        for lr in [0.03, 0.1]:
            for ne in [200, 400, 800]:
                grid.append({"max_depth": md, "learning_rate": lr, "n_estimators": ne})
    return grid


def et_grid():
    grid = []
    for ne in [300, 600, 1000]:
        for md in [None, 8, 12]:
            for msl in [1, 2]:
                grid.append(
                    {"n_estimators": ne, "max_depth": md, "min_samples_leaf": msl}
                )
    return grid


def build_estimator(
    model_name: str, params: Dict[str, Any], strategy: str, pos_weight: float, seed: int
):
    """
    strategy:
      - S1 uses class weights/scale_pos_weight
      - S2/S3 use SMOTE (handled outside), so do NOT apply class weight
    """
    if model_name == "LR":
        class_weight = None
        if strategy == "S1":
            class_weight = {0: 1.0, 1: float(pos_weight)}
        return LogisticRegression(
            penalty="l2",
            C=float(params["C"]),
            solver="liblinear",
            max_iter=3000,
            class_weight=class_weight,
            random_state=seed,
        )
    if model_name == "CatBoost":
        class_weights = None
        if strategy == "S1":
            class_weights = [1.0, float(pos_weight)]
        return CatBoostClassifier(
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            thread_count=-1,
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            iterations=int(params["iterations"]),
            l2_leaf_reg=3.0,
            class_weights=class_weights,
        )
    if model_name == "XGBoost":
        spw = 1.0
        if strategy == "S1":
            spw = float(pos_weight)
        return XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            n_estimators=int(params["n_estimators"]),
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            reg_lambda=XGB_REG_LAMBDA,
            tree_method="hist",
            scale_pos_weight=spw,
        )
    if model_name == "ExtraTrees":
        class_weight = None
        if strategy == "S1":
            class_weight = {0: 1.0, 1: float(pos_weight)}
        return ExtraTreesClassifier(
            random_state=seed,
            n_jobs=-1,
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=2,
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight=class_weight,
        )
    raise ValueError(f"Unknown model: {model_name}")


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
    else:
        return Pipeline([("prep", preprocessor), ("clf", estimator)])


def summarize_best_params_per_scheme(
    per_fold: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary_scheme_best: one row per (model,strategy), using best params (by mean AUPRC)
      scheme_best_params: mapping table with params retained (for traceability)
    """
    ok = per_fold[per_fold["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError(
            "All runs failed. Check per_fold_metrics_raw.csv -> error column."
        )
    agg = ok.groupby(["model", "strategy", "params"], as_index=False).agg(
        mean_AUPRC=("AUPRC", "mean"),
        std_AUPRC=("AUPRC", "std"),
        mean_AUROC=("AUROC", "mean"),
        std_AUROC=("AUROC", "std"),
        mean_Brier=("Brier", "mean"),
        std_Brier=("Brier", "std"),
        mean_LogLoss=("LogLoss", "mean"),
        std_LogLoss=("LogLoss", "std"),
        n_folds=("fold", "nunique"),
    )
    agg = agg.sort_values(
        by=["model", "strategy", "mean_AUPRC", "mean_AUROC", "mean_Brier"],
        ascending=[True, True, False, False, True],
    )
    best_rows = []
    for (m, s), sub in agg.groupby(["model", "strategy"], as_index=False):
        best_rows.append(sub.iloc[0])
    best_params = pd.DataFrame(best_rows).reset_index(drop=True)
    summary_scheme_best = best_params.copy()
    summary_scheme_best.insert(
        0,
        "scheme",
        summary_scheme_best["model"].astype(str)
        + "+"
        + summary_scheme_best["strategy"].astype(str),
    )
    summary_scheme_best = summary_scheme_best.sort_values(
        by=["mean_AUPRC", "mean_AUROC", "mean_Brier"], ascending=[False, False, True]
    ).reset_index(drop=True)
    summary_scheme_best.insert(0, "rank", np.arange(1, len(summary_scheme_best) + 1))
    summary_clean = summary_scheme_best.drop(columns=["params"])
    scheme_best_params = best_params[["model", "strategy", "params"]].copy()
    scheme_best_params.insert(
        0,
        "scheme",
        scheme_best_params["model"].astype(str)
        + "+"
        + scheme_best_params["strategy"].astype(str),
    )
    return (summary_clean, scheme_best_params)


def main():
    fix_seed(SEED)
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing DATA_CSV: {DATA_CSV.resolve()}")
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Missing SPLIT_CSV: {SPLIT_CSV.resolve()}")
    if not FEATURE_TYPES_CSV.exists():
        raise FileNotFoundError(
            f"Missing FEATURE_TYPES_CSV: {FEATURE_TYPES_CSV.resolve()}"
        )
    df = pd.read_csv(DATA_CSV)
    sp = pd.read_csv(SPLIT_CSV)
    for c in ["ID", "split", "fold_id", "Infection"]:
        if c not in sp.columns:
            raise ValueError(f"split_folds.csv missing column: {c}")
    sp_dev = sp[sp["split"].astype(str).str.upper() == "DEV"].copy()
    sp_dev = sp_dev[sp_dev["fold_id"].between(0, N_FOLDS - 1)].copy()
    if sp_dev.empty:
        raise ValueError("No DEV rows found in split file. Check split/fold_id.")
    folds = sorted(sp_dev["fold_id"].unique().tolist())
    print(f"[INFO] DEV folds found: {folds}")
    df = df.merge(
        sp[["ID", "Infection"]],
        on="ID",
        how="inner",
        suffixes=("", "_split"),
        validate="one_to_one",
    )
    y_col = "Infection_split" if "Infection_split" in df.columns else "Infection"
    df[y_col] = pd.to_numeric(df[y_col], errors="raise").astype(int)
    exclude_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    cont_cols, bin_cols, cat_cols = load_feature_types(
        FEATURE_TYPES_CSV, df.columns.tolist(), exclude=exclude_cols
    )
    cont_cols = [c for c in cont_cols if c in feature_cols]
    bin_cols = [c for c in bin_cols if c in feature_cols]
    cat_cols = [c for c in cat_cols if c in feature_cols]
    print(
        f"[INFO] Feature types: continuous={len(cont_cols)} | binary={len(bin_cols)} | categorical={len(cat_cols)}"
    )
    preprocessor = build_preprocessor(cont_cols, bin_cols, cat_cols)
    strategies = ["S0", "S1", "S2", "S3"]
    model_spaces = {
        "LR": lr_grid(),
        "CatBoost": cat_grid(),
        "XGBoost": xgb_grid(),
        "ExtraTrees": et_grid(),
    }
    n_jobs_total = 0
    for _, grid in model_spaces.items():
        n_jobs_total += len(grid) * len(strategies) * len(folds)
    print(f"[INFO] Total fits (approx): {n_jobs_total}")
    df_idx = df.set_index("ID", drop=False)
    records = []
    for model_name, grid in model_spaces.items():
        for strategy in strategies:
            for params in grid:
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
                    est = build_estimator(
                        model_name, params, strategy, pos_weight, SEED
                    )
                    pipe = build_pipeline(preprocessor, est, strategy, SEED)
                    try:
                        pipe.fit(Xtr, ytr)
                        p = predict_proba_pos(pipe, Xva)
                        m = safe_metrics(yva, p)
                        status = "ok"
                        err = ""
                    except Exception as e:
                        m = {
                            "AUROC": np.nan,
                            "AUPRC": np.nan,
                            "Brier": np.nan,
                            "LogLoss": np.nan,
                        }
                        status = "fail"
                        err = repr(e)[:300]
                    records.append(
                        {
                            "model": model_name,
                            "strategy": strategy,
                            "params": json.dumps(params, ensure_ascii=False),
                            "fold": int(f),
                            "n_tr": int(len(Xtr)),
                            "pos_tr": int(n_pos),
                            "neg_tr": int(n_neg),
                            "pos_weight": float(pos_weight),
                            "status": status,
                            "error": err,
                            **m,
                        }
                    )
                print(f"[DONE] {model_name} {strategy} params={params}")
    per_fold = pd.DataFrame(records)
    per_fold_path = OUT_DIR / "per_fold_metrics_raw.csv"
    per_fold.to_csv(per_fold_path, index=False, encoding="utf-8-sig")
    print("[OK] Saved:", per_fold_path)
    summary_clean, scheme_best_params = summarize_best_params_per_scheme(per_fold)
    summary_path = OUT_DIR / "summary_bestparams_per_scheme.csv"
    summary_clean.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("[OK] Saved:", summary_path)
    scheme_params_path = OUT_DIR / "scheme_to_best_params.csv"
    scheme_best_params.to_csv(scheme_params_path, index=False, encoding="utf-8-sig")
    print("[OK] Saved:", scheme_params_path)
    best_row = scheme_best_params.merge(
        summary_clean[
            ["scheme", "rank", "mean_AUPRC", "mean_AUROC", "mean_Brier", "mean_LogLoss"]
        ],
        on="scheme",
        how="left",
    )
    best_row = best_row.sort_values(
        by=["mean_AUPRC", "mean_AUROC", "mean_Brier"], ascending=[False, False, True]
    ).iloc[0]
    best_cfg = {
        "scheme": str(best_row["scheme"]),
        "model": str(best_row["model"]),
        "strategy": str(best_row["strategy"]),
        "params": json.loads(best_row["params"]),
        "rank_among_schemes": int(best_row["rank"]),
        "mean_AUPRC": float(best_row["mean_AUPRC"]),
        "mean_AUROC": float(best_row["mean_AUROC"]),
        "mean_Brier": float(best_row["mean_Brier"]),
        "mean_LogLoss": float(best_row["mean_LogLoss"]),
        "n_folds": int(N_FOLDS),
        "seed": int(SEED),
        "notes": "Best scheme selected by mean AUPRC; hyperparams picked within scheme by the same criterion.",
    }
    best_path = OUT_DIR / "best_config.json"
    best_path.write_text(
        json.dumps(best_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[OK] Best config saved:", best_path)
    print("\n==== BEST SCHEME (by mean AUPRC) ====")
    print(json.dumps(best_cfg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
