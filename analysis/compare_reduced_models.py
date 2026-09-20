"""Compare model configurations using the selected reduced predictor set.

The configured frozen-parameter analysis reuses full-model selections.
All CatBoost imbalance variants use the CatBoost S0 full-model parameters;
other model families retain their corresponding full-model parameters.
These comparisons are exploratory and do not establish model equivalence."""

import os, json, random, warnings
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
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
SEED = 42
N_FOLDS = 5
N_THRESH = 201
SMOTE_K_NEIGHBORS = 3
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_REG_LAMBDA = 1.0
FREEZE_HYPERPARAMS = True
DATA_CSV = Path("data/clinical_data.csv")
SPLIT_CSV = Path("data/split_folds.csv")
FEATURE_TYPES_CSV = Path("feature_schema.csv")
EXCLUDE_COLS = ["ID", "Infection", "Infection_split"]
TARGET_SCHEME = "CatBoost+S0"
SCHEME_TYPE = "scheme2"
TARGET_TOPK = 4
CONTRIBUTION_CSV = Path(
    f"outputs/feature_selection/{TARGET_SCHEME}/contribution_{SCHEME_TYPE}.csv"
)
OUT_DIR = Path("outputs/reduced_models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FULL_FEATURE_SUMMARY = Path("outputs/full_models/multi_scheme_summary_dev_indep.csv")
SCHEME_PARAMS_CSV = Path("outputs/model_selection/scheme_to_best_params.csv")
NEW_SCHEMES = [
    ("CatBoost", "S0"),
    ("CatBoost", "S1"),
    ("CatBoost", "S2"),
    ("CatBoost", "S3"),
    ("LR", "S0"),
    ("LR", "S1"),
    ("LR", "S2"),
    ("LR", "S3"),
    ("XGBoost", "S0"),
    ("XGBoost", "S1"),
    ("XGBoost", "S2"),
    ("XGBoost", "S3"),
    ("ExtraTrees", "S0"),
    ("ExtraTrees", "S1"),
    ("ExtraTrees", "S2"),
    ("ExtraTrees", "S3"),
]


def fix_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


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


def build_estimator(
    model_name: str, params: Dict[str, Any], strategy: str, pos_weight: float, seed: int
):
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
        return ImbPipeline(
            [("prep", preprocessor), ("smote", smote), ("clf", estimator)]
        )
    else:
        return Pipeline([("prep", preprocessor), ("clf", estimator)])


def get_model_grid(model_name: str) -> List[Dict[str, Any]]:
    if model_name == "LR":
        return lr_grid()
    elif model_name == "CatBoost":
        return cat_grid()
    elif model_name == "XGBoost":
        return xgb_grid()
    elif model_name == "ExtraTrees":
        return et_grid()
    raise ValueError(f"Unknown model: {model_name}")


def load_full_feature_best_params() -> Dict[str, Dict[str, Any]]:
    """Load full-model parameters, sharing CatBoost S0 parameters across CatBoost imbalance variants."""
    df = pd.read_csv(SCHEME_PARAMS_CSV)
    raw = {}
    for _, row in df.iterrows():
        scheme = str(row["scheme"])
        try:
            params = json.loads(row["params"])
        except Exception:
            continue
        raw[scheme] = params
    cb_s0_scheme = f"CatBoost+S0"
    if cb_s0_scheme not in raw:
        raise KeyError(
            f"In {SCHEME_PARAMS_CSV} does not contain {cb_s0_scheme}; CatBoost baseline parameters are unavailable"
        )
    cb_s0_params = raw[cb_s0_scheme]
    result = {}
    for scheme, params in raw.items():
        if scheme.startswith("CatBoost+"):
            result[scheme] = dict(cb_s0_params)
        else:
            result[scheme] = dict(params)
    print(f"\n[INFO] Shared frozen CatBoost parameters: {cb_s0_params}")
    for s, p in result.items():
        print(f"  {s}: {p}")
    return result


def predict_proba_pos(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    z = model.decision_function(X)
    return 1 / (1 + np.exp(-z))


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


def confusion_metrics_from_preds(
    y_true: np.ndarray, y_hat: np.ndarray
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
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
    return {
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "Sensitivity": float(sens) if sens == sens else np.nan,
        "Specificity": float(spec) if spec == spec else np.nan,
        "PPV": float(ppv) if ppv == ppv else np.nan,
        "NPV": float(npv) if npv == npv else np.nan,
        "F1": float(f1) if f1 == f1 else np.nan,
        "Accuracy": float(acc) if acc == acc else np.nan,
        "BalancedAcc": float(bal_acc) if bal_acc == bal_acc else np.nan,
        "YoudenJ": float(youden) if youden == youden else np.nan,
    }


def find_best_threshold_youden(
    y_true: np.ndarray, p: np.ndarray, n_thresh: int = 201
) -> Tuple[float, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    thresholds = np.linspace(0.0, 1.0, n_thresh)
    rows = []
    for t in thresholds:
        y_hat = (p >= t).astype(int)
        cm = confusion_metrics_from_preds(y_true, y_hat)
        rows.append({"threshold": float(t), **cm})
    scan_df = pd.DataFrame(rows)
    scan_df = scan_df.sort_values(
        by=["YoudenJ", "Sensitivity", "PPV", "threshold"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best_thr = float(scan_df.loc[0, "threshold"])
    return (best_thr, scan_df)


def detect_indep_split(sp: pd.DataFrame) -> str:
    splits = sp["split"].astype(str).str.upper().unique().tolist()
    for cand in ["VAL", "INDEP_TEST", "TEST"]:
        if cand in splits:
            return cand
    raise ValueError(f"Cannot find independent split among {splits}")


def safe_float(v) -> float:
    """Safely convert value to float, returning NaN on failure."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def summarize_best_params_per_scheme(
    per_fold: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ok = per_fold[per_fold["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("All runs failed.")
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


def evaluate_one_scheme_on_topk(
    model_name: str,
    strategy: str,
    features: List[str],
    preprocessor: ColumnTransformer,
    df_idx: pd.DataFrame,
    sp_dev: pd.DataFrame,
    sp_indep: pd.DataFrame,
    indep_split: str,
    y_col: str,
    folds: List[int],
    out_dir: Path,
    full_best_params: Dict[str, Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline for ONE scheme on the given feature set:
      1) Get best params (frozen from full-feature OR grid search on TopK)
      2) DEV OOF prediction -> Youden threshold
      3) Train on full DEV -> INDEP evaluation
      4) Save outputs to out_dir
    Returns summary metrics dict.
    """
    scheme_name = f"{model_name}+{strategy}"
    print(f"\n{'=' * 60}")
    print(f"[{scheme_name}] Evaluating {len(features)} predictors...")
    print(f"{'=' * 60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if FREEZE_HYPERPARAMS:
        if full_best_params is None:
            raise ValueError("FREEZE_HYPERPARAMS=True requires full_best_params")
        if scheme_name not in full_best_params:
            raise KeyError(
                f"Missing configuration in scheme_to_best_params.csv: {scheme_name}"
            )
        best_params = full_best_params[scheme_name]
        print(
            f"[{scheme_name}] Frozen hyperparameters: {best_params} (source: {SCHEME_PARAMS_CSV})"
        )
        records = [
            {
                "model": model_name,
                "strategy": strategy,
                "params": json.dumps(best_params, ensure_ascii=False),
                "fold": -1,
                "n_tr": -1,
                "pos_tr": -1,
                "neg_tr": -1,
                "pos_weight": -1.0,
                "status": "frozen",
                "error": "",
                "AUROC": -1.0,
                "AUPRC": -1.0,
                "Brier": -1.0,
                "LogLoss": -1.0,
            }
        ]
        per_fold = pd.DataFrame(records)
        per_fold.to_csv(
            out_dir / "per_fold_metrics_raw.csv", index=False, encoding="utf-8-sig"
        )
    else:
        print(
            f"[{scheme_name}] Step 1/3: Five-fold grid search on selected predictors..."
        )
        grid = get_model_grid(model_name)
        records = []
        for params in grid:
            for f in folds:
                tr_ids = sp_dev.loc[sp_dev["fold_id"] != f, "ID"].tolist()
                va_ids = sp_dev.loc[sp_dev["fold_id"] == f, "ID"].tolist()
                tr = df_idx.loc[tr_ids]
                va = df_idx.loc[va_ids]
                Xtr = tr[features]
                ytr = tr[y_col].values.astype(int)
                Xva = va[features]
                yva = va[y_col].values.astype(int)
                n_pos = int(ytr.sum())
                n_neg = int((ytr == 0).sum())
                pos_weight = n_neg / max(1, n_pos)
                est = build_estimator(model_name, params, strategy, pos_weight, SEED)
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
        per_fold = pd.DataFrame(records)
        per_fold.to_csv(
            out_dir / "per_fold_metrics_raw.csv", index=False, encoding="utf-8-sig"
        )
        _, scheme_best_params = summarize_best_params_per_scheme(per_fold)
        best_row = scheme_best_params.iloc[0]
        best_params = json.loads(best_row["params"])
        print(f"[{scheme_name}] Selected parameters: {best_params}")
    print(f"[{scheme_name}] Step 2/3: DEV OOF predictions and Youden threshold...")
    oof_rows = []
    for f in folds:
        tr_ids = sp_dev.loc[sp_dev["fold_id"] != f, "ID"].tolist()
        va_ids = sp_dev.loc[sp_dev["fold_id"] == f, "ID"].tolist()
        tr = df_idx.loc[tr_ids]
        va = df_idx.loc[va_ids]
        Xtr = tr[features]
        ytr = tr[y_col].values.astype(int)
        Xva = va[features]
        yva = va[y_col].values.astype(int)
        n_pos = int(ytr.sum())
        n_neg = int((ytr == 0).sum())
        pos_weight = n_neg / max(1, n_pos)
        est = build_estimator(model_name, best_params, strategy, pos_weight, SEED)
        pipe = build_pipeline(preprocessor, est, strategy, SEED)
        pipe.fit(Xtr, ytr)
        pva = predict_proba_pos(pipe, Xva)
        for _id, yt, pr, fid in zip(
            va["ID"].values.tolist(), yva.tolist(), pva.tolist(), [int(f)] * len(yva)
        ):
            oof_rows.append(
                {"ID": _id, "y_true": int(yt), "oof_prob": float(pr), "fold_id": fid}
            )
    dev_oof = (
        pd.DataFrame(oof_rows).sort_values(["fold_id", "ID"]).reset_index(drop=True)
    )
    dev_oof.to_csv(
        out_dir / "dev_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    dev_oof_basic = safe_metrics(dev_oof["y_true"].values, dev_oof["oof_prob"].values)
    best_thr, scan_df = find_best_threshold_youden(
        dev_oof["y_true"].values, dev_oof["oof_prob"].values, n_thresh=N_THRESH
    )
    scan_df.to_csv(
        out_dir / "dev_oof_metrics_threshold_scan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dev_oof_hat = (dev_oof["oof_prob"].values >= best_thr).astype(int)
    dev_oof_cm = confusion_metrics_from_preds(dev_oof["y_true"].values, dev_oof_hat)
    print(
        f"[{scheme_name}] DEV OOF: AUPRC={dev_oof_basic['AUPRC']:.4f}, AUROC={dev_oof_basic['AUROC']:.4f}, Thr={best_thr:.4f}"
    )
    print(f"[{scheme_name}] Step 3/3: Refit on DEV and evaluate the held-out split...")
    dev_all = df_idx.loc[sp_dev["ID"].tolist()]
    indep_all = df_idx.loc[sp_indep["ID"].tolist()]
    Xtr_full = dev_all[features]
    ytr_full = dev_all[y_col].values.astype(int)
    Xte = indep_all[features]
    yte = indep_all[y_col].values.astype(int)
    n_pos_full = int(ytr_full.sum())
    n_neg_full = int((ytr_full == 0).sum())
    pos_weight_full = n_neg_full / max(1, n_pos_full)
    est_final = build_estimator(
        model_name, best_params, strategy, pos_weight_full, SEED
    )
    pipe_final = build_pipeline(preprocessor, est_final, strategy, SEED)
    pipe_final.fit(Xtr_full, ytr_full)
    p_te = predict_proba_pos(pipe_final, Xte)
    indep_basic = safe_metrics(yte, p_te)
    y_hat_te = (p_te >= best_thr).astype(int)
    indep_cm = confusion_metrics_from_preds(yte, y_hat_te)
    indep_pred = (
        pd.DataFrame(
            {
                "ID": indep_all["ID"].values,
                "y_true": yte.astype(int),
                "pred_prob": p_te.astype(float),
                "pred_label": y_hat_te.astype(int),
                "threshold_used": float(best_thr),
            }
        )
        .sort_values("ID")
        .reset_index(drop=True)
    )
    indep_pred.to_csv(
        out_dir / "indep_predictions.csv", index=False, encoding="utf-8-sig"
    )
    cm_tbl = pd.DataFrame(
        [
            {
                "TN": indep_cm["TN"],
                "FP": indep_cm["FP"],
                "FN": indep_cm["FN"],
                "TP": indep_cm["TP"],
            }
        ]
    )
    cm_tbl.to_csv(
        out_dir / "indep_confusion_matrix.csv", index=False, encoding="utf-8-sig"
    )
    report_rows = []
    for ds_name, y_ds, p_ds, cm_ds in [
        ("DEV_OOF", dev_oof["y_true"].values, dev_oof["oof_prob"].values, dev_oof_cm),
        (indep_split, yte, p_te, indep_cm),
    ]:
        basic = safe_metrics(y_ds, p_ds) if ds_name == "DEV_OOF" else indep_basic
        report_rows.append(
            {
                "Scheme": scheme_name,
                "Dataset": ds_name,
                "n_total": int(len(y_ds)),
                "n_pos": int(y_ds.sum()),
                "n_neg": int((y_ds == 0).sum()),
                "AUROC": float(basic["AUROC"]),
                "AUPRC": float(basic["AUPRC"]),
                "Brier": float(basic["Brier"]),
                "LogLoss": float(basic["LogLoss"]),
                "Threshold": float(best_thr),
                "Sensitivity": float(cm_ds["Sensitivity"]),
                "Specificity": float(cm_ds["Specificity"]),
                "PPV": float(cm_ds["PPV"]),
                "NPV": float(cm_ds["NPV"]),
                "F1": float(cm_ds["F1"]),
                "Accuracy": float(cm_ds["Accuracy"]),
                "BalancedAcc": float(cm_ds["BalancedAcc"]),
                "YoudenJ": float(cm_ds["YoudenJ"]),
            }
        )
    report = pd.DataFrame(report_rows)
    metric_cols = [
        "AUROC",
        "AUPRC",
        "Brier",
        "LogLoss",
        "Threshold",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "F1",
        "Accuracy",
        "BalancedAcc",
        "YoudenJ",
    ]
    for c in metric_cols:
        if c in report.columns:
            report[c] = report[c].astype(float).round(4)
    report.to_csv(
        out_dir / "report_metrics_table.csv", index=False, encoding="utf-8-sig"
    )
    roc_fpr, roc_tpr, _ = roc_curve(
        dev_oof["y_true"].values, dev_oof["oof_prob"].values
    )
    pr_prec, pr_rec, _ = precision_recall_curve(
        dev_oof["y_true"].values, dev_oof["oof_prob"].values
    )
    curves = pd.DataFrame(
        {
            "dev_roc_fpr": pd.Series(roc_fpr),
            "dev_roc_tpr": pd.Series(roc_tpr),
            "dev_pr_precision": pd.Series(pr_prec),
            "dev_pr_recall": pd.Series(pr_rec),
        }
    )
    curves.to_csv(
        out_dir / "report_curves_roc_pr.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"[{scheme_name}] Complete. Held-out average precision={indep_basic['AUPRC']:.4f}, AUROC={indep_basic['AUROC']:.4f}"
    )
    return {
        "Scheme": scheme_name,
        "Model": model_name,
        "Strategy": strategy,
        "DEV_AUROC": float(dev_oof_basic["AUROC"]),
        "DEV_AUPRC": float(dev_oof_basic["AUPRC"]),
        "DEV_Brier": float(dev_oof_basic["Brier"]),
        "DEV_LogLoss": float(dev_oof_basic["LogLoss"]),
        "DEV_Threshold": float(best_thr),
        "DEV_Sensitivity": float(dev_oof_cm["Sensitivity"]),
        "DEV_Specificity": float(dev_oof_cm["Specificity"]),
        "DEV_PPV": float(dev_oof_cm["PPV"]),
        "DEV_NPV": float(dev_oof_cm["NPV"]),
        "DEV_F1": float(dev_oof_cm["F1"]),
        "DEV_YoudenJ": float(dev_oof_cm["YoudenJ"]),
        "INDEP_Split": indep_split,
        "INDEP_AUROC": float(indep_basic["AUROC"]),
        "INDEP_AUPRC": float(indep_basic["AUPRC"]),
        "INDEP_Brier": float(indep_basic["Brier"]),
        "INDEP_LogLoss": float(indep_basic["LogLoss"]),
        "INDEP_Threshold": float(best_thr),
        "INDEP_Sensitivity": float(indep_cm["Sensitivity"]),
        "INDEP_Specificity": float(indep_cm["Specificity"]),
        "INDEP_PPV": float(indep_cm["PPV"]),
        "INDEP_NPV": float(indep_cm["NPV"]),
        "INDEP_F1": float(indep_cm["F1"]),
        "INDEP_YoudenJ": float(indep_cm["YoudenJ"]),
    }


def main():
    fix_seed(SEED)
    print("=" * 70)
    print("  Model comparison on MCFS-selected predictors")
    print(
        f"  TopK = {TARGET_TOPK}, configuration = {TARGET_SCHEME}, scheme = {SCHEME_TYPE}"
    )
    if FREEZE_HYPERPARAMS:
        print(f"  Using frozen full-model hyperparameters")
    else:
        print(f"  Retuning hyperparameters on the selected predictors")
    print("=" * 70)
    full_best_params = None
    if FREEZE_HYPERPARAMS:
        print(f"\n[0/5] Loading selected full-model parameters: {SCHEME_PARAMS_CSV}")
        if not SCHEME_PARAMS_CSV.exists():
            raise FileNotFoundError(
                f"Missing: {SCHEME_PARAMS_CSV} (run select_models.py first)"
            )
        full_best_params = load_full_feature_best_params()
    print(f"\n[1/5] Loading consensus file: {CONTRIBUTION_CSV}")
    if not CONTRIBUTION_CSV.exists():
        raise FileNotFoundError(f"Missing: {CONTRIBUTION_CSV}")
    contrib_df = pd.read_csv(CONTRIBUTION_CSV)
    contrib_df["Rank"] = pd.to_numeric(contrib_df["Rank"], errors="coerce")
    contrib_df = contrib_df.sort_values("Rank")
    features = contrib_df["feature"].head(TARGET_TOPK).tolist()
    print(f"  Top{TARGET_TOPK} predictors: {features}")
    print(f"\n[2/5] Loading data...")
    df = pd.read_csv(DATA_CSV)
    sp = pd.read_csv(SPLIT_CSV)
    indep_split = detect_indep_split(sp)
    df = df.merge(
        sp[["ID", "split", "Infection"]],
        on="ID",
        how="inner",
        suffixes=("", "_split"),
        validate="one_to_one",
    )
    y_col = "Infection_split" if "Infection_split" in df.columns else "Infection"
    df[y_col] = pd.to_numeric(df[y_col], errors="raise").astype(int)
    sp_dev = sp[sp["split"].astype(str).str.upper() == "DEV"].copy()
    sp_dev = sp_dev[sp_dev["fold_id"].between(0, N_FOLDS - 1)].copy()
    if sp_dev.empty:
        raise ValueError("No DEV rows found.")
    folds = sorted(sp_dev["fold_id"].unique().tolist())
    print(f"  DEV folds: {folds} | DEV samples: {len(sp_dev)}")
    sp_indep = sp[sp["split"].astype(str).str.upper() == indep_split].copy()
    if sp_indep.empty:
        raise ValueError(f"No rows for independent split: {indep_split}")
    print(f"  INDEP split: {indep_split} ({len(sp_indep)} samples)")
    exclude_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    cont_cols, bin_cols, cat_cols = load_feature_types(
        FEATURE_TYPES_CSV, df.columns.tolist(), exclude=exclude_cols
    )
    cont_cols = [c for c in cont_cols if c in features]
    bin_cols = [c for c in bin_cols if c in features]
    cat_cols = [c for c in cat_cols if c in features]
    print(
        f"\n[3/5] Predictor types (top {TARGET_TOPK}): continuous={len(cont_cols)} | binary={len(bin_cols)} | categorical={len(cat_cols)}"
    )
    preprocessor = build_preprocessor(cont_cols, bin_cols, cat_cols)
    df_idx = df.set_index("ID", drop=False)
    print(f"\n[4/5] Running {len(NEW_SCHEMES)} configurations...")
    all_summary_rows = []
    for model_name, strategy in NEW_SCHEMES:
        scheme_name = f"{model_name}+{strategy}"
        scheme_out_dir = OUT_DIR / scheme_name
        result = evaluate_one_scheme_on_topk(
            model_name=model_name,
            strategy=strategy,
            features=features,
            preprocessor=preprocessor,
            df_idx=df_idx,
            sp_dev=sp_dev,
            sp_indep=sp_indep,
            indep_split=indep_split,
            y_col=y_col,
            folds=folds,
            out_dir=scheme_out_dir,
            full_best_params=full_best_params,
        )
        all_summary_rows.append(result)
    print(f"\n[5/5] Generating comparison table...")
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df = summary_df.sort_values(
        by=["INDEP_AUPRC", "INDEP_AUROC"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))
    summary_df.insert(2, "n_features", TARGET_TOPK)
    comp_path = OUT_DIR / "multi_scheme_comparison.csv"
    summary_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
    print(f"  [SAVE] {comp_path}")
    print("\n" + "-" * 70)
    print(
        f"  Reduced predictor set (top {TARGET_TOPK}) model comparison (sorted by held-out average precision)"
    )
    print("-" * 70)
    print_cols = [
        "rank",
        "Scheme",
        "n_features",
        "DEV_AUPRC",
        "INDEP_AUPRC",
        "INDEP_AUROC",
    ]
    available_cols = [c for c in print_cols if c in summary_df.columns]
    print(summary_df[available_cols].round(4).to_string(index=False))
    print(f"\n  Comparing against full-model results...")
    if FULL_FEATURE_SUMMARY.exists():
        full_df = pd.read_csv(FULL_FEATURE_SUMMARY)
        comparison_rows = []
        for _, row in summary_df.iterrows():
            scheme = row["Scheme"]
            full_match = full_df[full_df["Scheme"] == scheme]
            if not full_match.empty:
                fr = full_match.iloc[0]
                delta = (
                    float(row["INDEP_AUPRC"])
                    - safe_float(fr.get("INDEP_AUPRC", float("nan")))
                    if row["INDEP_AUPRC"] == row["INDEP_AUPRC"]
                    and str(fr.get("INDEP_AUPRC", "nan")) != "nan"
                    else float("nan")
                )
                comparison_rows.append(
                    {
                        "Scheme": scheme,
                        "Model": str(row.get("Model", "")),
                        "Strategy": str(row.get("Strategy", "")),
                        "n_features_topk": int(TARGET_TOPK),
                        "n_features_full": 63,
                        "TopK_DEV_AUPRC": safe_float(
                            row.get("DEV_AUPRC", float("nan"))
                        ),
                        "TopK_INDEP_AUPRC": safe_float(
                            row.get("INDEP_AUPRC", float("nan"))
                        ),
                        "TopK_INDEP_AUROC": safe_float(
                            row.get("INDEP_AUROC", float("nan"))
                        ),
                        "Full_DEV_AUPRC": safe_float(fr.get("DEV_AUPRC", float("nan"))),
                        "Full_INDEP_AUPRC": safe_float(
                            fr.get("INDEP_AUPRC", float("nan"))
                        ),
                        "Full_INDEP_AUROC": safe_float(
                            fr.get("INDEP_AUROC", float("nan"))
                        ),
                        "AUPRC_Delta": delta,
                    }
                )
            else:
                comparison_rows.append(
                    {
                        "Scheme": scheme,
                        "Model": str(row.get("Model", "")),
                        "Strategy": str(row.get("Strategy", "")),
                        "n_features_topk": int(TARGET_TOPK),
                        "n_features_full": 63,
                        "TopK_DEV_AUPRC": safe_float(
                            row.get("DEV_AUPRC", float("nan"))
                        ),
                        "TopK_INDEP_AUPRC": safe_float(
                            row.get("INDEP_AUPRC", float("nan"))
                        ),
                        "TopK_INDEP_AUROC": safe_float(
                            row.get("INDEP_AUROC", float("nan"))
                        ),
                        "Full_DEV_AUPRC": float("nan"),
                        "Full_INDEP_AUPRC": float("nan"),
                        "Full_INDEP_AUROC": float("nan"),
                        "AUPRC_Delta": float("nan"),
                    }
                )
        comp_vs_full = pd.DataFrame(comparison_rows)
        comp_vs_full = comp_vs_full.sort_values(
            by=["TopK_INDEP_AUPRC", "TopK_INDEP_AUROC"], ascending=[False, False]
        ).reset_index(drop=True)
        vs_path = OUT_DIR / "summary_comparison_vs_full.csv"
        comp_vs_full.to_csv(vs_path, index=False, encoding="utf-8-sig")
        print(f"  [SAVE] {vs_path}")
        print("\n" + "-" * 70)
        print(f"  Top{TARGET_TOPK} versus full 64-predictor models")
        print("-" * 70)
        vs_cols = [
            "Scheme",
            "TopK_INDEP_AUPRC",
            "Full_INDEP_AUPRC",
            "AUPRC_Delta",
            "TopK_INDEP_AUROC",
            "Full_INDEP_AUROC",
        ]
        print(comp_vs_full[vs_cols].round(4).to_string(index=False))
    else:
        print(
            f"  [WARN] Full-model results not found: {FULL_FEATURE_SUMMARY}; skipping comparison."
        )
    print(f"\n{'=' * 70}")
    print(f"  Analysis complete. Output directory: {OUT_DIR.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
