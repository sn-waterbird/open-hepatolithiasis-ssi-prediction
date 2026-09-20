"""Evaluate full-predictor models using saved model-selection configurations.

Generate DEV out-of-fold probabilities, select Youden operating thresholds,
refit on DEV, and evaluate the fixed held-out split. Save predictions, metrics,
and curve coordinates under outputs/full_models/."""

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
from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

SEED = 42
N_FOLDS = 5
DATA_CSV = Path("data/clinical_data.csv")
SPLIT_CSV = Path("data/split_folds.csv")
FEATURE_TYPES_CSV = Path("feature_schema.csv")
SCHEME_PARAMS_CSV = Path("outputs/model_selection/scheme_to_best_params.csv")
TARGET_SCHEMES = [
    "XGBoost+S0",
    "XGBoost+S1",
    "XGBoost+S2",
    "XGBoost+S3",
    "ExtraTrees+S0",
    "ExtraTrees+S1",
    "ExtraTrees+S2",
    "ExtraTrees+S3",
    "CatBoost+S0",
    "CatBoost+S1",
    "CatBoost+S2",
    "CatBoost+S3",
    "LR+S0",
    "LR+S1",
    "LR+S2",
    "LR+S3",
]
OUT_ROOT = Path("outputs/full_models")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
EXCLUDE_COLS = ["ID", "Infection", "Infection_split"]
SMOTE_K_NEIGHBORS = 3
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_REG_LAMBDA = 1.0
N_THRESH = 201


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
    fpr = fp / (fp + tn) if fp + tn > 0 else np.nan
    fnr = fn / (fn + tp) if fn + tp > 0 else np.nan
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
        "FPR": float(fpr) if fpr == fpr else np.nan,
        "FNR": float(fnr) if fnr == fnr else np.nan,
    }


def find_best_threshold_youden(
    y_true: np.ndarray, p: np.ndarray, n_thresh: int = 201
) -> Tuple[float, pd.DataFrame]:
    """
    Scan thresholds on [0,1], choose threshold maximizing Youden's J = Sens + Spec - 1.
    Tie-breakers:
      1) higher YoudenJ
      2) higher Sensitivity
      3) higher PPV
      4) lower threshold
    """
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


def parse_params_string(params_str: str) -> Dict[str, Any]:
    """
    Robust parse for params column in csv.
    Supports JSON-like or python-dict-like strings.
    """
    if pd.isna(params_str):
        raise ValueError("params is NaN")
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
    if not csv_path.exists():
        raise FileNotFoundError(
            f"scheme_to_best_params.csv not found: {csv_path.resolve()}"
        )
    df = pd.read_csv(csv_path)
    required_cols = ["scheme", "model", "strategy", "params"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"scheme_to_best_params.csv missing columns: {missing}")
    hit = df[df["scheme"].astype(str) == str(target_scheme)].copy()
    if hit.empty:
        raise ValueError(
            f"TARGET_SCHEME={target_scheme} not found in {csv_path}.\nAvailable schemes: {df['scheme'].astype(str).tolist()}"
        )
    if len(hit) > 1:
        raise ValueError(
            f"TARGET_SCHEME={target_scheme} matched multiple rows, please check csv."
        )
    row = hit.iloc[0]
    cfg = {
        "scheme": str(row["scheme"]),
        "model": str(row["model"]),
        "strategy": str(row["strategy"]),
        "params": parse_params_string(row["params"]),
    }
    return cfg


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
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
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


def detect_indep_split(sp: pd.DataFrame) -> str:
    """
    Prefer VAL if present.
    Otherwise choose INDEP_TEST, else TEST.
    """
    splits = sp["split"].astype(str).str.upper().unique().tolist()
    for cand in ["VAL", "INDEP_TEST", "TEST"]:
        if cand in splits:
            return cand
    raise ValueError(
        f"Cannot find independent split among {splits}. Expected one of VAL/INDEP_TEST/TEST."
    )


def run_one_scheme(
    target_scheme: str,
    data_csv: Path,
    split_csv: Path,
    feature_types_csv: Path,
    scheme_params_csv: Path,
    out_root: Path,
) -> Dict[str, Any]:
    cfg = load_target_scheme_config(scheme_params_csv, target_scheme)
    scheme_name = cfg["scheme"]
    model_name = cfg["model"]
    strategy = cfg["strategy"]
    params = cfg["params"]
    out_dir = out_root / scheme_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_cfg_path = out_dir / "selected_scheme_config.json"
    selected_cfg_path.write_text(
        json.dumps(
            {
                "scheme": scheme_name,
                "model": model_name,
                "strategy": strategy,
                "params": params,
                "seed": SEED,
                "n_folds": N_FOLDS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[INFO] Selected scheme config:")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    print(f"[INFO] Output directory: {out_dir.resolve()}")
    df = pd.read_csv(data_csv)
    sp = pd.read_csv(split_csv)
    for c in ["ID", "split", "fold_id", "Infection"]:
        if c not in sp.columns:
            raise ValueError(f"split_folds.csv missing column: {c}")
    indep_split = detect_indep_split(sp)
    print(f"[INFO] Independent split detected: {indep_split}")
    sp_dev = sp[sp["split"].astype(str).str.upper() == "DEV"].copy()
    sp_dev = sp_dev[sp_dev["fold_id"].between(0, N_FOLDS - 1)].copy()
    if sp_dev.empty:
        raise ValueError("No DEV rows found in split file. Check split/fold_id.")
    folds = sorted(sp_dev["fold_id"].unique().tolist())
    print(f"[INFO] DEV folds found: {folds}")
    sp_indep = sp[sp["split"].astype(str).str.upper() == indep_split].copy()
    if sp_indep.empty:
        raise ValueError(f"No rows found for independent split: {indep_split}")
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
        feature_types_csv, df.columns.tolist(), exclude=exclude_cols
    )
    cont_cols = [c for c in cont_cols if c in feature_cols]
    bin_cols = [c for c in bin_cols if c in feature_cols]
    cat_cols = [c for c in cat_cols if c in feature_cols]
    print(
        f"[INFO] Feature types: continuous={len(cont_cols)} | binary={len(bin_cols)} | categorical={len(cat_cols)}"
    )
    preprocessor = build_preprocessor(cont_cols, bin_cols, cat_cols)
    df_idx = df.set_index("ID", drop=False)
    print("\n" + "=" * 60)
    print(f"[STEP1] DEV OOF predictions | {scheme_name}")
    print("=" * 60)
    oof_rows = []
    fold_metrics_rows = []
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
        est = build_estimator(model_name, params, strategy, pos_weight, SEED)
        pipe = build_pipeline(preprocessor, est, strategy, SEED)
        pipe.fit(Xtr, ytr)
        pva = predict_proba_pos(pipe, Xva)
        for _id, yt, pr in zip(va["ID"].values.tolist(), yva.tolist(), pva.tolist()):
            oof_rows.append(
                {"ID": _id, "y_true": int(yt), "oof_prob": float(pr), "fold_id": int(f)}
            )
        bm = safe_basic_metrics(yva, pva)
        fold_metrics_rows.append(
            {
                "fold_id": int(f),
                "n_val": int(len(yva)),
                "pos_val": int(yva.sum()),
                "neg_val": int((yva == 0).sum()),
                **bm,
            }
        )
        print(
            f"[OK] fold {f} done | val_AUPRC={bm['AUPRC']:.4f} val_AUROC={bm['AUROC']:.4f}"
        )
    dev_oof = (
        pd.DataFrame(oof_rows).sort_values(["fold_id", "ID"]).reset_index(drop=True)
    )
    dev_oof_path = out_dir / "dev_oof_predictions.csv"
    dev_oof.to_csv(dev_oof_path, index=False, encoding="utf-8-sig")
    print("[SAVE] DEV OOF:", dev_oof_path)
    fold_metrics = pd.DataFrame(fold_metrics_rows).sort_values("fold_id")
    fold_metrics_path = out_dir / "dev_oof_fold_metrics.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    print("[SAVE] DEV fold metrics:", fold_metrics_path)
    dev_oof_basic = safe_basic_metrics(
        dev_oof["y_true"].values, dev_oof["oof_prob"].values
    )
    print("\n" + "=" * 60)
    print("[STEP2] Threshold selection on DEV OOF by Youden's J")
    print("=" * 60)
    best_thr, scan_df = find_best_threshold_youden(
        dev_oof["y_true"].values, dev_oof["oof_prob"].values, n_thresh=N_THRESH
    )
    scan_path = out_dir / "dev_oof_metrics_threshold_scan.csv"
    scan_df.to_csv(scan_path, index=False, encoding="utf-8-sig")
    print("[SAVE] Threshold scan:", scan_path)
    dev_oof_hat = (dev_oof["oof_prob"].values >= best_thr).astype(int)
    dev_oof_cm = confusion_metrics_from_preds(dev_oof["y_true"].values, dev_oof_hat)
    thr_info = {
        "scheme": scheme_name,
        "threshold_rule": "YoudenJ (maximize Sens+Spec-1) on DEV OOF",
        "best_threshold": float(best_thr),
        "dev_oof_threshold_free": {k: float(v) for k, v in dev_oof_basic.items()},
        "dev_oof_at_threshold": {
            k: float(v) if v == v else None for k, v in dev_oof_cm.items()
        },
        "model": model_name,
        "strategy": strategy,
        "params": params,
        "seed": int(SEED),
        "n_folds": int(N_FOLDS),
    }
    thr_path = out_dir / "dev_oof_threshold_youden.json"
    thr_path.write_text(
        json.dumps(thr_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[SAVE] Threshold info:", thr_path)
    print(
        f"[INFO] Best threshold = {best_thr:.4f} | DEV-OOF YoudenJ={dev_oof_cm['YoudenJ']:.4f} Sens={dev_oof_cm['Sensitivity']:.4f} Spec={dev_oof_cm['Specificity']:.4f}"
    )
    print("\n" + "=" * 60)
    print(f"[STEP3] Retrain on ALL DEV and evaluate on {indep_split}")
    print("=" * 60)
    dev_ids_all = sp_dev["ID"].tolist()
    indep_ids = sp_indep["ID"].tolist()
    dev_all = df_idx.loc[dev_ids_all]
    indep_all = df_idx.loc[indep_ids]
    Xtr = dev_all[feature_cols]
    ytr = dev_all[y_col].values.astype(int)
    Xte = indep_all[feature_cols]
    yte = indep_all[y_col].values.astype(int)
    n_pos = int(ytr.sum())
    n_neg = int((ytr == 0).sum())
    pos_weight = n_neg / max(1, n_pos)
    est_final = build_estimator(model_name, params, strategy, pos_weight, SEED)
    pipe_final = build_pipeline(preprocessor, est_final, strategy, SEED)
    pipe_final.fit(Xtr, ytr)
    p_te = predict_proba_pos(pipe_final, Xte)
    indep_basic = safe_basic_metrics(yte, p_te)
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
    indep_pred_path = out_dir / "indep_predictions.csv"
    indep_pred.to_csv(indep_pred_path, index=False, encoding="utf-8-sig")
    print("[SAVE] Independent predictions:", indep_pred_path)
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
    cm_path = out_dir / "indep_confusion_matrix.csv"
    cm_tbl.to_csv(cm_path, index=False, encoding="utf-8-sig")
    print("[SAVE] Independent confusion matrix:", cm_path)
    indep_metrics = {
        "scheme": scheme_name,
        "split": indep_split,
        "threshold_used": float(best_thr),
        "threshold_rule": "YoudenJ on DEV OOF",
        "threshold_free": {k: float(v) for k, v in indep_basic.items()},
        "at_threshold": {k: float(v) if v == v else None for k, v in indep_cm.items()},
        "n_total": int(len(yte)),
        "n_pos": int(yte.sum()),
        "n_neg": int((yte == 0).sum()),
        "model": model_name,
        "strategy": strategy,
        "params": params,
        "seed": int(SEED),
    }
    indep_metrics_path = out_dir / "indep_metrics.json"
    indep_metrics_path.write_text(
        json.dumps(indep_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[SAVE] Independent metrics:", indep_metrics_path)
    print(
        f"[INFO] {indep_split} threshold-free: AUPRC={indep_basic['AUPRC']:.4f} AUROC={indep_basic['AUROC']:.4f} Brier={indep_basic['Brier']:.4f} LogLoss={indep_basic['LogLoss']:.4f}"
    )
    print(
        f"[INFO] {indep_split} at thr={best_thr:.4f}: Sens={indep_cm['Sensitivity']:.4f} Spec={indep_cm['Specificity']:.4f} PPV={indep_cm['PPV']:.4f} NPV={indep_cm['NPV']:.4f} F1={indep_cm['F1']:.4f}"
    )
    print("\n" + "=" * 60)
    print("[STEP4] Generate readable report table")
    print("=" * 60)
    report_rows = []
    report_rows.append(
        {
            "Scheme": scheme_name,
            "Dataset": "DEV_OOF",
            "n_total": int(len(dev_oof)),
            "n_pos": int(dev_oof["y_true"].sum()),
            "n_neg": int((dev_oof["y_true"].values == 0).sum()),
            "AUROC": float(dev_oof_basic["AUROC"]),
            "AUPRC": float(dev_oof_basic["AUPRC"]),
            "Brier": float(dev_oof_basic["Brier"]),
            "LogLoss": float(dev_oof_basic["LogLoss"]),
            "Threshold": float(best_thr),
            "Sensitivity": (
                float(dev_oof_cm["Sensitivity"])
                if dev_oof_cm["Sensitivity"] == dev_oof_cm["Sensitivity"]
                else np.nan
            ),
            "Specificity": (
                float(dev_oof_cm["Specificity"])
                if dev_oof_cm["Specificity"] == dev_oof_cm["Specificity"]
                else np.nan
            ),
            "PPV": (
                float(dev_oof_cm["PPV"])
                if dev_oof_cm["PPV"] == dev_oof_cm["PPV"]
                else np.nan
            ),
            "NPV": (
                float(dev_oof_cm["NPV"])
                if dev_oof_cm["NPV"] == dev_oof_cm["NPV"]
                else np.nan
            ),
            "F1": (
                float(dev_oof_cm["F1"])
                if dev_oof_cm["F1"] == dev_oof_cm["F1"]
                else np.nan
            ),
            "Accuracy": (
                float(dev_oof_cm["Accuracy"])
                if dev_oof_cm["Accuracy"] == dev_oof_cm["Accuracy"]
                else np.nan
            ),
            "BalancedAcc": (
                float(dev_oof_cm["BalancedAcc"])
                if dev_oof_cm["BalancedAcc"] == dev_oof_cm["BalancedAcc"]
                else np.nan
            ),
            "YoudenJ": (
                float(dev_oof_cm["YoudenJ"])
                if dev_oof_cm["YoudenJ"] == dev_oof_cm["YoudenJ"]
                else np.nan
            ),
        }
    )
    report_rows.append(
        {
            "Scheme": scheme_name,
            "Dataset": indep_split,
            "n_total": int(len(yte)),
            "n_pos": int(yte.sum()),
            "n_neg": int((yte == 0).sum()),
            "AUROC": float(indep_basic["AUROC"]),
            "AUPRC": float(indep_basic["AUPRC"]),
            "Brier": float(indep_basic["Brier"]),
            "LogLoss": float(indep_basic["LogLoss"]),
            "Threshold": float(best_thr),
            "Sensitivity": (
                float(indep_cm["Sensitivity"])
                if indep_cm["Sensitivity"] == indep_cm["Sensitivity"]
                else np.nan
            ),
            "Specificity": (
                float(indep_cm["Specificity"])
                if indep_cm["Specificity"] == indep_cm["Specificity"]
                else np.nan
            ),
            "PPV": (
                float(indep_cm["PPV"]) if indep_cm["PPV"] == indep_cm["PPV"] else np.nan
            ),
            "NPV": (
                float(indep_cm["NPV"]) if indep_cm["NPV"] == indep_cm["NPV"] else np.nan
            ),
            "F1": float(indep_cm["F1"]) if indep_cm["F1"] == indep_cm["F1"] else np.nan,
            "Accuracy": (
                float(indep_cm["Accuracy"])
                if indep_cm["Accuracy"] == indep_cm["Accuracy"]
                else np.nan
            ),
            "BalancedAcc": (
                float(indep_cm["BalancedAcc"])
                if indep_cm["BalancedAcc"] == indep_cm["BalancedAcc"]
                else np.nan
            ),
            "YoudenJ": (
                float(indep_cm["YoudenJ"])
                if indep_cm["YoudenJ"] == indep_cm["YoudenJ"]
                else np.nan
            ),
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
        report[c] = report[c].astype(float).round(4)
    report_path = out_dir / "report_metrics_table.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print("[SAVE] Report table:", report_path)
    print(report.to_string(index=False))
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
    curves_path = out_dir / "report_curves_roc_pr.csv"
    curves.to_csv(curves_path, index=False, encoding="utf-8-sig")
    print("[SAVE] Curve points (DEV OOF):", curves_path)
    print("\n[FINISHED] All outputs are in:", out_dir.resolve())
    return {
        "Scheme": scheme_name,
        "Model": model_name,
        "Strategy": strategy,
        "DEV_AUROC": float(dev_oof_basic["AUROC"]),
        "DEV_AUPRC": float(dev_oof_basic["AUPRC"]),
        "DEV_Brier": float(dev_oof_basic["Brier"]),
        "DEV_LogLoss": float(dev_oof_basic["LogLoss"]),
        "DEV_Threshold": float(best_thr),
        "DEV_Sensitivity": (
            float(dev_oof_cm["Sensitivity"])
            if dev_oof_cm["Sensitivity"] == dev_oof_cm["Sensitivity"]
            else np.nan
        ),
        "DEV_Specificity": (
            float(dev_oof_cm["Specificity"])
            if dev_oof_cm["Specificity"] == dev_oof_cm["Specificity"]
            else np.nan
        ),
        "DEV_PPV": (
            float(dev_oof_cm["PPV"])
            if dev_oof_cm["PPV"] == dev_oof_cm["PPV"]
            else np.nan
        ),
        "DEV_NPV": (
            float(dev_oof_cm["NPV"])
            if dev_oof_cm["NPV"] == dev_oof_cm["NPV"]
            else np.nan
        ),
        "DEV_F1": (
            float(dev_oof_cm["F1"]) if dev_oof_cm["F1"] == dev_oof_cm["F1"] else np.nan
        ),
        "DEV_YoudenJ": (
            float(dev_oof_cm["YoudenJ"])
            if dev_oof_cm["YoudenJ"] == dev_oof_cm["YoudenJ"]
            else np.nan
        ),
        "INDEP_Split": indep_split,
        "INDEP_AUROC": float(indep_basic["AUROC"]),
        "INDEP_AUPRC": float(indep_basic["AUPRC"]),
        "INDEP_Brier": float(indep_basic["Brier"]),
        "INDEP_LogLoss": float(indep_basic["LogLoss"]),
        "INDEP_Threshold": float(best_thr),
        "INDEP_Sensitivity": (
            float(indep_cm["Sensitivity"])
            if indep_cm["Sensitivity"] == indep_cm["Sensitivity"]
            else np.nan
        ),
        "INDEP_Specificity": (
            float(indep_cm["Specificity"])
            if indep_cm["Specificity"] == indep_cm["Specificity"]
            else np.nan
        ),
        "INDEP_PPV": (
            float(indep_cm["PPV"]) if indep_cm["PPV"] == indep_cm["PPV"] else np.nan
        ),
        "INDEP_NPV": (
            float(indep_cm["NPV"]) if indep_cm["NPV"] == indep_cm["NPV"] else np.nan
        ),
        "INDEP_F1": (
            float(indep_cm["F1"]) if indep_cm["F1"] == indep_cm["F1"] else np.nan
        ),
        "INDEP_YoudenJ": (
            float(indep_cm["YoudenJ"])
            if indep_cm["YoudenJ"] == indep_cm["YoudenJ"]
            else np.nan
        ),
    }


def main():
    fix_seed(SEED)
    for p in [DATA_CSV, SPLIT_CSV, FEATURE_TYPES_CSV, SCHEME_PARAMS_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")
    if not isinstance(TARGET_SCHEMES, list) or len(TARGET_SCHEMES) == 0:
        raise ValueError("TARGET_SCHEMES must be a non-empty list.")
    print("[INFO] TARGET_SCHEMES =", TARGET_SCHEMES)
    summary_rows = []
    for scheme in TARGET_SCHEMES:
        print("\n" + "#" * 80)
        print(f"[RUN] Start scheme: {scheme}")
        print("#" * 80)
        try:
            row = run_one_scheme(
                target_scheme=scheme,
                data_csv=DATA_CSV,
                split_csv=SPLIT_CSV,
                feature_types_csv=FEATURE_TYPES_CSV,
                scheme_params_csv=SCHEME_PARAMS_CSV,
                out_root=OUT_ROOT,
            )
            summary_rows.append(row)
            print(f"[DONE] Finished scheme: {scheme}")
        except Exception as e:
            print(f"[ERROR] Scheme failed: {scheme}")
            print(f"[ERROR] {repr(e)}")
    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        metric_cols = [
            c
            for c in summary_df.columns
            if c not in ["Scheme", "Model", "Strategy", "INDEP_Split"]
        ]
        for c in metric_cols:
            summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce").round(4)
        summary_path = OUT_ROOT / "multi_scheme_summary_dev_indep.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 80)
        print("[SUMMARY] Multi-scheme summary saved:")
        print(summary_path.resolve())
        print(summary_df.to_string(index=False))
    else:
        print("[WARNING] No scheme completed successfully.")


if __name__ == "__main__":
    main()
