"""Evaluate ranked feature subsets with the saved model configurations.

Generate DEV out-of-fold and held-out metrics for each subset size.
The manuscript selects the final subset using DEV performance; held-out
summaries describe evaluation results and are not the selection criterion."""

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
INPUT_ROOT = Path("outputs/feature_selection")
OUTPUT_ROOT = Path("outputs/feature_subsets")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MAX_TOPK = 20
N_THRESH = 201
EPS = 1e-15
EXCLUDE_COLS = ["ID", "Infection", "Infection_split"]
SMOTE_K_NEIGHBORS = 3
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_REG_LAMBDA = 1.0


def fix_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_params_string(params_str: str) -> Dict[str, Any]:
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
            f"Scheme={target_scheme} not found in {csv_path}.\nAvailable schemes: {df['scheme'].astype(str).tolist()}"
        )
    if len(hit) > 1:
        raise ValueError(f"Scheme={target_scheme} matched multiple rows.")
    row = hit.iloc[0]
    return {
        "scheme": str(row["scheme"]),
        "model": str(row["model"]),
        "strategy": str(row["strategy"]),
        "params": parse_params_string(row["params"]),
    }


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
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
    }


def find_best_threshold_youden(
    y_true: np.ndarray, p: np.ndarray, n_thresh: int = 201
) -> Tuple[float, Dict[str, float]]:
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
    return (best_thr, best_row)


def detect_indep_split(sp: pd.DataFrame) -> str:
    splits = sp["split"].astype(str).str.upper().unique().tolist()
    for cand in ["VAL", "INDEP_TEST", "TEST"]:
        if cand in splits:
            return cand
    raise ValueError(f"Cannot find independent split among {splits}.")


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


def read_contribution_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing contribution file: {path.resolve()}")
    df = pd.read_csv(path)
    required = ["Scheme", "Model", "Strategy", "feature", "Contribution_pct", "Rank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Contribution_pct"] = pd.to_numeric(df["Contribution_pct"], errors="coerce")
    df = df.sort_values(
        ["Rank", "Contribution_pct"], ascending=[True, False]
    ).reset_index(drop=True)
    return df


def sort_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the output metric table using the configured ranking columns."""
    required_cols = ["DEV_AUPRC", "DEV_AUROC", "INDEP_AUPRC", "TopK"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing required columns for sorting: {missing}")
        return df
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df_sorted = df.sort_values(
        by=["DEV_AUPRC", "DEV_AUROC", "INDEP_AUPRC", "TopK"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return df_sorted


def evaluate_topk_for_one_scheme(
    contribution_df: pd.DataFrame,
    scheme_type: str,
    data_csv: Path,
    split_csv: Path,
    feature_types_csv: Path,
    scheme_params_csv: Path,
    max_topk: int = 30,
) -> pd.DataFrame:
    scheme_name = str(contribution_df["Scheme"].iloc[0])
    model_name = str(contribution_df["Model"].iloc[0])
    strategy = str(contribution_df["Strategy"].iloc[0])
    cfg = load_target_scheme_config(scheme_params_csv, scheme_name)
    params = cfg["params"]
    df = pd.read_csv(data_csv)
    sp = pd.read_csv(split_csv)
    for c in ["ID", "split", "fold_id", "Infection"]:
        if c not in sp.columns:
            raise ValueError(f"split_folds.csv missing column: {c}")
    indep_split = detect_indep_split(sp)
    sp_dev = sp[sp["split"].astype(str).str.upper() == "DEV"].copy()
    sp_dev = sp_dev[sp_dev["fold_id"].between(0, N_FOLDS - 1)].copy()
    if sp_dev.empty:
        raise ValueError("No DEV rows found.")
    folds = sorted(sp_dev["fold_id"].unique().tolist())
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
    ranked_features = contribution_df["feature"].astype(str).tolist()
    ranked_features = [f for f in ranked_features if f in df.columns]
    if len(ranked_features) == 0:
        return pd.DataFrame()
    topk_max = min(max_topk, len(ranked_features))
    exclude_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    cont_all, bin_all, cat_all = load_feature_types(
        feature_types_csv, df.columns.tolist(), exclude=exclude_cols
    )
    df_idx = df.set_index("ID", drop=False)
    rows = []
    for k in range(1, topk_max + 1):
        feature_cols = ranked_features[:k]
        cont_cols = [c for c in cont_all if c in feature_cols]
        bin_cols = [c for c in bin_all if c in feature_cols]
        cat_cols = [c for c in cat_all if c in feature_cols]
        preprocessor = build_preprocessor(cont_cols, bin_cols, cat_cols)
        oof_rows = []
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
            for _id, yt, pr in zip(
                va["ID"].values.tolist(), yva.tolist(), pva.tolist()
            ):
                oof_rows.append(
                    {
                        "ID": _id,
                        "y_true": int(yt),
                        "oof_prob": float(pr),
                        "fold_id": int(f),
                    }
                )
        dev_oof = (
            pd.DataFrame(oof_rows).sort_values(["fold_id", "ID"]).reset_index(drop=True)
        )
        y_dev = dev_oof["y_true"].values.astype(int)
        p_dev = dev_oof["oof_prob"].values.astype(float)
        dev_basic = safe_basic_metrics(y_dev, p_dev)
        best_thr, best_thr_row = find_best_threshold_youden(
            y_dev, p_dev, n_thresh=N_THRESH
        )
        dev_at_thr = confusion_metrics(y_dev, p_dev, best_thr)
        dev_ids_all = sp_dev["ID"].tolist()
        indep_ids = sp_indep["ID"].tolist()
        dev_all = df_idx.loc[dev_ids_all]
        indep_all = df_idx.loc[indep_ids]
        Xtr_all = dev_all[feature_cols]
        ytr_all = dev_all[y_col].values.astype(int)
        Xte = indep_all[feature_cols]
        yte = indep_all[y_col].values.astype(int)
        n_pos = int(ytr_all.sum())
        n_neg = int((ytr_all == 0).sum())
        pos_weight = n_neg / max(1, n_pos)
        est_final = build_estimator(model_name, params, strategy, pos_weight, SEED)
        pipe_final = build_pipeline(preprocessor, est_final, strategy, SEED)
        pipe_final.fit(Xtr_all, ytr_all)
        p_te = predict_proba_pos(pipe_final, Xte)
        indep_basic = safe_basic_metrics(yte, p_te)
        indep_at_thr = confusion_metrics(yte, p_te, best_thr)
        rows.append(
            {
                "Scheme": scheme_name,
                "Model": model_name,
                "Strategy": strategy,
                "FeatureSetType": scheme_type,
                "TopK": k,
                "DEV_n_total": int(len(y_dev)),
                "DEV_n_pos": int(np.sum(y_dev)),
                "DEV_n_neg": int(np.sum(y_dev == 0)),
                "DEV_AUROC": dev_basic["AUROC"],
                "DEV_AUPRC": dev_basic["AUPRC"],
                "DEV_Brier": dev_basic["Brier"],
                "DEV_LogLoss": dev_basic["LogLoss"],
                "DEV_Thr_used": best_thr,
                "DEV_Sensitivity": dev_at_thr["Sensitivity"],
                "DEV_Specificity": dev_at_thr["Specificity"],
                "DEV_PPV": dev_at_thr["PPV"],
                "DEV_NPV": dev_at_thr["NPV"],
                "DEV_F1": dev_at_thr["F1"],
                "DEV_Accuracy": dev_at_thr["Accuracy"],
                "DEV_BalancedAcc": dev_at_thr["BalancedAcc"],
                "DEV_YoudenJ": dev_at_thr["YoudenJ"],
                "DEV_FPR": dev_at_thr["FPR"],
                "DEV_FNR": dev_at_thr["FNR"],
                "DEV_TP": dev_at_thr["TP"],
                "DEV_FP": dev_at_thr["FP"],
                "DEV_TN": dev_at_thr["TN"],
                "DEV_FN": dev_at_thr["FN"],
                "INDEP_Split": indep_split,
                "INDEP_n_total": int(len(yte)),
                "INDEP_n_pos": int(np.sum(yte)),
                "INDEP_n_neg": int(np.sum(yte == 0)),
                "INDEP_AUROC": indep_basic["AUROC"],
                "INDEP_AUPRC": indep_basic["AUPRC"],
                "INDEP_Brier": indep_basic["Brier"],
                "INDEP_LogLoss": indep_basic["LogLoss"],
                "INDEP_Thr_used": best_thr,
                "INDEP_Sensitivity": indep_at_thr["Sensitivity"],
                "INDEP_Specificity": indep_at_thr["Specificity"],
                "INDEP_PPV": indep_at_thr["PPV"],
                "INDEP_NPV": indep_at_thr["NPV"],
                "INDEP_F1": indep_at_thr["F1"],
                "INDEP_Accuracy": indep_at_thr["Accuracy"],
                "INDEP_BalancedAcc": indep_at_thr["BalancedAcc"],
                "INDEP_YoudenJ": indep_at_thr["YoudenJ"],
                "INDEP_FPR": indep_at_thr["FPR"],
                "INDEP_FNR": indep_at_thr["FNR"],
                "INDEP_TP": indep_at_thr["TP"],
                "INDEP_FP": indep_at_thr["FP"],
                "INDEP_TN": indep_at_thr["TN"],
                "INDEP_FN": indep_at_thr["FN"],
            }
        )
        print(
            f"[OK] {scheme_name} | {scheme_type} | top-{k} done | INDEP_AUPRC={indep_basic['AUPRC']:.4f} INDEP_F1={indep_at_thr['F1']:.4f}"
        )
    out = pd.DataFrame(rows)
    num_cols = [
        c
        for c in out.columns
        if c not in ["Scheme", "Model", "Strategy", "FeatureSetType", "INDEP_Split"]
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def main():
    fix_seed(SEED)
    for p in [DATA_CSV, SPLIT_CSV, FEATURE_TYPES_CSV, SCHEME_PARAMS_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT.resolve()}")
    scheme_dirs = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]
    scheme_dirs = sorted(scheme_dirs, key=lambda x: x.name)
    all_scheme1_outputs = []
    all_scheme2_outputs = []
    print("[INFO] Found scheme dirs:", [p.name for p in scheme_dirs])
    for scheme_dir in scheme_dirs:
        print("\n" + "=" * 80)
        print(f"[RUN] Processing scheme dir: {scheme_dir.name}")
        print("=" * 80)
        path1 = scheme_dir / "contribution_scheme1.csv"
        path2 = scheme_dir / "contribution_scheme2.csv"
        if path1.exists():
            df1 = read_contribution_file(path1)
            out1 = evaluate_topk_for_one_scheme(
                contribution_df=df1,
                scheme_type="scheme1",
                data_csv=DATA_CSV,
                split_csv=SPLIT_CSV,
                feature_types_csv=FEATURE_TYPES_CSV,
                scheme_params_csv=SCHEME_PARAMS_CSV,
                max_topk=MAX_TOPK,
            )
            save1 = scheme_dir / "topk_metrics_scheme1.csv"
            out1.to_csv(save1, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {save1}")
            all_scheme1_outputs.append(out1)
        else:
            print(f"[WARN] Missing: {path1}")
        if path2.exists():
            df2 = read_contribution_file(path2)
            out2 = evaluate_topk_for_one_scheme(
                contribution_df=df2,
                scheme_type="scheme2",
                data_csv=DATA_CSV,
                split_csv=SPLIT_CSV,
                feature_types_csv=FEATURE_TYPES_CSV,
                scheme_params_csv=SCHEME_PARAMS_CSV,
                max_topk=MAX_TOPK,
            )
            save2 = scheme_dir / "topk_metrics_scheme2.csv"
            out2.to_csv(save2, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {save2}")
            all_scheme2_outputs.append(out2)
        else:
            print(f"[WARN] Missing: {path2}")
    if len(all_scheme1_outputs) > 0:
        all1 = pd.concat(all_scheme1_outputs, axis=0, ignore_index=True)
        save_all1 = OUTPUT_ROOT / "all_schemes_topk_metrics_scheme1.csv"
        all1.to_csv(save_all1, index=False, encoding="utf-8-sig")
        print(f"\n[SAVE] {save_all1}")
        all1_sorted = sort_metrics_dataframe(all1)
        save_sorted1 = OUTPUT_ROOT / "ranked_median.csv"
        all1_sorted.to_csv(save_sorted1, index=False, encoding="utf-8-sig")
        print(f"[SAVE] Ranked metric table: {save_sorted1}")
    if len(all_scheme2_outputs) > 0:
        all2 = pd.concat(all_scheme2_outputs, axis=0, ignore_index=True)
        save_all2 = OUTPUT_ROOT / "all_schemes_topk_metrics_scheme2.csv"
        all2.to_csv(save_all2, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {save_all2}")
        all2_sorted = sort_metrics_dataframe(all2)
        save_sorted2 = OUTPUT_ROOT / "ranked_mean.csv"
        all2_sorted.to_csv(save_sorted2, index=False, encoding="utf-8-sig")
        print(f"[SAVE] Ranked metric table: {save_sorted2}")
    print("\n[FINISHED] Done.")


if __name__ == "__main__":
    main()
