# -*- coding: utf-8 -*-
"""
multi_scheme_feature_importance_internal.py

内部筛选专用：
- 多方案特征重要性提取
- 主标准：Permutation Importance (PI)
- 模型内补充：
    * CatBoost   : PVC + LFC
    * XGBoost    : gain
    * ExtraTrees : feature_importances_

输入：
- DATA_CSV
- SPLIT_CSV
- FEATURE_TYPES_CSV
- SCHEME_PARAMS_CSV

输出：
OUT_ROOT/
  ├── <scheme_name>/
  │    ├── selected_scheme_config.json
  │    ├── feature_importance_foldwise.csv
  │    └── feature_importance_summary.csv
  └── all_schemes_feature_importance_summary.csv

说明：
1) PI 在每个 fold 的验证集上计算（AUPRC/average_precision 作为评分）
2) 模型内重要性在每个 fold 的已训练模型上提取
3) 所有展开后的特征（one-hot、missing indicator）都会聚合回原始特征
"""

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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier


# =========================
# Fixed Config (EDIT HERE ONLY)
# =========================
SEED = 42
N_FOLDS = 5

DATA_CSV = Path("开腹肝胆道结石手术SSI清洗.csv")
SPLIT_CSV = Path("splits/split_folds_full.csv")
FEATURE_TYPES_CSV = Path("特征类别/feature_types_summary.csv")

SCHEME_PARAMS_CSV = Path("训练全特征/scheme_to_best_params.csv")

TARGET_SCHEMES = [
    "CatBoost+S0",
    "CatBoost+S1",
    "CatBoost+S2",
    "CatBoost+S3",
]      

"""
    "CatBoost+S0",
    "CatBoost+S1",
    "CatBoost+S2",
    "CatBoost+S3",
    "ExtraTrees+S0",
    "ExtraTrees+S1",
    "ExtraTrees+S2",
    "ExtraTrees+S3",
    "XGBoost+S3",
"""

OUT_ROOT = Path("特征重要性")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = ["ID", "Infection", "Infection_split"]
SMOTE_K_NEIGHBORS = 3

# XGB stable knobs
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_REG_LAMBDA = 1.0

# Permutation Importance config
PI_SCORING = "average_precision"  # AUPRC
PI_N_REPEATS = 30                 # 内部筛选够用；想更稳可改 30
PI_N_JOBS = -1


# =========================
# Reproducibility
# =========================
def fix_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# =========================
# Scheme loading
# =========================
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
        raise FileNotFoundError(f"scheme_to_best_params.csv not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    required_cols = ["scheme", "model", "strategy", "params"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"scheme_to_best_params.csv missing columns: {missing}")

    hit = df[df["scheme"].astype(str) == str(target_scheme)].copy()
    if hit.empty:
        raise ValueError(
            f"TARGET_SCHEME={target_scheme} not found in {csv_path}.\n"
            f"Available schemes: {df['scheme'].astype(str).tolist()}"
        )

    if len(hit) > 1:
        raise ValueError(f"TARGET_SCHEME={target_scheme} matched multiple rows, please check csv.")

    row = hit.iloc[0]
    cfg = {
        "scheme": str(row["scheme"]),
        "model": str(row["model"]),
        "strategy": str(row["strategy"]),
        "params": parse_params_string(row["params"]),
    }
    return cfg


# =========================
# Feature types & preprocessing
# =========================
def load_feature_types(ft_path: Path, df_columns: List[str], exclude: List[str]) -> Tuple[List[str], List[str], List[str]]:
    ft = pd.read_csv(ft_path)
    if "feature" not in ft.columns or "type" not in ft.columns:
        raise ValueError("feature_types_summary.csv must contain columns: feature, type")

    feature_set = set(df_columns)
    exclude_set = set(exclude)

    cont, binary, cat = [], [], []
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

    return uniq(cont), uniq(binary), uniq(cat)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(continuous_cols: List[str], binary_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []

    # Continuous: median + missing indicator
    if continuous_cols:
        cont_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True))
        ])
        transformers.append(("cont", cont_pipe, continuous_cols))

    # Binary: mode
    if binary_cols:
        bin_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent"))
        ])
        transformers.append(("bin", bin_pipe, binary_cols))

    # Categorical: mode + to_str + onehot
    if cat_cols:
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("to_str", FunctionTransformer(lambda x: x.astype(str), feature_names_out="one-to-one")),
            ("ohe", make_ohe())
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


# =========================
# Model builders
# =========================
def build_estimator(model_name: str, params: Dict[str, Any], strategy: str, pos_weight: float, seed: int):
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


def build_pipeline(preprocessor: ColumnTransformer, estimator, strategy: str, seed: int):
    if strategy in ("S2", "S3"):
        ratio = (1.0 / 3.0) if strategy == "S2" else (1.0 / 2.0)
        smote = SMOTE(
            sampling_strategy=ratio,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=seed,
        )
        return IMBPipeline([
            ("prep", preprocessor),
            ("smote", smote),
            ("clf", estimator),
        ])
    else:
        return Pipeline([
            ("prep", preprocessor),
            ("clf", estimator),
        ])


# =========================
# Split handling
# =========================
def get_dev_split(sp: pd.DataFrame) -> pd.DataFrame:
    sp_dev = sp[sp["split"].astype(str).str.upper() == "DEV"].copy()
    sp_dev = sp_dev[sp_dev["fold_id"].between(0, N_FOLDS - 1)].copy()
    if sp_dev.empty:
        raise ValueError("No DEV rows found in split file. Check split/fold_id.")
    return sp_dev


# =========================
# Feature name helpers
# =========================
def get_transformed_feature_names(preprocessor_fitted: ColumnTransformer) -> List[str]:
    try:
        names = preprocessor_fitted.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        pass

    names = []
    for name, trans, cols in preprocessor_fitted.transformers_:
        if name == "remainder":
            continue

        if name == "cont":
            cols = list(cols)
            names.extend([f"cont__{c}" for c in cols])
            imp = trans.named_steps["imp"]
            if hasattr(imp, "indicator_") and imp.indicator_ is not None:
                miss_idx = imp.indicator_.features_
                for j in miss_idx:
                    names.append(f"cont__{cols[j]}__missing")

        elif name == "bin":
            names.extend([f"bin__{c}" for c in cols])

        elif name == "cat":
            ohe = trans.named_steps["ohe"]
            try:
                cat_names = ohe.get_feature_names_out(cols)
                names.extend([f"cat__{str(x)}" for x in cat_names])
            except Exception:
                names.extend([f"cat__{c}" for c in cols])

    return names


def raw_group_from_transformed_name(name: str, raw_features: List[str]) -> str:
    s = str(name)
    s = s.replace("cont__", "").replace("bin__", "").replace("cat__", "")

    if s in raw_features:
        return s

    if s.endswith("__missing"):
        base = s.replace("__missing", "")
        if base in raw_features:
            return base

    for rf in sorted(raw_features, key=len, reverse=True):
        if s == rf:
            return rf
        if s.startswith(rf + "_"):
            return rf
        if rf in s:
            return rf

    return s


def aggregate_transformed_importance_to_raw(
    values: np.ndarray,
    transformed_feature_names: List[str],
    raw_feature_names: List[str]
) -> Dict[str, float]:
    """
    将展开后的特征重要性聚合回原始特征。
    聚合规则：同一原始特征下所有展开列直接求和。
    """
    values = np.asarray(values, dtype=float)
    if len(values) != len(transformed_feature_names):
        raise ValueError("Length mismatch between values and transformed_feature_names.")

    out = {rf: 0.0 for rf in raw_feature_names}

    for v, tf in zip(values, transformed_feature_names):
        rf = raw_group_from_transformed_name(tf, raw_feature_names)
        if rf in out:
            out[rf] += float(v)

    return out


# =========================
# Model-specific importance extractors
# =========================
def extract_catboost_internal_importance(
    clf,
    X_val_trans: np.ndarray,
    y_val: np.ndarray,
    transformed_feature_names: List[str],
    raw_feature_names: List[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
      pvc_raw_dict, lfc_raw_dict
    """
    pvc_vals = clf.get_feature_importance(type="PredictionValuesChange")

    pool_val = Pool(
        data=X_val_trans,
        label=y_val,
        feature_names=transformed_feature_names
    )
    lfc_vals = clf.get_feature_importance(data=pool_val, type="LossFunctionChange")

    pvc_raw = aggregate_transformed_importance_to_raw(
        pvc_vals, transformed_feature_names, raw_feature_names
    )
    lfc_raw = aggregate_transformed_importance_to_raw(
        lfc_vals, transformed_feature_names, raw_feature_names
    )
    return pvc_raw, lfc_raw


def extract_xgboost_gain_importance(
    clf,
    transformed_feature_names: List[str],
    raw_feature_names: List[str]
) -> Dict[str, float]:
    """
    XGBoost booster.get_score(importance_type='gain')
    keys like f0, f1, ...
    """
    booster = clf.get_booster()
    gain_dict = booster.get_score(importance_type="gain")

    vals = np.zeros(len(transformed_feature_names), dtype=float)
    for i in range(len(transformed_feature_names)):
        key = f"f{i}"
        vals[i] = float(gain_dict.get(key, 0.0))

    gain_raw = aggregate_transformed_importance_to_raw(
        vals, transformed_feature_names, raw_feature_names
    )
    return gain_raw


def extract_extratrees_importance(
    clf,
    transformed_feature_names: List[str],
    raw_feature_names: List[str]
) -> Dict[str, float]:
    vals = np.asarray(clf.feature_importances_, dtype=float)

    et_raw = aggregate_transformed_importance_to_raw(
        vals, transformed_feature_names, raw_feature_names
    )
    return et_raw


# =========================
# Summary
# =========================
def make_summary_table(
    foldwise_df: pd.DataFrame,
    scheme_name: str,
    model_name: str,
    strategy: str
) -> pd.DataFrame:
    """
    把每折结果聚合为每个特征一行的 summary 表
    """
    features = sorted(foldwise_df["feature"].astype(str).unique().tolist())

    rows = []
    for feat in features:
        sub = foldwise_df[foldwise_df["feature"].astype(str) == feat].copy()

        row = {
            "Scheme": scheme_name,
            "Model": model_name,
            "Strategy": strategy,
            "feature": feat,
        }

        # PI
        pi_vals = sub["PI_mean"].astype(float).values
        row["PI_mean"] = np.mean(pi_vals)
        row["PI_median"] = np.median(pi_vals)
        row["PI_std"] = np.std(pi_vals, ddof=0)
        row["PI_pos_ratio"] = np.mean(pi_vals > 0)

        # CatBoost supplements
        if "PVC" in sub.columns:
            pvc_vals = sub["PVC"].astype(float).values
            if np.isfinite(pvc_vals).any():
                row["PVC_mean"] = np.nanmean(pvc_vals)
                row["PVC_median"] = np.nanmedian(pvc_vals)
                row["PVC_std"] = np.nanstd(pvc_vals)
                row["PVC_pos_ratio"] = np.nanmean(pvc_vals > 0)
            else:
                row["PVC_mean"] = np.nan
                row["PVC_median"] = np.nan
                row["PVC_std"] = np.nan
                row["PVC_pos_ratio"] = np.nan

        if "LFC" in sub.columns:
            lfc_vals = sub["LFC"].astype(float).values
            if np.isfinite(lfc_vals).any():
                row["LFC_mean"] = np.nanmean(lfc_vals)
                row["LFC_median"] = np.nanmedian(lfc_vals)
                row["LFC_std"] = np.nanstd(lfc_vals)
                row["LFC_pos_ratio"] = np.nanmean(lfc_vals > 0)
            else:
                row["LFC_mean"] = np.nan
                row["LFC_median"] = np.nan
                row["LFC_std"] = np.nan
                row["LFC_pos_ratio"] = np.nan

        # XGBoost supplement
        if "GAIN" in sub.columns:
            gain_vals = sub["GAIN"].astype(float).values
            if np.isfinite(gain_vals).any():
                row["GAIN_mean"] = np.nanmean(gain_vals)
                row["GAIN_median"] = np.nanmedian(gain_vals)
                row["GAIN_std"] = np.nanstd(gain_vals)
                row["GAIN_pos_ratio"] = np.nanmean(gain_vals > 0)
            else:
                row["GAIN_mean"] = np.nan
                row["GAIN_median"] = np.nan
                row["GAIN_std"] = np.nan
                row["GAIN_pos_ratio"] = np.nan

        # ExtraTrees supplement
        if "ET_IMP" in sub.columns:
            et_vals = sub["ET_IMP"].astype(float).values
            if np.isfinite(et_vals).any():
                row["ET_IMP_mean"] = np.nanmean(et_vals)
                row["ET_IMP_median"] = np.nanmedian(et_vals)
                row["ET_IMP_std"] = np.nanstd(et_vals)
                row["ET_IMP_pos_ratio"] = np.nanmean(et_vals > 0)
            else:
                row["ET_IMP_mean"] = np.nan
                row["ET_IMP_median"] = np.nan
                row["ET_IMP_std"] = np.nan
                row["ET_IMP_pos_ratio"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    # 排序：优先 PI_mean，再 PI_median
    out = out.sort_values(
        by=["PI_mean", "PI_median", "feature"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # 美化数值
    num_cols = [c for c in out.columns if c not in ["Scheme", "Model", "Strategy", "feature"]]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)

    return out


# =========================
# Core runner
# =========================
def run_one_scheme(
    target_scheme: str,
    data_csv: Path,
    split_csv: Path,
    feature_types_csv: Path,
    scheme_params_csv: Path,
    out_root: Path,
) -> pd.DataFrame:
    cfg = load_target_scheme_config(scheme_params_csv, target_scheme)
    scheme_name = cfg["scheme"]
    model_name = cfg["model"]
    strategy = cfg["strategy"]
    params = cfg["params"]

    out_dir = out_root / scheme_name
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_cfg_path = out_dir / "selected_scheme_config.json"
    selected_cfg_path.write_text(
        json.dumps({
            "scheme": scheme_name,
            "model": model_name,
            "strategy": strategy,
            "params": params,
            "seed": SEED,
            "n_folds": N_FOLDS,
            "PI_SCORING": PI_SCORING,
            "PI_N_REPEATS": PI_N_REPEATS,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(f"[RUN] Scheme: {scheme_name}")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    print(f"[INFO] Output dir: {out_dir.resolve()}")
    print("=" * 80)

    # load data
    df = pd.read_csv(data_csv)
    sp = pd.read_csv(split_csv)

    for c in ["ID", "split", "fold_id", "Infection"]:
        if c not in sp.columns:
            raise ValueError(f"split_folds_full.csv missing column: {c}")

    sp_dev = get_dev_split(sp)
    folds = sorted(sp_dev["fold_id"].unique().tolist())

    # merge authoritative labels
    df = df.merge(
        sp[["ID", "Infection"]],
        on="ID",
        how="inner",
        suffixes=("", "_split"),
        validate="one_to_one"
    )
    y_col = "Infection_split" if "Infection_split" in df.columns else "Infection"
    df[y_col] = pd.to_numeric(df[y_col], errors="raise").astype(int)

    exclude_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    raw_feature_names = [c for c in df.columns if c not in exclude_cols]

    # feature types
    cont_cols, bin_cols, cat_cols = load_feature_types(
        feature_types_csv,
        df.columns.tolist(),
        exclude=exclude_cols
    )
    cont_cols = [c for c in cont_cols if c in raw_feature_names]
    bin_cols = [c for c in bin_cols if c in raw_feature_names]
    cat_cols = [c for c in cat_cols if c in raw_feature_names]

    print(f"[INFO] Feature types: continuous={len(cont_cols)} | binary={len(bin_cols)} | categorical={len(cat_cols)}")

    preprocessor = build_preprocessor(cont_cols, bin_cols, cat_cols)
    df_idx = df.set_index("ID", drop=False)

    foldwise_rows = []

    for f in folds:
        print("-" * 60)
        print(f"[FOLD] {scheme_name} | fold={f}")

        tr_ids = sp_dev.loc[sp_dev["fold_id"] != f, "ID"].tolist()
        va_ids = sp_dev.loc[sp_dev["fold_id"] == f, "ID"].tolist()

        tr = df_idx.loc[tr_ids]
        va = df_idx.loc[va_ids]

        Xtr = tr[raw_feature_names]
        ytr = tr[y_col].values.astype(int)
        Xva = va[raw_feature_names]
        yva = va[y_col].values.astype(int)

        n_pos = int(ytr.sum())
        n_neg = int((ytr == 0).sum())
        pos_weight = n_neg / max(1, n_pos)

        est = build_estimator(model_name, params, strategy, pos_weight, SEED)
        pipe = build_pipeline(preprocessor, est, strategy, SEED)
        pipe.fit(Xtr, ytr)

        # ===== 1) PI on validation fold =====
        pi = permutation_importance(
            estimator=pipe,
            X=Xva,
            y=yva,
            scoring=PI_SCORING,
            n_repeats=PI_N_REPEATS,
            random_state=SEED,
            n_jobs=PI_N_JOBS
        )

        pi_df = pd.DataFrame({
            "feature": raw_feature_names,
            "PI_mean": pi.importances_mean.astype(float),
            "PI_std": pi.importances_std.astype(float),
        })

        # ===== 2) Model-specific internal importance =====
        prep_fitted = pipe.named_steps["prep"]
        clf_fitted = pipe.named_steps["clf"]

        Xva_trans = np.asarray(prep_fitted.transform(Xva))
        transformed_feature_names = get_transformed_feature_names(prep_fitted)

        if model_name == "CatBoost":
            pvc_raw, lfc_raw = extract_catboost_internal_importance(
                clf=clf_fitted,
                X_val_trans=Xva_trans,
                y_val=yva,
                transformed_feature_names=transformed_feature_names,
                raw_feature_names=raw_feature_names
            )
            internal_df = pd.DataFrame({
                "feature": raw_feature_names,
                "PVC": [pvc_raw.get(fea, np.nan) for fea in raw_feature_names],
                "LFC": [lfc_raw.get(fea, np.nan) for fea in raw_feature_names],
            })

        elif model_name == "XGBoost":
            gain_raw = extract_xgboost_gain_importance(
                clf=clf_fitted,
                transformed_feature_names=transformed_feature_names,
                raw_feature_names=raw_feature_names
            )
            internal_df = pd.DataFrame({
                "feature": raw_feature_names,
                "GAIN": [gain_raw.get(fea, np.nan) for fea in raw_feature_names],
            })

        elif model_name == "ExtraTrees":
            et_raw = extract_extratrees_importance(
                clf=clf_fitted,
                transformed_feature_names=transformed_feature_names,
                raw_feature_names=raw_feature_names
            )
            internal_df = pd.DataFrame({
                "feature": raw_feature_names,
                "ET_IMP": [et_raw.get(fea, np.nan) for fea in raw_feature_names],
            })

        else:
            internal_df = pd.DataFrame({"feature": raw_feature_names})

        fold_df = pi_df.merge(internal_df, on="feature", how="left")
        fold_df.insert(0, "fold_id", int(f))
        fold_df.insert(0, "Strategy", strategy)
        fold_df.insert(0, "Model", model_name)
        fold_df.insert(0, "Scheme", scheme_name)

        foldwise_rows.append(fold_df)

    foldwise_df = pd.concat(foldwise_rows, axis=0, ignore_index=True)

    # 保存每折明细
    foldwise_path = out_dir / "feature_importance_foldwise.csv"
    foldwise_df.to_csv(foldwise_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Foldwise importance: {foldwise_path}")

    # 汇总表
    summary_df = make_summary_table(
        foldwise_df=foldwise_df,
        scheme_name=scheme_name,
        model_name=model_name,
        strategy=strategy
    )
    summary_path = out_dir / "feature_importance_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Summary importance: {summary_path}")

    return summary_df


# =========================
# Main
# =========================
def main():
    fix_seed(SEED)

    for p in [DATA_CSV, SPLIT_CSV, FEATURE_TYPES_CSV, SCHEME_PARAMS_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")

    if not isinstance(TARGET_SCHEMES, list) or len(TARGET_SCHEMES) == 0:
        raise ValueError("TARGET_SCHEMES must be a non-empty list.")

    print("[INFO] TARGET_SCHEMES =", TARGET_SCHEMES)

    all_summary = []

    for scheme in TARGET_SCHEMES:
        try:
            summary_df = run_one_scheme(
                target_scheme=scheme,
                data_csv=DATA_CSV,
                split_csv=SPLIT_CSV,
                feature_types_csv=FEATURE_TYPES_CSV,
                scheme_params_csv=SCHEME_PARAMS_CSV,
                out_root=OUT_ROOT,
            )
            all_summary.append(summary_df)
            print(f"[DONE] Finished scheme: {scheme}")
        except Exception as e:
            print(f"[ERROR] Scheme failed: {scheme}")
            print(f"[ERROR] {repr(e)}")

    if len(all_summary) > 0:
        all_summary_df = pd.concat(all_summary, axis=0, ignore_index=True)
        all_summary_path = OUT_ROOT / "all_schemes_feature_importance_summary.csv"
        all_summary_df.to_csv(all_summary_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 80)
        print("[SUMMARY] All schemes summary saved:")
        print(all_summary_path.resolve())
    else:
        print("[WARNING] No scheme completed successfully.")


if __name__ == "__main__":
    main()