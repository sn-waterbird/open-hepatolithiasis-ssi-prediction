# -*- coding: utf-8 -*-
"""
特征选择消融.py

作用：
  证明 MCFS 的"三维共识"优于任一单维度特征选择策略。
  比较 5 种策略在 K=4,6,8 下的 CatBoost+S0 表现。

策略说明：
  1) PI-only    — 仅按排列重要性（PI_norm）排序选 TopK
  2) PVC-only   — 仅按预测值变化（Internal1_norm）排序选 TopK
  3) LFC-only   — 仅按损失函数变化（Internal2_norm）排序选 TopK
  4) Consensus  — 三维共识几何平均（当前 MCFS 完整流程）
  5) SkipCV     — 跳过 CV 稳定性过滤，直接用全部特征的共识排序选 TopK

设计：
  - 模型：CatBoost+S0（已确认为最优组合）
  - 参数：固定（从 训练全特征/best_config.json 读取），控制变量
  - 评价：5 折 CV → DEV OOF → Youden 阈值 → INDEP 评估
  - K 值：4, 6, 8

输出：
  消融实验/
    ├── ablation_comparison.csv          # 5策略 × 3K值 完整对比
    ├── ablation_comparison_summary.csv   # 精简对比表
    └── {Strategy}_Top{K}/               # 每个策略×K值的详细输出
"""

import os, json, random, warnings, re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, roc_curve, precision_recall_curve, confusion_matrix,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# =========================
# Fixed Config
# =========================
SEED = 42
N_FOLDS = 5
N_THRESH = 201

DATA_CSV = Path("开腹肝胆道结石手术SSI清洗.csv")
SPLIT_CSV = Path("splits/split_folds_full.csv")
FEATURE_TYPES_CSV = Path("特征类别/feature_types_summary.csv")
EXCLUDE_COLS = ["ID", "Infection", "Infection_split"]

TARGET_SCHEME = "CatBoost+S0"
SCHEME_TYPE = "scheme2"
CONTRIBUTION_CSV = Path(f"特征筛选共识/{TARGET_SCHEME}/contribution_{SCHEME_TYPE}.csv")
SELECTED_FEATURES_CSV = Path(f"特征筛选共识/{TARGET_SCHEME}/selected_features_{SCHEME_TYPE}.csv")
BEST_CONFIG_JSON = Path("训练全特征/best_config.json")

OUT_DIR = Path("消融实验")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed hyperparams (from best_config.json for CatBoost+S0)
FIXED_PARAMS = {
    "depth": 3,
    "learning_rate": 0.03,
    "iterations": 600,
}

K_VALUES = [2, 4, 6]

# =========================
# Reproducibility
# =========================
def fix_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# =========================
# Feature Type Helpers
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
    if continuous_cols:
        cont_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True))
        ])
        transformers.append(("cont", cont_pipe, continuous_cols))
    if binary_cols:
        bin_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent"))
        ])
        transformers.append(("bin", bin_pipe, binary_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("to_str", FunctionTransformer(lambda x: x.astype(str), feature_names_out="one-to-one")),
            ("ohe", make_ohe())
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def build_estimator(params: Dict[str, Any], seed: int):
    return CatBoostClassifier(
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        thread_count=-1,
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        iterations=int(params["iterations"]),
        l2_leaf_reg=3.0,
    )


def build_pipeline(preprocessor: ColumnTransformer, estimator):
    return Pipeline([("prep", preprocessor), ("clf", estimator)])


# =========================
# Metrics Helpers
# =========================
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


def confusion_metrics_from_preds(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv == ppv and sens == sens and (ppv + sens) > 0) else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    bal_acc = (sens + spec) / 2 if (sens == sens and spec == spec) else np.nan
    youden = sens + spec - 1 if (sens == sens and spec == spec) else np.nan
    return {
        "TP": float(tp), "FP": float(fp), "TN": float(tn), "FN": float(fn),
        "Sensitivity": float(sens) if sens == sens else np.nan,
        "Specificity": float(spec) if spec == spec else np.nan,
        "PPV": float(ppv) if ppv == ppv else np.nan,
        "NPV": float(npv) if npv == npv else np.nan,
        "F1": float(f1) if f1 == f1 else np.nan,
        "Accuracy": float(acc) if acc == acc else np.nan,
        "BalancedAcc": float(bal_acc) if bal_acc == bal_acc else np.nan,
        "YoudenJ": float(youden) if youden == youden else np.nan,
    }


def find_best_threshold_youden(y_true: np.ndarray, p: np.ndarray, n_thresh: int = 201) -> Tuple[float, pd.DataFrame]:
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
    return best_thr, scan_df


def detect_indep_split(sp: pd.DataFrame) -> str:
    splits = sp["split"].astype(str).str.upper().unique().tolist()
    for cand in ["VAL", "INDEP_TEST", "TEST"]:
        if cand in splits:
            return cand
    raise ValueError(f"Cannot find independent split among {splits}")


def safe_float(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


# =========================
# Get TopK Features for Each Strategy
# =========================
def _minmax(x: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-15:
        return pd.Series(0.5, index=x.index)
    return (x - lo) / (hi - lo)


def get_topk_features_for_strategy(
    strategy: str,
    k: int,
    contrib_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> List[str]:
    """
    Return the TopK feature list for a given ablation strategy.

    Parameters
    ----------
    strategy : str
        One of ['PI-only', 'PVC-only', 'LFC-only', 'Consensus', 'SkipCV'].
    k : int
        Number of features to select.
    contrib_df : pd.DataFrame
        Contribution file (CV-passed features only).
    selected_df : pd.DataFrame
        Selected features file (all features, with keep_scheme2 flag).
    """
    if strategy == "PI-only":
        col = "PI_norm"
        df = contrib_df.copy()
        df = df.sort_values(col, ascending=False)
        return df["feature"].head(k).tolist()

    elif strategy == "PVC-only":
        col = "Internal1_norm"
        df = contrib_df.copy()
        df = df.sort_values(col, ascending=False)
        return df["feature"].head(k).tolist()

    elif strategy == "LFC-only":
        col = "Internal2_norm"
        df = contrib_df.copy()
        df = df.sort_values(col, ascending=False)
        return df["feature"].head(k).tolist()

    elif strategy == "Consensus":
        # Full MCFS: use the Rank column from contribution file
        df = contrib_df.copy()
        df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
        df = df.sort_values("Rank")
        return df["feature"].head(k).tolist()

    elif strategy == "SkipCV":
        # Skip CV filter: use ALL features, normalize raw values, compute consensus
        df = selected_df.copy()
        # Only use features with non-NaN mean values
        df = df.dropna(subset=["PI_mean"])
        if df.empty:
            raise ValueError("selected_df is empty after dropping NaN PI_mean")

        # Normalize each dimension across ALL features
        pi_raw = pd.to_numeric(df["PI_mean"], errors="coerce").fillna(0)
        pvc_raw = pd.to_numeric(df["PVC_mean"], errors="coerce").fillna(0)
        lfc_raw = pd.to_numeric(df["LFC_mean"], errors="coerce").fillna(0)

        # Handle negative LFC: shift so min becomes 0
        # But min-max normalization handles negatives naturally
        pi_norm = _minmax(pi_raw)
        pvc_norm = _minmax(pvc_raw)
        lfc_norm = _minmax(lfc_raw)

        # Clamp to [eps, 1] for geometric mean
        eps = 1e-15
        pi_norm = pi_norm.clip(lower=eps)
        pvc_norm = pvc_norm.clip(lower=eps)
        lfc_norm = lfc_norm.clip(lower=eps)

        # Consensus = geometric mean
        consensus = (pi_norm * pvc_norm * lfc_norm) ** (1.0 / 3.0)
        df = df.assign(Consensus_skipcv=consensus.values)
        df = df.sort_values("Consensus_skipcv", ascending=False)
        return df["feature"].head(k).tolist()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# =========================
# Evaluate One Feature Set
# =========================
def evaluate_feature_set(
    features: List[str],
    preprocessor: ColumnTransformer,
    df_idx: pd.DataFrame,
    sp_dev: pd.DataFrame,
    sp_indep: pd.DataFrame,
    indep_split: str,
    y_col: str,
    folds: List[int],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Evaluate CatBoost+S0 with fixed params on the given feature set.
    Returns summary metrics dict.
    """
    n_feat = len(features)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: DEV OOF with fixed params + Youden threshold ---
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

        est = build_estimator(FIXED_PARAMS, SEED)
        pipe = build_pipeline(preprocessor, est)
        pipe.fit(Xtr, ytr)
        pva = predict_proba_pos(pipe, Xva)
        for _id, yt, pr, fid in zip(va["ID"].values.tolist(), yva.tolist(), pva.tolist(), [int(f)] * len(yva)):
            oof_rows.append({"ID": _id, "y_true": int(yt), "oof_prob": float(pr), "fold_id": fid})

    dev_oof = pd.DataFrame(oof_rows).sort_values(["fold_id", "ID"]).reset_index(drop=True)
    dev_oof.to_csv(out_dir / "dev_oof_predictions.csv", index=False, encoding="utf-8-sig")

    dev_oof_basic = safe_metrics(dev_oof["y_true"].values, dev_oof["oof_prob"].values)
    best_thr, scan_df = find_best_threshold_youden(
        dev_oof["y_true"].values, dev_oof["oof_prob"].values, n_thresh=N_THRESH,
    )
    scan_df.to_csv(out_dir / "dev_oof_metrics_threshold_scan.csv", index=False, encoding="utf-8-sig")

    dev_oof_hat = (dev_oof["oof_prob"].values >= best_thr).astype(int)
    dev_oof_cm = confusion_metrics_from_preds(dev_oof["y_true"].values, dev_oof_hat)

    # --- Step 2: Train on full DEV -> INDEP ---
    dev_all = df_idx.loc[sp_dev["ID"].tolist()]
    indep_all = df_idx.loc[sp_indep["ID"].tolist()]
    Xtr_full = dev_all[features]
    ytr_full = dev_all[y_col].values.astype(int)
    Xte = indep_all[features]
    yte = indep_all[y_col].values.astype(int)

    est_final = build_estimator(FIXED_PARAMS, SEED)
    pipe_final = build_pipeline(preprocessor, est_final)
    pipe_final.fit(Xtr_full, ytr_full)
    p_te = predict_proba_pos(pipe_final, Xte)

    indep_basic = safe_metrics(yte, p_te)
    y_hat_te = (p_te >= best_thr).astype(int)
    indep_cm = confusion_metrics_from_preds(yte, y_hat_te)

    # Save INDEP outputs
    indep_pred = pd.DataFrame({
        "ID": indep_all["ID"].values,
        "y_true": yte.astype(int),
        "pred_prob": p_te.astype(float),
        "pred_label": y_hat_te.astype(int),
        "threshold_used": float(best_thr),
    }).sort_values("ID").reset_index(drop=True)
    indep_pred.to_csv(out_dir / "indep_predictions.csv", index=False, encoding="utf-8-sig")

    cm_tbl = pd.DataFrame([{
        "TN": indep_cm["TN"], "FP": indep_cm["FP"],
        "FN": indep_cm["FN"], "TP": indep_cm["TP"],
    }])
    cm_tbl.to_csv(out_dir / "indep_confusion_matrix.csv", index=False, encoding="utf-8-sig")

    # Report table
    report_rows = []
    for ds_name, y_ds, p_ds, cm_ds in [
        ("DEV_OOF", dev_oof["y_true"].values, dev_oof["oof_prob"].values, dev_oof_cm),
        (indep_split, yte, p_te, indep_cm),
    ]:
        basic = safe_metrics(y_ds, p_ds) if ds_name == "DEV_OOF" else indep_basic
        report_rows.append({
            "Dataset": ds_name,
            "n_total": int(len(y_ds)), "n_pos": int(y_ds.sum()),
            "n_neg": int((y_ds == 0).sum()),
            "AUROC": float(basic["AUROC"]), "AUPRC": float(basic["AUPRC"]),
            "Brier": float(basic["Brier"]), "LogLoss": float(basic["LogLoss"]),
            "Threshold": float(best_thr),
            "Sensitivity": float(cm_ds["Sensitivity"]),
            "Specificity": float(cm_ds["Specificity"]),
            "PPV": float(cm_ds["PPV"]), "NPV": float(cm_ds["NPV"]),
            "F1": float(cm_ds["F1"]), "Accuracy": float(cm_ds["Accuracy"]),
            "BalancedAcc": float(cm_ds["BalancedAcc"]),
            "YoudenJ": float(cm_ds["YoudenJ"]),
        })
    report = pd.DataFrame(report_rows)
    metric_cols = [
        "AUROC", "AUPRC", "Brier", "LogLoss", "Threshold",
        "Sensitivity", "Specificity", "PPV", "NPV", "F1",
        "Accuracy", "BalancedAcc", "YoudenJ",
    ]
    for c in metric_cols:
        if c in report.columns:
            report[c] = report[c].astype(float).round(4)
    report.to_csv(out_dir / "report_metrics_table.csv", index=False, encoding="utf-8-sig")

    # ROC/PR curves
    roc_fpr, roc_tpr, _ = roc_curve(dev_oof["y_true"].values, dev_oof["oof_prob"].values)
    pr_prec, pr_rec, _ = precision_recall_curve(dev_oof["y_true"].values, dev_oof["oof_prob"].values)
    curves = pd.DataFrame({
        "dev_roc_fpr": pd.Series(roc_fpr),
        "dev_roc_tpr": pd.Series(roc_tpr),
        "dev_pr_precision": pd.Series(pr_prec),
        "dev_pr_recall": pd.Series(pr_rec),
    })
    curves.to_csv(out_dir / "report_curves_roc_pr.csv", index=False, encoding="utf-8-sig")

    return {
        "n_features": n_feat,
        "Features": "; ".join(features),
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


# =========================
# Main
# =========================
def main():
    fix_seed(SEED)
    print("=" * 70)
    print("  特征选择消融实验 — 证明三维共识优于单维度")
    print(f"  模型: CatBoost+S0 | 参数: 固定 (depth=3, lr=0.03, iter=600)")
    print(f"  K 值: {K_VALUES}")
    print("=" * 70)

    # 1. Load fixed params from best_config.json (validate)
    if BEST_CONFIG_JSON.exists():
        with open(BEST_CONFIG_JSON, "r") as f:
            cfg = json.load(f)
        print(f"\n[确认] 固定参数来源: {BEST_CONFIG_JSON}")
        print(f"  depth={cfg['params']['depth']}, lr={cfg['params']['learning_rate']}, "
              f"iter={cfg['params']['iterations']}")
    else:
        print(f"\n[警告] 未找到 {BEST_CONFIG_JSON}，使用默认固定参数")

    # 2. Load contribution + selected features
    print(f"\n[1/5] 加载贡献度文件 & 特征选择文件...")
    if not CONTRIBUTION_CSV.exists():
        raise FileNotFoundError(f"Missing: {CONTRIBUTION_CSV}")
    contrib_df = pd.read_csv(CONTRIBUTION_CSV)
    print(f"  贡献度文件 (CV过滤后): {len(contrib_df)} 个特征")

    if not SELECTED_FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing: {SELECTED_FEATURES_CSV}")
    selected_df = pd.read_csv(SELECTED_FEATURES_CSV)
    print(f"  特征选择文件 (全部): {len(selected_df)} 个特征")

    # 3. Load data + split
    print(f"\n[2/5] 加载数据...")
    df = pd.read_csv(DATA_CSV)
    sp = pd.read_csv(SPLIT_CSV)
    indep_split = detect_indep_split(sp)

    df = df.merge(sp[["ID", "split", "Infection"]], on="ID", how="inner",
                  suffixes=("", "_split"), validate="one_to_one")
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

    # 4. Print TopK features for each strategy
    print(f"\n[3/5] 各策略 TopK 特征预览:")
    strategies = ["PI-only", "PVC-only", "LFC-only", "Consensus", "SkipCV"]
    strategy_features_cache = {}  # (strategy, k) -> features list

    for strategy in strategies:
        for k in K_VALUES:
            feats = get_topk_features_for_strategy(strategy, k, contrib_df, selected_df)
            strategy_features_cache[(strategy, k)] = feats
            print(f"  [{strategy}] K={k}: {feats}")

    # 5. Feature types (restricted to union of all features used)
    exclude_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    all_used_features = set()
    for feats in strategy_features_cache.values():
        all_used_features.update(feats)
    all_used_features = list(all_used_features)

    cont_cols, bin_cols, cat_cols = load_feature_types(
        FEATURE_TYPES_CSV, df.columns.tolist(), exclude=exclude_cols,
    )
    cont_cols = [c for c in cont_cols if c in all_used_features]
    bin_cols = [c for c in bin_cols if c in all_used_features]
    cat_cols = [c for c in cat_cols if c in all_used_features]
    print(f"\n[4/5] 特征类型 (限于消融用到的全部特征, n={len(all_used_features)}): "
          f"连续={len(cont_cols)} | 二值={len(bin_cols)} | 类别={len(cat_cols)}")

    preprocessor = build_preprocessor(cont_cols, bin_cols, cat_cols)
    df_idx = df.set_index("ID", drop=False)

    # 6. Evaluate each strategy × K
    print(f"\n[5/5] 运行消融评价 ({len(strategies)} 策略 × {len(K_VALUES)} K值)...")
    all_results = []

    for strategy in strategies:
        for k in K_VALUES:
            features = strategy_features_cache[(strategy, k)]
            scheme_name = f"{strategy}_Top{k}"
            scheme_out_dir = OUT_DIR / scheme_name

            print(f"\n{'=' * 60}")
            print(f"[{scheme_name}] 开始评价 ({len(features)} 特征)...")
            print(f"{'=' * 60}")

            # Build preprocessor restricted to this feature set
            feat_set = set(features)
            pre_cont = [c for c in cont_cols if c in feat_set]
            pre_bin = [c for c in bin_cols if c in feat_set]
            pre_cat = [c for c in cat_cols if c in feat_set]
            pre_local = build_preprocessor(pre_cont, pre_bin, pre_cat)

            result = evaluate_feature_set(
                features=features,
                preprocessor=pre_local,
                df_idx=df_idx,
                sp_dev=sp_dev,
                sp_indep=sp_indep,
                indep_split=indep_split,
                y_col=y_col,
                folds=folds,
                out_dir=scheme_out_dir,
            )
            result["Strategy"] = strategy
            result["K"] = k
            all_results.append(result)

    # 7. Build comparison tables
    print(f"\n{'=' * 70}")
    print(f"  生成对比表...")
    print(f"{'=' * 70}")

    summary_df = pd.DataFrame(all_results)

    # Reorder columns for readability
    col_order = ["Strategy", "K", "n_features", "Features",
                 "DEV_AUPRC", "DEV_AUROC", "DEV_Brier", "DEV_LogLoss",
                 "DEV_Threshold", "DEV_Sensitivity", "DEV_Specificity",
                 "DEV_PPV", "DEV_NPV", "DEV_F1", "DEV_YoudenJ",
                 "INDEP_AUPRC", "INDEP_AUROC", "INDEP_Brier", "INDEP_LogLoss",
                 "INDEP_Sensitivity", "INDEP_Specificity", "INDEP_PPV",
                 "INDEP_NPV", "INDEP_F1", "INDEP_YoudenJ"]
    available_cols = [c for c in col_order if c in summary_df.columns]
    summary_df = summary_df[available_cols]

    # Sort by Strategy, K
    strategy_order = {s: i for i, s in enumerate(strategies)}
    summary_df["_strategy_order"] = summary_df["Strategy"].map(strategy_order)
    summary_df = summary_df.sort_values(["_strategy_order", "K"]).drop(columns=["_strategy_order"])
    summary_df = summary_df.reset_index(drop=True)

    # Save full comparison
    comp_path = OUT_DIR / "ablation_comparison.csv"
    summary_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
    print(f"\n  [保存] {comp_path}")

    # Save summary (compact version for paper)
    summary_paper = summary_df[["Strategy", "K", "n_features",
                                "DEV_AUPRC", "DEV_AUROC",
                                "INDEP_AUPRC", "INDEP_AUROC"]].copy()
    for c in ["DEV_AUPRC", "DEV_AUROC", "INDEP_AUPRC", "INDEP_AUROC"]:
        if c in summary_paper.columns:
            summary_paper[c] = summary_paper[c].astype(float).round(4)
    summary_path = OUT_DIR / "ablation_comparison_summary.csv"
    summary_paper.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] {summary_path}")

    # Print
    print(f"\n{'=' * 70}")
    print(f"  消融实验 — 核心结果 (TopK策略 × K值)")
    print(f"{'=' * 70}")
    print(summary_paper.to_string(index=False))

    # Print key findings
    print(f"\n{'=' * 70}")
    print(f"  关键发现:")
    print(f"{'=' * 70}")
    for k in K_VALUES:
        subset = summary_df[summary_df["K"] == k].copy()
        if subset.empty:
            continue
        best_dev = subset.loc[subset["DEV_AUPRC"].idxmax()]
        best_indep = subset.loc[subset["INDEP_AUPRC"].idxmax()]
        consensus_row = subset[subset["Strategy"] == "Consensus"]
        if not consensus_row.empty:
            c_dev = consensus_row["DEV_AUPRC"].values[0]
            c_indep = consensus_row["INDEP_AUPRC"].values[0]
            print(f"\n  K={k}:")
            print(f"    DEV AUPRC 最优: {best_dev['Strategy']} ({best_dev['DEV_AUPRC']:.4f})")
            print(f"    INDEP AUPRC 最优: {best_indep['Strategy']} ({best_indep['INDEP_AUPRC']:.4f})")
            print(f"    Consensus DEV AUPRC: {c_dev:.4f}")
            print(f"    Consensus INDEP AUPRC: {c_indep:.4f}")

    print(f"\n{'=' * 70}")
    print(f"  全部完成！输出目录: {OUT_DIR.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
