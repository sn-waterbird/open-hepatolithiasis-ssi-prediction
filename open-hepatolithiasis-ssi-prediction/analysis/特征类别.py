# -*- coding: utf-8 -*-
"""
infer_feature_types_by_cardinality.py

规则（按你要求）：
- 排除：ID, Infection（可在 EXCLUDE_COLS 增删）
- 对每个特征：先把缺失值去掉，再算唯一值个数 n_unique
    * n_unique == 2            -> Binary
    * 3 <= n_unique <= 5       -> Categorical (multi-class)
    * 其他 (<=1 或 >5)         -> Continuous

输出：
- feature_types_summary.csv
- continuous_cols.txt
- binary_cols.txt
- categorical_cols.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# Config
# =========================
CSV_PATH = Path("开腹肝胆道结石手术SSI清洗.csv")
OUT_DIR = Path("./特征类别")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = {"ID", "Infection"}  # 如需额外排除列，在这里加

# =========================
# Load
# =========================
df = pd.read_csv(CSV_PATH)

missing_excludes = [c for c in EXCLUDE_COLS if c not in df.columns]
if missing_excludes:
    raise ValueError(f"Exclude columns not found: {missing_excludes}")

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

# =========================
# Type inference
# =========================
rows = []
binary_cols = []
categorical_cols = []
continuous_cols = []

for col in feature_cols:
    s = df[col]

    # 统一把常见“缺失占位符”当缺失（稳一点：如果你之前已经清洗成NaN，也不会有副作用）
    if s.dtype == object:
        s = s.astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan, "/": np.nan})

    s_non_missing = s.dropna()

    n_non_missing = int(s_non_missing.shape[0])
    n_missing = int(s.shape[0] - n_non_missing)

    # 注意：如果是数值列，nunique会把 1 和 1.0 当一个值；字符串也一样
    n_unique = int(pd.Series(s_non_missing).nunique(dropna=True))

    if n_unique == 2:
        ftype = "binary"
        binary_cols.append(col)
    elif 3 <= n_unique <= 5:
        ftype = "categorical"
        categorical_cols.append(col)
    else:
        ftype = "continuous"
        continuous_cols.append(col)

    # 抽样展示唯一值（最多展示前10个，便于你人工 sanity check）
    uniq_preview = pd.Series(s_non_missing.unique()).head(10).tolist()

    rows.append({
        "feature": col,
        "type": ftype,
        "n_unique_non_missing": n_unique,
        "n_missing": n_missing,
        "missing_ratio": round(n_missing / len(df), 6),
        "unique_preview_head10": uniq_preview,
    })

summary = pd.DataFrame(rows).sort_values(["type", "n_unique_non_missing", "missing_ratio"], ascending=[True, True, False])

# =========================
# Save
# =========================
summary_path = OUT_DIR / "feature_types_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

(OUT_DIR / "continuous_cols.txt").write_text("\n".join(continuous_cols), encoding="utf-8")
(OUT_DIR / "binary_cols.txt").write_text("\n".join(binary_cols), encoding="utf-8")
(OUT_DIR / "categorical_cols.txt").write_text("\n".join(categorical_cols), encoding="utf-8")

print("==== Done ====")
print("N samples:", len(df))
print("Total features (excluded ID/Infection):", len(feature_cols))
print("Continuous:", len(continuous_cols))
print("Binary:", len(binary_cols))
print("Categorical:", len(categorical_cols))
print("Saved:")
print(" -", summary_path)
print(" -", OUT_DIR / "continuous_cols.txt")
print(" -", OUT_DIR / "binary_cols.txt")
print(" -", OUT_DIR / "categorical_cols.txt")

# 打印各类前20个，方便你快速目视确认
print("\n[Binary head20]:", binary_cols[:20])
print("[Categorical head20]:", categorical_cols[:20])
print("[Continuous head20]:", continuous_cols[:20])