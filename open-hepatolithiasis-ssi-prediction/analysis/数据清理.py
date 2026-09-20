# -*- coding: utf-8 -*-
"""
SSI strict numeric cleaning
---------------------------
目标：
1) "/" -> NaN
2) 非法字符 -> NaN
3) 全部变量（除ID）强制转数值
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("开腹肝胆道结石手术SSI数据.csv")
OUTPUT_PATH = Path("开腹肝胆道结石手术SSI清洗.csv")

df = pd.read_csv(INPUT_PATH)

print("原始shape:", df.shape)

# =========================
# 1️⃣ 去除字符串空格
# =========================
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip()

# =========================
# 2️⃣ "/" 统一替换为 NaN
# =========================
df.replace("/", np.nan, inplace=True)

# =========================
# 3️⃣ 强制数值转换（除ID）
# =========================
for col in df.columns:
    if col == "ID":
        continue
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 4️⃣ 输出缺失统计
# =========================
missing_counts = df.isna().sum()
missing_ratio = (df.isna().mean() * 100).round(2)

missing_df = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_ratio_%": missing_ratio
}).sort_values("missing_count", ascending=False)

print("\n==== Missing Summary ====")
print(missing_df[missing_df["missing_count"] > 0])

df.to_csv(OUTPUT_PATH, index=False)

print("\n清洗完成，保存为:", OUTPUT_PATH)