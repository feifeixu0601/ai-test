"""
数据质量审计工具
对训练数据做全面体检，输出质量报告

检测维度：
  1. 完整性（缺失值）
  2. 准确性（异常值、重复）
  3. 一致性（格式不统一）
  4. 均衡性（类别分布）
  5. 统计摘要（基础统计信息）
"""

import numpy as np
import pandas as pd
import random

print("=" * 60)
print("数据质量审计工具")
print("=" * 60)

random.seed(42)
np.random.seed(42)


# ========== 1. 创建模拟数据集（故意埋入质量问题）==========

print("\n--- 1. 创建模拟数据集 ---")

n = 500

# 模拟一个用户购买数据集
data = {
    "user_id": [f"U{i:04d}" for i in range(n)],
    "age": np.random.randint(18, 65, n).astype(float),
    "gender": np.random.choice(["male", "female", "other"], n, p=[0.45, 0.45, 0.10]),
    "income": np.random.randint(3000, 30000, n).astype(float),
    "purchase_amount": np.random.uniform(10, 5000, n),
    "rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.15, 0.35, 0.35]),
    "purchase_date": pd.date_range("2023-01-01", periods=n, freq="4h"),
}

df = pd.DataFrame(data)

# ===== 故意埋入质量问题 =====

# 问题1：重复数据（准确性）
# 先从干净数据中复制，确保重复行和原行完全一致
duplicate_rows = df.head(10).copy()
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 问题2：缺失值（完整性）
# 排除后面会被异常值覆盖的行，避免 age 缺失被覆盖
exclude = {5, 12, 23, 45}
pool = [i for i in range(n) if i not in exclude]

# 一次性随机抽 110 个不重复的行号
all_missing_idx = random.sample(pool, 110)

# 前 80 个用于 age 缺失
df.loc[all_missing_idx[:80], "age"] = np.nan

# 后 30 个用于 income 缺失
df.loc[all_missing_idx[80:], "income"] = np.nan

# 问题3：异常值（准确性）
df.loc[5, "age"] = 200       # 年龄 200 岁
df.loc[12, "age"] = -5       # 年龄 -5 岁
df.loc[23, "income"] = -1000  # 负收入
df.loc[45, "purchase_amount"] = 999999  # 离谱消费金额

# 问题4：类别不均衡（均衡性）
imbalance_idx = random.sample(range(300, n), 200)
df.loc[imbalance_idx, "rating"] = 5

# 问题5：格式不一致（一致性）
gender_mixed = ["male", "Male", "MALE", "female", "Female", "FEMALE", "m", "f"]
for i in range(50, 100):
    df.loc[i, "gender"] = random.choice(gender_mixed)

print(f"  数据集大小：{len(df)} 行 × {len(df.columns)} 列")
print(f"  列名：{list(df.columns)}")
print(f"  故意埋入 5 类质量问题：")
print(f"    1. 重复行（前 10 行被复制了一份）")
print(f"    2. 缺失值（age、income 有 NaN）")
print(f"    3. 异常值（age=200、age=-5、income=-1000、消费=999999）")
print(f"    4. 类别不均衡（rating=5 被大量注入）")
print(f"    5. 格式不一致（gender 有 Male/MALE/m 等多种写法）")


# ========== 2. 完整性检测 ==========

print(f"\n{'─' * 60}")
print(f"2. 完整性检测（缺失值）")
print(f"{'─' * 60}")

missing_stats = df.isnull().sum()
missing_pct = df.isnull().mean() * 100

print(f"\n  {'列名':<20s} {'缺失数':>8s} {'缺失率':>8s} {'判定':>10s}")
print(f"  {'─' * 50}")

for col in df.columns:
    miss_count = missing_stats[col]
    miss_pct = missing_pct[col]

    if miss_pct == 0:
        verdict = "[PASS]"
    elif miss_pct < 5:
        verdict = "[PASS] 轻微"
    elif miss_pct < 20:
        verdict = "[WARN] 中等"
    else:
        verdict = "[FAIL] 严重"

    print(f"  {col:<20s} {miss_count:>8d} {miss_pct:>7.1f}% {verdict:>10s}")


# ========== 3. 准确性检测 ==========

print(f"\n{'─' * 60}")
print(f"3. 准确性检测（异常值与重复）")
print(f"{'─' * 60}")

# 3a. 异常值检测（基于业务逻辑的硬规则）
print(f"\n  [3a] 业务规则异常值检测")

hard_rules = {
    "age": {"min": 0, "max": 120, "desc": "年龄应在 0~120 之间"},
    "income": {"min": 0, "max": 1000000, "desc": "收入应为正数"},
    "purchase_amount": {"min": 0, "max": 100000, "desc": "消费金额应在合理范围内"},
}

anomaly_total = 0
for col, rule in hard_rules.items():
    anomalies = df[(df[col] < rule["min"]) | (df[col] > rule["max"])]
    # 排除 NaN（NaN 的比较结果是 False，不会被选中）
    anomalies = anomalies.dropna(subset=[col])

    if len(anomalies) > 0:
        anomaly_total += len(anomalies)
        print(f"    [WARN] {col}: {len(anomalies)} 条异常（{rule['desc']}）")
        for idx, row in anomalies.iterrows():
            print(f"           行 {idx}: {col} = {row[col]}")
    else:
        print(f"    [PASS] {col}: 无异常")

print(f"\n  异常值总数：{anomaly_total}")

# 3b. 统计异常值检测（IQR 方法）
print(f"\n  [3b] IQR 统计异常值检测")
print(f"  IQR = Q3 - Q1，超出 [Q1-1.5*IQR, Q3+1.5*IQR] 的值视为异常")

numeric_cols = ["age", "income", "purchase_amount"]
for col in numeric_cols:
    col_data = df[col].dropna()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = col_data[(col_data < lower) | (col_data > upper)]
    pct = len(outliers) / len(col_data) * 100

    verdict = "[PASS]" if pct < 5 else "[WARN]"
    print(f"    {col}: Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}")
    print(f"      异常范围：[{lower:.1f}, {upper:.1f}] 之外")
    print(f"      异常数：{len(outliers)}（{pct:.1f}%）{verdict}")

# 3c. 重复数据检测
print(f"\n  [3c] 重复数据检测")
duplicate_count = df.duplicated().sum()
dup_pct = duplicate_count / len(df) * 100

if dup_pct == 0:
    print(f"    [PASS] 无重复行")
elif dup_pct < 5:
    print(f"    [WARN] 有 {duplicate_count} 行重复（{dup_pct:.1f}%）")
else:
    print(f"    [FAIL] 有 {duplicate_count} 行重复（{dup_pct:.1f}%），需要去重处理")


# ========== 4. 一致性检测 ==========

print(f"\n{'─' * 60}")
print(f"4. 一致性检测（格式不统一）")
print(f"{'─' * 60}")

# 4a. 分类变量的格式一致性
print(f"\n  [4a] 分类变量格式一致性")

for col in ["gender", "rating"]:
    unique_vals = df[col].dropna().unique()
    print(f"\n    {col} 的所有取值（{len(unique_vals)} 种）：")
    for val in sorted(unique_vals, key=str):
        count = (df[col] == val).sum()
        print(f"      '{val}'  （{count} 条）")

    # 判断是否格式不统一
    str_vals = [str(v).lower() for v in unique_vals]
    normalized_unique = set(str_vals)
    if len(normalized_unique) < len(unique_vals):
        print(f"    [WARN] 格式不统一！归一化后只有 {len(normalized_unique)} 种唯一值")
        print(f"           建议：统一为小写并做映射（如 Male/MALE/m → male）")
    else:
        print(f"    [PASS] 格式统一")

# 4b. 数值范围一致性
print(f"\n  [4b] 数值类型一致性")

for col in ["age", "income", "purchase_amount"]:
    dtype = df[col].dtype
    non_numeric = df[col].dropna().apply(
        lambda x: not isinstance(x, (int, float, np.integer, np.floating))
    )
    bad_count = non_numeric.sum()

    if bad_count > 0:
        print(f"    [WARN] {col}: 有 {bad_count} 条非数值数据混入")
    else:
        print(f"    [PASS] {col}: 全部为数值型（dtype={dtype}）")


# ========== 5. 均衡性检测 ==========

print(f"\n{'─' * 60}")
print(f"5. 均衡性检测（类别分布）")
print(f"{'─' * 60}")

for col in ["rating", "gender"]:
    print(f"\n  {col} 的类别分布：")
    value_counts = df[col].value_counts()
    total = value_counts.sum()

    for val, count in value_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    '{val}': {count:>4d} ({pct:>5.1f}%)  {bar}")

    # 判断是否均衡
    max_pct = value_counts.max() / total * 100
    min_pct = value_counts.min() / total * 100

    if max_pct > 70:
        print(f"    [FAIL] 严重不均衡！最大类别占比 {max_pct:.1f}%")
        print(f"           建议：过采样少数类 或 欠采样多数类")
    elif max_pct / max(min_pct, 0.01) > 3:
        print(f"    [WARN] 存在不均衡（最大/最小 = {max_pct/max(min_pct, 0.01):.1f}倍）")
    else:
        print(f"    [PASS] 类别分布较均衡")


# ========== 6. 质量汇总报告 ==========

print(f"\n{'=' * 60}")
print(f"数据质量审计汇总报告")
print(f"{'=' * 60}")

quality_issues = []

# 收集所有问题（按问题埋入顺序：重复 → 缺失 → 异常 → 不均衡 → 格式）

# 问题1：重复数据
if duplicate_count > 0:
    quality_issues.append({
        "维度": "准确性",
        "问题": f"有 {duplicate_count} 行重复数据（{dup_pct:.1f}%）",
        "严重度": "中" if dup_pct < 5 else "高",
    })

# 问题2：缺失值
for col in df.columns:
    miss_pct = df[col].isnull().mean() * 100
    if miss_pct > 5:
        quality_issues.append({
            "维度": "完整性",
            "问题": f"{col} 缺失率 {miss_pct:.1f}%",
            "严重度": "高" if miss_pct > 20 else "中",
        })

# 问题3：异常值
for col, rule in hard_rules.items():
    anomalies = df[(df[col] < rule["min"]) | (df[col] > rule["max"])]
    anomalies = anomalies.dropna(subset=[col])
    if len(anomalies) > 0:
        quality_issues.append({
            "维度": "准确性",
            "问题": f"{col} 有 {len(anomalies)} 条业务规则异常",
            "严重度": "高",
        })

# 问题4：类别不均衡
rating_counts = df["rating"].value_counts()
if rating_counts.max() / len(df) > 0.5:
    quality_issues.append({
        "维度": "均衡性",
        "问题": f"rating 类别不均衡（最大类别占比 {rating_counts.max()/len(df)*100:.1f}%）",
        "严重度": "高",
    })

# 问题5：格式不一致
gender_lower_unique = len(set(str(v).lower() for v in df["gender"].dropna().unique()))
gender_raw_unique = len(df["gender"].dropna().unique())
if gender_lower_unique < gender_raw_unique:
    quality_issues.append({
        "维度": "一致性",
        "问题": f"gender 格式不统一（{gender_raw_unique} 种写法 → 归一化后 {gender_lower_unique} 种）",
        "严重度": "中",
    })

# 输出
print(f"\n  发现问题总数：{len(quality_issues)}")
print(f"\n  {'维度':<10s} {'严重度':>6s}  问题描述")
print(f"  {'─' * 55}")

for issue in quality_issues:
    severity = issue["严重度"]
    marker = "[FAIL]" if severity == "高" else "[WARN]"
    print(f"  {issue['维度']:<10s} {marker} {severity}  {issue['问题']}")

# 总体判定
high_count = sum(1 for i in quality_issues if i["严重度"] == "高")
med_count = sum(1 for i in quality_issues if i["严重度"] == "中")

print(f"\n  总体判定：")
if high_count == 0 and med_count == 0:
    print(f"    [PASS] 数据质量良好，可用于模型训练")
elif high_count == 0:
    print(f"    [WARN] 有 {med_count} 个中等问题，建议修复后再训练")
else:
    print(f"    [FAIL] 有 {high_count} 个严重问题、{med_count} 个中等问题")
    print(f"           不建议直接用于模型训练，需要先修复数据")

print(f"\n  修复建议（按问题顺序）：")
print(f"    1. 重复行：直接 drop_duplicates() 去重")
print(f"    2. 缺失值：根据缺失率选择删除或填充（均值/众数/模型插补）")
print(f"    3. 异常值：人工审核确认是数据错误还是真实极端值")
print(f"    4. 类别不均衡：过采样(SMOTE)、欠采样、或调整类别权重")
print(f"    5. 格式不一致：做归一化映射（统一大小写、映射别名）")
