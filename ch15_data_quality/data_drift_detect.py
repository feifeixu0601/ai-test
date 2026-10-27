"""
数据漂移检测
比较训练数据和线上数据的分布差异

方法：
  1. PSI（Population Stability Index）—— 分布漂移量化
  2. KS 检验（Kolmogorov-Smirnov）—— 统计显著性
  3. 分布可视化对比
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("数据漂移检测")
print("=" * 60)

np.random.seed(42)


# ========== 1. 创建"训练数据"和"线上数据" ==========

print("\n--- 1. 创建训练数据和线上数据 ---")

n = 1000

# 训练数据：以年轻人为主
train_data = pd.DataFrame({
    "age": np.concatenate([
        np.random.normal(28, 5, int(n * 0.6)),    # 60% 年轻人
        np.random.normal(45, 8, int(n * 0.4)),    # 40% 中年人
    ]),
    "income": np.random.normal(8000, 3000, n),
    "purchase_amount": np.random.gamma(2, 500, n),
})

# 线上数据：中老年用户变多了（分布偏移）
online_data = pd.DataFrame({
    "age": np.concatenate([
        np.random.normal(28, 5, int(n * 0.3)),    # 30% 年轻人（下降了）
        np.random.normal(52, 10, int(n * 0.7)),   # 70% 中老年人（上升了）
    ]),
    "income": np.random.normal(9500, 4000, n),     # 收入也变了
    "purchase_amount": np.random.gamma(3, 400, n),  # 消费模式也变了
})

# 清理异常值（避免数据科学库报错）
for df in [train_data, online_data]:
    df["age"] = df["age"].clip(15, 80)
    df["income"] = df["income"].clip(1000, 50000)
    df["purchase_amount"] = df["purchase_amount"].clip(0, 20000)

print(f"  训练数据：{len(train_data)} 条")
print(f"  线上数据：{len(online_data)} 条")
print(f"  特征列：{list(train_data.columns)}")


# ========== 2. PSI 计算 ==========

print(f"\n--- 2. PSI 分布漂移检测 ---")


def calc_psi(train_series, online_series, bins=10):
    """
    计算 PSI（Population Stability Index）

    原理：
      1. 用训练数据的分位数切分成 bins 个桶
      2. 统计训练数据和线上数据在每个桶中的比例
      3. PSI = Σ (线上比例 - 训练比例) × ln(线上比例 / 训练比例)

    参数：
        train_series: 训练数据（Series 或 array）
        online_series: 线上数据（Series 或 array）
        bins: 分桶数

    返回：
        psi: PSI 值
        bin_info: 每个桶的详细信息
    """
    # 用训练数据的分位数定义桶边界
    breakpoints = np.percentile(train_series.dropna(),
                                np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # 去重（如果有重复分位数）

    # 统计每个桶中的比例
    train_counts = np.histogram(train_series.dropna(), bins=breakpoints)[0]
    online_counts = np.histogram(online_series.dropna(), bins=breakpoints)[0]

    # 转为比例（加极小值避免 log(0)）
    train_pct = train_counts / train_counts.sum() + 1e-10
    online_pct = online_counts / online_counts.sum() + 1e-10

    # 计算 PSI
    psi = np.sum((online_pct - train_pct) * np.log(online_pct / train_pct))

    # 桶信息
    bin_info = []
    for i in range(len(breakpoints) - 1):
        bin_info.append({
            "range": f"[{breakpoints[i]:.1f}, {breakpoints[i+1]:.1f})",
            "train_pct": train_pct[i],
            "online_pct": online_pct[i],
        })

    return psi, bin_info


def judge_psi(psi):
    """根据 PSI 值给出判定"""
    if psi < 0.1:
        return "[PASS] 无显著漂移"
    elif psi < 0.25:
        return "[WARN] 轻微漂移，需要关注"
    else:
        return "[FAIL] 显著漂移，需要处理"


# 对每个特征计算 PSI
print(f"\n  {'特征':<20s} {'PSI值':>8s} {'判定':<25s}")
print(f"  {'─' * 55}")

psi_results = {}
for col in train_data.columns:
    psi, bin_info = calc_psi(train_data[col], online_data[col])
    psi_results[col] = {"psi": psi, "bin_info": bin_info}
    verdict = judge_psi(psi)
    print(f"  {col:<20s} {psi:>8.4f} {verdict:<25s}")


# ========== 3. KS 检验 ==========

print(f"\n--- 3. KS 检验（统计显著性） ---")
print(f"  KS 检验判断两个分布是否有统计上的显著差异")
print(f"  p < 0.05 → 两个分布显著不同（有漂移）")

print(f"\n  {'特征':<20s} {'KS统计量':>10s} {'p值':>12s} {'判定':>10s}")
print(f"  {'─' * 55}")

for col in train_data.columns:
    ks_stat, p_value = stats.ks_2samp(train_data[col], online_data[col])

    if p_value < 0.01:
        verdict = "[FAIL] 显著"
    elif p_value < 0.05:
        verdict = "[WARN] 边缘"
    else:
        verdict = "[PASS] 不显著"

    print(f"  {col:<20s} {ks_stat:>10.4f} {p_value:>12.6f} {verdict:>10s}")


# ========== 4. 分布可视化 ==========

print(f"\n--- 4. 分布可视化 ---")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, col in enumerate(train_data.columns):
    ax = axes[idx]

    # 训练数据分布
    ax.hist(train_data[col], bins=30, alpha=0.5, color="#2E86C1",
            label="训练数据", density=True)

    # 线上数据分布
    ax.hist(online_data[col], bins=30, alpha=0.5, color="#E74C3C",
            label="线上数据", density=True)

    psi_val = psi_results[col]["psi"]
    verdict = judge_psi(psi_val)

    ax.set_title(f"{col}\nPSI = {psi_val:.4f}  {verdict}", fontsize=11)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel("密度", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("训练数据 vs 线上数据 分布对比", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/data_drift.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  分布对比图已保存到 charts/data_drift.png")

print(f"\n  从图上能直观看到：")
print(f"    1. age 分布明显偏移——线上中老年用户变多了")
print(f"    2. income 分布也发生了变化")
print(f"    3. purchase_amount 的分布形状和峰值都变了")
print(f"    4. 蓝色（训练）和红色（线上）重叠越少，漂移越严重")
