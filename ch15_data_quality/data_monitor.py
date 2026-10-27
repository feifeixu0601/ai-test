"""
数据质量监控流水线
将数据质量审计 + 漂移检测组合成完整流程

功能：
  1. 对新数据做质量审计（缺失、异常、重复、格式）
  2. 和基准数据对比，检测分布漂移
  3. 输出综合监控报告
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

print("=" * 60)
print("数据质量监控流水线")
print("=" * 60)

np.random.seed(42)


# ========== 1. 数据质量检测模块 ==========

class DataQualChecker:
    """数据质量检测器"""

    def __init__(self, df):
        self.df = df
        self.issues = []

    def check_missing(self, threshold=3):
        """完整性检测"""
        for col in self.df.columns:
            miss_pct = self.df[col].isnull().mean() * 100
            if miss_pct > threshold:
                self.issues.append({
                    "维度": "完整性",
                    "列": col,
                    "问题": f"缺失率 {miss_pct:.1f}%（阈值 {threshold}%）",
                    "严重度": "高" if miss_pct > 20 else "中",
                })

    def check_outliers(self, rules):
        """准确性检测（业务规则）"""
        for col, rule in rules.items():
            if col not in self.df.columns:
                continue
            anomalies = self.df[
                (self.df[col] < rule["min"]) | (self.df[col] > rule["max"])
            ].dropna(subset=[col])

            if len(anomalies) > 0:
                self.issues.append({
                    "维度": "准确性",
                    "列": col,
                    "问题": f"{len(anomalies)} 条超出 [{rule['min']}, {rule['max']}]",
                    "严重度": "高",
                })

    def check_duplicates(self):
        """重复数据检测"""
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            dup_pct = dup_count / len(self.df) * 100
            self.issues.append({
                "维度": "准确性",
                "列": "全表",
                "问题": f"{dup_count} 行重复（{dup_pct:.1f}%）",
                "严重度": "中" if dup_pct < 5 else "高",
            })

    def check_balance(self, col, threshold=70):
        """均衡性检测"""
        if col not in self.df.columns:
            return
        counts = self.df[col].value_counts()
        total = counts.sum()
        max_pct = counts.max() / total * 100

        if max_pct > threshold:
            self.issues.append({
                "维度": "均衡性",
                "列": col,
                "问题": f"最大类别占比 {max_pct:.1f}%（阈值 {threshold}%）",
                "严重度": "高" if max_pct > 90 else "中",
            })

    def run_all(self, outlier_rules=None, balance_col=None):
        """执行所有检测"""
        self.check_missing()
        if outlier_rules:
            self.check_outliers(outlier_rules)
        self.check_duplicates()
        if balance_col:
            self.check_balance(balance_col)
        return self.issues


# ========== 2. 漂移检测模块 ==========

class DriftDetector:
    """数据漂移检测器"""

    def __init__(self, baseline_df):
        self.baseline = baseline_df

    def calc_psi(self, train_series, online_series, bins=10):
        """计算 PSI"""
        breakpoints = np.percentile(
            train_series.dropna(), np.linspace(0, 100, bins + 1)
        )
        breakpoints = np.unique(breakpoints)

        train_counts = np.histogram(train_series.dropna(), bins=breakpoints)[0]
        online_counts = np.histogram(online_series.dropna(), bins=breakpoints)[0]

        train_pct = train_counts / train_counts.sum() + 1e-10
        online_pct = online_counts / online_counts.sum() + 1e-10

        psi = np.sum((online_pct - train_pct) * np.log(online_pct / train_pct))
        return psi

    def detect(self, new_df, threshold=0.25):
        """检测漂移"""
        drift_results = []

        common_cols = set(self.baseline.columns) & set(new_df.columns)
        for col in sorted(common_cols):
            if not np.issubdtype(self.baseline[col].dtype, np.number):
                continue

            psi = self.calc_psi(self.baseline[col], new_df[col])
            ks_stat, p_value = stats.ks_2samp(
                self.baseline[col].dropna(), new_df[col].dropna()
            )

            if psi > threshold:
                severity = "高"
            elif psi > 0.1:
                severity = "中"
            else:
                severity = "低"

            drift_results.append({
                "特征": col,
                "PSI": round(psi, 4),
                "KS统计量": round(ks_stat, 4),
                "p值": round(p_value, 6),
                "漂移程度": severity,
            })

        return drift_results


# ========== 3. 综合报告模块 ==========

def print_monitor_report(quality_issues, drift_results):
    """打印监控报告"""

    print(f"\n{'=' * 60}")
    print(f"数据质量监控报告")
    print(f"{'=' * 60}")
    print(f"  报告时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 质量问题
    print(f"\n{'─' * 60}")
    print(f"  一、数据质量检测")
    print(f"{'─' * 60}")

    if not quality_issues:
        print(f"  [PASS] 未发现数据质量问题")
    else:
        print(f"\n  发现 {len(quality_issues)} 个问题：")
        for issue in quality_issues:
            marker = "[FAIL]" if issue["严重度"] == "高" else "[WARN]"
            print(f"    {marker} [{issue['维度']}] {issue['列']}：{issue['问题']}")

    # 漂移检测
    print(f"\n{'─' * 60}")
    print(f"  二、数据漂移检测")
    print(f"{'─' * 60}")

    if not drift_results:
        print(f"  [PASS] 无漂移检测结果")
    else:
        print(f"\n  {'特征':<20s} {'PSI':>8s} {'KS值':>8s} {'p值':>10s} {'程度':>6s}")
        print(f"  {'─' * 55}")
        for r in drift_results:
            if r["漂移程度"] == "高":
                marker = "[FAIL]"
            elif r["漂移程度"] == "中":
                marker = "[WARN]"
            else:
                marker = "[PASS]"
            print(f"  {r['特征']:<20s} {r['PSI']:>8.4f} {r['KS统计量']:>8.4f} "
                  f"{r['p值']:>10.6f} {marker:>6s}")

    # 总体判定
    print(f"\n{'─' * 60}")
    print(f"  三、总体判定")
    print(f"{'─' * 60}")

    quality_high = sum(1 for i in quality_issues if i["严重度"] == "高")
    drift_high = sum(1 for r in drift_results if r["漂移程度"] == "高")

    if quality_high == 0 and drift_high == 0:
        print(f"  [PASS] 数据质量良好，无显著漂移")
        print(f"         可以正常使用/继续使用当前模型")
    elif quality_high > 0 and drift_high > 0:
        print(f"  [FAIL] 数据质量有严重问题 + 存在显著漂移")
        print(f"         建议：修复数据质量问题 + 重新训练模型")
    elif quality_high > 0:
        print(f"  [FAIL] 数据质量有严重问题")
        print(f"         建议：先修复数据再训练")
    else:
        print(f"  [WARN] 存在显著漂移")
        print(f"         建议：评估是否需要重新训练模型")


# ========== 4. 主程序 ==========

if __name__ == "__main__":
    # ---- 构造演示数据 ----

    n = 500

    # 基准数据（训练数据）
    baseline_df = pd.DataFrame({
        "age": np.random.normal(30, 8, n).clip(18, 70),
        "income": np.random.normal(8000, 2500, n).clip(1000, 50000),
        "purchase_amount": np.random.gamma(2, 400, n).clip(0, 20000),
        "rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.15, 0.35, 0.35]),
    })

    # 新数据（线上数据，故意有漂移和质量问题）
    new_df = pd.DataFrame({
        "age": np.random.normal(42, 12, n).clip(18, 80),
        "income": np.random.normal(10000, 3500, n).clip(1000, 60000),
        "purchase_amount": np.random.gamma(3, 350, n).clip(0, 25000),
        "rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.02, 0.03, 0.10, 0.25, 0.60]),
    })

    # 埋入质量问题
    new_df.loc[0:20, "age"] = np.nan  # 缺失值
    new_df.loc[5, "income"] = -5000   # 异常值

    print(f"\n  基准数据（训练集）：{len(baseline_df)} 条")
    print(f"  新数据（线上数据）：{len(new_df)} 条")

    # ---- 执行质量检测 ----
    print(f"\n  [Step 1] 执行数据质量检测...")
    checker = DataQualChecker(new_df)
    quality_issues = checker.run_all(
        outlier_rules={
            "age": {"min": 0, "max": 120},
            "income": {"min": 0, "max": 100000},
            "purchase_amount": {"min": 0, "max": 50000},
        },
        balance_col="rating",
    )

    # ---- 执行漂移检测 ----
    print(f"  [Step 2] 执行数据漂移检测...")
    detector = DriftDetector(baseline_df)
    drift_results = detector.detect(new_df)

    # ---- 输出报告 ----
    print_monitor_report(quality_issues, drift_results)
