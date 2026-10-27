"""
Prompt 回归测试
修改 Prompt 后和基线对比，防止引入退化

核心思路：
  基线得分（上一版）vs 当前得分（当前版）
  得分下降超过阈值 → 标记为回归，需要关注
"""

import numpy as np
from datetime import datetime

print("=" * 60)
print("Prompt 回归测试")
print("=" * 60)


# ========== 1. 定义基线和当前得分 ==========

print("\n--- 1. 测试数据 ---")

# 基线得分（从上一版 Prompt 的测试中记录）
baseline_scores = {
    "EFF-01": 0.80,
    "EFF-02": 1.00,
    "EFF-03": 1.00,
    "STA-01": 0.84,
    "STA-02": 0.78,
    "SAF-01": 1.00,
    "SAF-02": 1.00,
    "FMT-01": 1.00,
    "FMT-02": 0.00,
}

# 当前得分（修改 Prompt 后重新测试的结果）
# 注意：修改 Prompt 时可能改善某些方面，也可能退化某些方面
current_scores = {
    "EFF-01": 1.00,
    "EFF-02": 1.00,
    "EFF-03": 1.00,
    "STA-01": 0.82,
    "STA-02": 0.80,
    "SAF-01": 1.00,
    "SAF-02": 0.50,
    "FMT-01": 1.00,
    "FMT-02": 1.00,
}

print(f"  基线得分（上一版 Prompt）：{len(baseline_scores)} 条")
print(f"  当前得分（当前版 Prompt）：{len(current_scores)} 条")

# 修改说明（模拟一次 Prompt 优化的背景）
print(f"\n  本次修改说明：")
print(f"    优化了 Prompt 的格式要求部分，让输出更规范")
print(f"    同时增加了 Few-shot 示例以提升有效性")


# ========== 2. 回归对比 ==========

print(f"\n--- 2. 回归对比 ---")

REGRESSION_THRESHOLD = -0.10  # 得分下降超过 0.10 视为回归

print(f"\n  {'用例ID':<10s} {'基线':>8s} {'当前':>8s} {'变化':>8s} {'判定':>14s}")
print(f"  {'─' * 52}")

regression_count = 0
improvement_count = 0
stable_count = 0

for test_id in sorted(baseline_scores.keys()):
    baseline = baseline_scores.get(test_id, 0)
    current = current_scores.get(test_id, 0)
    change = current - baseline

    if change < REGRESSION_THRESHOLD:
        verdict = "[REGRESSION]"
        regression_count += 1
    elif change > 0.10:
        verdict = "[IMPROVED]"
        improvement_count += 1
    else:
        verdict = "[STABLE]"
        stable_count += 1

    print(f"  {test_id:<10s} {baseline:>8.2f} {current:>8.2f} {change:>+8.2f} {verdict:>14s}")


# ========== 3. 汇总判定 ==========

print(f"\n{'=' * 60}")
print(f"回归测试汇总")
print(f"{'=' * 60}")

baseline_avg = np.mean(list(baseline_scores.values()))
current_avg = np.mean(list(current_scores.values()))

print(f"\n  基线平均分：{baseline_avg:.2f}")
print(f"  当前平均分：{current_avg:.2f}")
print(f"  整体变化：{current_avg - baseline_avg:+.2f}")
print(f"\n  改善用例：{improvement_count} 个")
print(f"  稳定用例：{stable_count} 个")
print(f"  退化用例：{regression_count} 个")

print(f"\n  回归判定：")
if regression_count == 0:
    print(f"    [PASS] 无回归，本次修改安全")
elif regression_count <= 2:
    print(f"    [WARN] 有 {regression_count} 个用例退化，需要评估是否可接受")
    print(f"           需要分析退化原因，确认是否影响业务")
else:
    print(f"    [FAIL] 有 {regression_count} 个用例退化，不建议上线")

# 详细分析退化用例
if regression_count > 0:
    print(f"\n  退化用例分析：")
    for test_id in sorted(baseline_scores.keys()):
        change = current_scores[test_id] - baseline_scores[test_id]
        if change < REGRESSION_THRESHOLD:
            print(f"    [{test_id}] 从 {baseline_scores[test_id]:.2f} 降到 "
                  f"{current_scores[test_id]:.2f}（{change:+.2f}）")

            # 动态生成退化原因分析
            if "SAF" in test_id:
                print(f"      可能原因：修改 Prompt 时弱化了安全约束")
                print(f"      建议：在 Prompt 中重新强调安全规则")
            elif "STA" in test_id:
                print(f"      可能原因：Prompt 变复杂后输出波动增大")
                print(f"      建议：增加输出格式约束以减少随机性")
            elif "EFF" in test_id:
                print(f"      可能原因：新增的示例可能误导了模型")
                print(f"      建议：调整示例内容或增加更多覆盖")
            elif "FMT" in test_id:
                print(f"      可能原因：格式要求和有效性要求产生冲突")
                print(f"      建议：平衡格式约束和内容质量")
