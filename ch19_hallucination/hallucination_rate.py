"""
幻觉率评估与降级策略
量化模型的幻觉率，并给出对应的降级建议

数据来源：
  从 hallucination_detect.py 的输出手动填写，
  或从真实 LLM 批量测试结果读取
"""

import numpy as np
from datetime import datetime

print("=" * 60)
print("幻觉率评估与降级策略")
print("=" * 60)


# ========== 1. 测试数据 ==========

print("\n--- 1. 测试数据 ---")

# 模拟一批测试结果（综合判定后的结果）
# 实际使用时从 hallucination_detect.py 的输出读取
# 或填入真实 LLM 的检测结果
test_results = [
    {"id": "T-01", "question": "鲁迅是谁？", "verdict": "可信"},
    {"id": "T-02", "question": "鲁迅原名？", "verdict": "疑似幻觉"},
    {"id": "T-03", "question": "地球是什么？", "verdict": "可信"},
    {"id": "T-04", "question": "水的化学式？", "verdict": "可信"},
    {"id": "T-05", "question": "介绍 Python", "verdict": "存疑-需审核"},
    {"id": "T-06", "question": "介绍长城", "verdict": "可信"},
    {"id": "T-07", "question": "月球有多远？", "verdict": "可信"},
    {"id": "T-08", "question": "量子计算原理？", "verdict": "疑似幻觉"},
    {"id": "T-09", "question": "光速是多少？", "verdict": "可信"},
    {"id": "T-10", "question": "爱因斯坦生平？", "verdict": "存疑-需审核"},
    {"id": "T-11", "question": "太阳系行星？", "verdict": "可信"},
    {"id": "T-12", "question": "DNA 结构？", "verdict": "可信"},
    {"id": "T-13", "question": "历史上的今天？", "verdict": "疑似幻觉"},
    {"id": "T-14", "question": "如何学编程？", "verdict": "可信"},
    {"id": "T-15", "question": "量子纠缠应用？", "verdict": "存疑-需审核"},
    {"id": "T-16", "question": "黑洞是怎么形成的？", "verdict": "可信"},
    {"id": "T-17", "question": "推荐一本书？", "verdict": "疑似幻觉"},
    {"id": "T-18", "question": "中国经济数据？", "verdict": "存疑-需审核"},
    {"id": "T-19", "question": "机器学习定义？", "verdict": "可信"},
    {"id": "T-20", "question": "宇宙有多大？", "verdict": "可信"},
]

print(f"  测试样本数：{len(test_results)}")

verdicts = {}
for r in test_results:
    v = r["verdict"]
    verdicts[v] = verdicts.get(v, 0) + 1

for v, count in verdicts.items():
    print(f"    {v}：{count} 个（{count/len(test_results)*100:.1f}%）")


# ========== 2. 幻觉率计算 ==========

print(f"\n--- 2. 幻觉率计算 ---")

total = len(test_results)

# 严格模式：只把"疑似幻觉"算作幻觉
strict_hallucination = sum(1 for r in test_results if r["verdict"] == "疑似幻觉")
strict_rate = strict_hallucination / total

# 宽松模式：把"疑似幻觉"和"存疑"都算作幻觉
loose_hallucination = sum(1 for r in test_results if r["verdict"] in ["疑似幻觉", "存疑-需审核"])
loose_rate = loose_hallucination / total

print(f"\n  严格幻觉率（仅'疑似幻觉'）：{strict_hallucination}/{total} = {strict_rate:.1%}")
print(f"  宽松幻觉率（含'存疑'）：{loose_hallucination}/{total} = {loose_rate:.1%}")


def judge_hallucination_rate(rate):
    """根据幻觉率给出判定"""
    if rate < 0.05:
        return "[PASS]", "幻觉率低，大部分场景可靠"
    elif rate < 0.15:
        return "[WARN]", "幻觉率中等，高风险场景需要人工审核"
    elif rate < 0.30:
        return "[FAIL]", "幻觉率较高，需要优化 Prompt 或模型"
    else:
        return "[FAIL]", "幻觉率严重，不建议直接面向用户"


strict_verdict, strict_desc = judge_hallucination_rate(strict_rate)
loose_verdict, loose_desc = judge_hallucination_rate(loose_rate)

print(f"\n  严格模式判定：{strict_verdict}  {strict_desc}")
print(f"  宽松模式判定：{loose_verdict}  {loose_desc}")


# ========== 3. 降级策略建议 ==========

print(f"\n{'─' * 60}")
print(f"3. 降级策略建议")
print(f"{'─' * 60}")

print(f"\n  根据幻觉率水平，建议采取以下降级策略：")
print()

if strict_rate < 0.05:
    print(f"  当前幻觉率低（{strict_rate:.1%}），建议策略：")
    print(f"    1. 维持现有部署模式")
    print(f"    2. 对'存疑'回答做抽查审核（{verdicts.get('存疑-需审核', 0)} 个）")
    print(f"    3. 定期重新评估幻觉率")
elif strict_rate < 0.15:
    print(f"  当前幻觉率中等（{strict_rate:.1%}），建议策略：")
    print(f"    1. 对高风险场景（医疗、金融）启用人工审核")
    print(f"    2. 对'疑似幻觉'回答自动触发重新生成")
    print(f"    3. 在 Prompt 中增加'不确定请说明'的约束")
else:
    print(f"  当前幻觉率较高（{strict_rate:.1%}），建议策略：")
    print(f"    1. 全部回答接入人工审核通道")
    print(f"    2. 切换为检索增强模式（用搜索代替纯生成）")
    print(f"    3. 对无法验证的问题明确声明'我对此不确定'")
    print(f"    4. 重新训练模型或优化 Prompt")

# 针对不同类型问题的降级建议
print(f"\n  按问题类型的降级策略：")

strategies = {
    "事实性问题（历史、科学）": "知识库验证 + 检索增强，禁止无依据回答",
    "时效性问题（新闻、数据）": "声明知识截止时间 + 引用信息来源",
    "主观性问题（推荐、评价）": "给出多个选项 + 说明推荐理由",
    "高风险问题（医疗、法律、金融）": "强制人工审核 + 免责声明",
    "创意性问题（写作、设计）": "允许生成，但标注为'AI 生成内容'",
}

for q_type, strategy in strategies.items():
    print(f"    {q_type}")
    print(f"      → {strategy}")


# ========== 4. 幻觉率趋势可视化 ==========

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("charts", exist_ok=True)

print(f"\n--- 4. 幻觉率趋势可视化 ---")

# 模拟多个版本的幻觉率变化（实际使用时从历史数据读取）
versions = ["v1.0", "v1.1", "v1.2", "v2.0", "v2.1"]
strict_rates = [0.25, 0.20, 0.15, 0.10, 0.08]
loose_rates = [0.45, 0.38, 0.30, 0.22, 0.18]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(versions, strict_rates, "o-", color="#2ECC71", linewidth=2.5,
        markersize=10, label="严格幻觉率（仅'疑似幻觉'）")
ax.plot(versions, loose_rates, "s--", color="#E74C3C", linewidth=2.5,
        markersize=10, label="宽松幻觉率（含'存疑'）")

# 画阈值线
ax.axhline(y=0.05, color="green", linestyle=":", alpha=0.5, label="PASS 阈值（5%）")
ax.axhline(y=0.15, color="orange", linestyle=":", alpha=0.5, label="WARN 阈值（15%）")
ax.axhline(y=0.30, color="red", linestyle=":", alpha=0.5, label="FAIL 阈值（30%）")

# 标注数值
for i, (sr, lr) in enumerate(zip(strict_rates, loose_rates)):
    ax.annotate(f"{sr:.0%}", xy=(i, sr), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=11)
    ax.annotate(f"{lr:.0%}", xy=(i, lr), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=11)

ax.set_xlabel("模型版本", fontsize=12)
ax.set_ylabel("幻觉率", fontsize=12)
ax.set_title("幻觉率趋势监控\n（每次模型更新后重新评估）",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig("charts/hallucination_trend.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  幻觉率趋势图已保存到 charts/hallucination_trend.png")

print(f"\n  从图上能看到：")
print(f"    1. 严格幻觉率从 25% 降到 8%，持续改善")
print(f"    2. 宽松幻觉率从 45% 降到 18%，但始终高于严格率")
print(f"    3. 虚线是三个阈值线，帮助判断每个版本是否达标")
print(f"    4. 监控趋势比单次数值更有价值——万一幻觉率回升，说明模型更新引入了问题")
