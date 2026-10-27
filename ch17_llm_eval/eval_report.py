"""
大模型评测报告生成
含雷达图的多维度评测报告

说明：
  这里手动输入评测得分（实际使用时从 llm_eval_framework.py 的输出读取）
  重点是报告格式和雷达图的生成方式
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("charts", exist_ok=True)

from datetime import datetime

print("=" * 60)
print("大模型评测报告生成")
print("=" * 60)


# ========== 1. 评测数据 ==========

# 各维度得分（0~1 之间，实际使用时从评测框架输出读取）
eval_data = {
    "知识问答": 0.95,
    "数学推理": 0.88,
    "代码生成": 0.82,
    "阅读理解": 0.90,
    "有害内容防御": 0.96,
    "错误信息防御": 0.85,
    "隐私保护": 0.98,
    "指令遵循": 0.78,
    "诚实性": 0.92,
    "有用性": 0.85,
}

# 按三大维度分组
dimension_groups = {
    "能力": ["知识问答", "数学推理", "代码生成", "阅读理解"],
    "安全": ["有害内容防御", "错误信息防御", "隐私保护"],
    "对齐": ["指令遵循", "诚实性", "有用性"],
}

print(f"\n  评测维度数：{len(eval_data)}")
print(f"  能力维度：{len(dimension_groups['能力'])} 个子项")
print(f"  安全维度：{len(dimension_groups['安全'])} 个子项")
print(f"  对齐维度：{len(dimension_groups['对齐'])} 个子项")


# ========== 2. 雷达图 ==========

print(f"\n--- 2. 生成雷达图 ---")

labels = list(eval_data.keys())
values = list(eval_data.values())

# 雷达图需要闭合
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values_plot = values + [values[0]]
angles += [angles[0]]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# 绘制雷达图
ax.fill(angles, values_plot, color="#2E86C1", alpha=0.2)
ax.plot(angles, values_plot, "o-", color="#2E86C1", linewidth=2.5, markersize=8)

# 标注每个点的分数
for angle, value, label in zip(angles[:-1], values, labels):
    ax.text(angle, value + 0.05, f"{value:.2f}",
            ha="center", va="center", fontsize=10, fontweight="bold")

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
ax.grid(True, alpha=0.3)

ax.set_title("大模型评测雷达图\n（外圈 = 越好）",
             fontsize=16, fontweight="bold", pad=30)

plt.tight_layout()
plt.savefig("charts/eval_radar.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  雷达图已保存到 charts/eval_radar.png")


# ========== 3. 评测报告 ==========

print(f"\n--- 3. 生成评测报告 ---")

lines = []


def add(text=""):
    lines.append(text)


add("=" * 60)
add("大模型评测报告")
add("=" * 60)
add(f"")
add(f"  报告时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
add(f"  被测模型：大语言模型（通用对话）")
add(f"  评测框架：三维评估（能力、安全、对齐）")
add(f"  评测基准：自建评测集（12 条用例）")

# 各维度汇总
add(f"\n{'─' * 60}")
add(f"一、各维度得分")
add(f"{'─' * 60}")

for dim_name, sub_items in dimension_groups.items():
    dim_scores = [eval_data[item] for item in sub_items]
    dim_avg = np.mean(dim_scores)

    add(f"\n  [{dim_name}] 平均分：{dim_avg:.2f}")
    for item in sub_items:
        score = eval_data[item]
        if score >= 0.8:
            marker = "[PASS]"
        elif score >= 0.6:
            marker = "[WARN]"
        else:
            marker = "[FAIL]"
        add(f"    {marker} {item:<12s} {score:.2f}")

# 总体评价
total_avg = np.mean(list(eval_data.values()))
add(f"\n{'─' * 60}")
add(f"二、综合评价")
add(f"{'─' * 60}")

add(f"\n  总体平均分：{total_avg:.2f} / 1.00")

if total_avg >= 0.8:
    add(f"  [PASS] 模型综合表现良好")
elif total_avg >= 0.5:
    add(f"  [WARN] 模型综合表现一般，有提升空间")
else:
    add(f"  [FAIL] 模型综合表现不佳")

# 找出强项和弱项
best_item = max(eval_data.items(), key=lambda x: x[1])
worst_item = min(eval_data.items(), key=lambda x: x[1])

add(f"\n  最强项：{best_item[0]}（{best_item[1]:.2f}）")
add(f"  最弱项：{worst_item[0]}（{worst_item[1]:.2f}）")

add(f"\n  改进建议：")
if worst_item[1] < 0.8:
    add(f"    1. 重点提升'{worst_item[0]}'（当前 {worst_item[1]:.2f}）")
add(f"    2. 持续监控安全维度，防止模型能力提升时安全性下降")
add(f"    3. 每次模型更新后跑同一套评测，对比雷达图变化")
add(f"    4. 对薄弱子项补充针对性训练数据或优化 Prompt")
add(f"    5. 建议扩大评测集规模（每个子项 10~20 条为佳）")

add(f"\n{'=' * 60}")
add(f"报告生成完毕")

report_text = "\n".join(lines)
print(report_text)

with open("eval_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n报告已保存到 eval_report.txt")
