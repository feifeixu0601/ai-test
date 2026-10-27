"""
精确率与召回率对比
用不同模型来理解精确率和召回率的权衡
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 设置中文字体（Mac 用户改成 "PingFang SC"）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("精确率与召回率对比")
print("=" * 60)

# 真实标签：1 = 有病(20人)，0 = 没病(980人)
np.random.seed(42)
y_true = np.array([1] * 20 + [0] * 980)

# ========== 三种模型策略 ==========

# 模型A：全部预测"没病"（保守策略，不误报但全漏掉）
y_pred_a = np.zeros(1000, dtype=int)

# 模型B：合理策略（找对大部分，少量误报）
y_pred_b = np.zeros(1000, dtype=int)
y_pred_b[:16] = 1                      # TP = 16
fp_indices = np.random.choice(range(20, 1000), 8, replace=False)
y_pred_b[fp_indices] = 1               # FP = 8

# 模型C：激进策略（把很多没病的也报成有病）
y_pred_c = np.zeros(1000, dtype=int)
y_pred_c[:18] = 1                      # TP = 18，只漏 2 个
fp_indices_c = np.random.choice(range(20, 1000), 100, replace=False)
y_pred_c[fp_indices_c] = 1             # FP = 100，误报很多

# ========== 计算各指标 ==========
def calc_all_metrics(y_true, y_pred, name):
    """计算并打印所有分类指标"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- {name} ---")
    print(f"  准确率 (Accuracy):  {acc:.2%}")
    print(f"  精确率 (Precision): {prec:.2%}")
    print(f"  召回率 (Recall):    {rec:.2%}")
    print(f"  F1 值 (F1 Score):   {f1:.2%}")

    return {"name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}

metrics_a = calc_all_metrics(y_true, y_pred_a, "模型A：全部预测没病（保守）")
metrics_b = calc_all_metrics(y_true, y_pred_b, "模型B：合理策略")
metrics_c = calc_all_metrics(y_true, y_pred_c, "模型C：全部报有病（激进）")

# ========== 可视化对比 ==========
all_metrics = [metrics_a, metrics_b, metrics_c]
metric_names = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["准确率", "精确率", "召回率", "F1值"]

x = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

colors = ["#AED6F1", "#2E86C1", "#E74C3C"]
for i, (m, color) in enumerate(zip(all_metrics, colors)):
    values = [m[key] for key in metric_names]
    bars = ax.bar(x + i * width, values, width, label=m["name"][:8],
                  color=color, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.0%}", ha="center", fontsize=9)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=13)
ax.set_ylim(0, 1.15)
ax.set_title("不同模型策略的指标对比", fontsize=16, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/03_precision_recall_compare.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n对比图已保存到 charts/03_precision_recall_compare.png")
