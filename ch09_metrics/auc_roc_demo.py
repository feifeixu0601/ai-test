"""
AUC-ROC 曲线演示
理解 ROC 曲线的含义和 AUC 的计算
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 设置中文字体（Mac 用户改成 "PingFang SC"）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("AUC-ROC 曲线演示")
print("=" * 60)

# ========== 生成模拟数据 ==========
np.random.seed(42)
n_positive = 200    # 正类样本数
n_negative = 800    # 负类样本数

y_true = np.array([1] * n_positive + [0] * n_negative)

# 模型A：好模型（正类概率高，负类概率低，区分度好）
scores_a = np.concatenate([
    np.random.normal(0.75, 0.12, n_positive),   # 正类：概率集中在 0.75 附近
    np.random.normal(0.25, 0.12, n_negative),    # 负类：概率集中在 0.25 附近
])
scores_a = np.clip(scores_a, 0, 1)

# 模型B：中等模型（正负类有一定区分度，但重叠较多）
scores_b = np.concatenate([
    np.random.normal(0.6, 0.2, n_positive),
    np.random.normal(0.4, 0.2, n_negative),
])
scores_b = np.clip(scores_b, 0, 1)

# 模型C：差模型（正负类几乎重叠，基本是随机猜）
scores_c = np.concatenate([
    np.random.normal(0.5, 0.25, n_positive),
    np.random.normal(0.5, 0.25, n_negative),
])
scores_c = np.clip(scores_c, 0, 1)

# ========== 计算 ROC 曲线和 AUC ==========
fpr_a, tpr_a, thresholds_a = roc_curve(y_true, scores_a)
fpr_b, tpr_b, thresholds_b = roc_curve(y_true, scores_b)
fpr_c, tpr_c, thresholds_c = roc_curve(y_true, scores_c)

auc_a = auc(fpr_a, tpr_a)
auc_b = auc(fpr_b, tpr_b)
auc_c = auc(fpr_c, tpr_c)

print(f"\n模型A 的 AUC: {auc_a:.4f}")
print(f"模型B 的 AUC: {auc_b:.4f}")
print(f"模型C 的 AUC: {auc_c:.4f}")

# ========== 画 ROC 曲线 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：ROC 曲线
ax1.plot(fpr_a, tpr_a, color="#2E86C1", linewidth=2.5, label=f"模型A (AUC={auc_a:.3f})")
ax1.plot(fpr_b, tpr_b, color="#E67E22", linewidth=2.5, label=f"模型B (AUC={auc_b:.3f})")
ax1.plot(fpr_c, tpr_c, color="#E74C3C", linewidth=2.5, label=f"模型C (AUC={auc_c:.3f})")

# 对角线（随机猜测）
ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="随机猜测 (AUC=0.5)")

ax1.set_xlabel("假正率 FPR（误报率）", fontsize=12)
ax1.set_ylabel("真正率 TPR（召回率）", fontsize=12)
ax1.set_title("ROC 曲线对比", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(True, alpha=0.3)

# 右图：三个模型的预测概率分布
bins = np.linspace(0, 1, 30)
ax2.hist(scores_a[y_true == 1], bins=bins, alpha=0.6, color="#2E86C1",
         label="模型A-正类", density=True)
ax2.hist(scores_a[y_true == 0], bins=bins, alpha=0.6, color="#AED6F1",
         label="模型A-负类", density=True)
ax2.set_xlabel("模型输出的概率", fontsize=12)
ax2.set_ylabel("密度", fontsize=12)
ax2.set_title("模型A 的预测概率分布\n（正负类区分明显）", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("charts/04_auc_roc.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nROC 曲线图已保存到 charts/04_auc_roc.png")

# ========== AUC 解读 ==========
print(f"\n--- AUC 解读 ---")
print(f"  AUC = 1.00: 完美模型，不管什么阈值都能完美区分")
print(f"  AUC = 0.50: 随机猜测，毫无区分能力")
print(f"  AUC < 0.50: 比随机还差，预测方向反了")
print(f"")
print(f"  本例中：")
print(f"    模型A (AUC={auc_a:.2f}) → 区分能力{'优秀' if auc_a > 0.9 else '良好' if auc_a > 0.8 else '一般'}")
print(f"    模型B (AUC={auc_b:.2f}) → 区分能力{'优秀' if auc_b > 0.9 else '良好' if auc_b > 0.8 else '一般'}")
print(f"    模型C (AUC={auc_c:.2f}) → 区分能力{'优秀' if auc_c > 0.9 else '良好' if auc_c > 0.8 else '一般'}")
