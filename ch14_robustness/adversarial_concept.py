"""
对抗样本概念可视化
在 2D 平面上演示对抗攻击的原理

核心内容：
  1. 训练一个简单的二分类模型
  2. 可视化决策边界
  3. 生成对抗样本，展示微小扰动如何翻转预测
  4. 对比不同扰动大小 ε 的效果
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("对抗样本概念可视化")
print("=" * 60)


# ========== 1. 创建训练数据 ==========

print("\n--- 1. 创建训练数据 ---")

np.random.seed(42)

# 两个类别的数据点
X_class0 = np.random.randn(80, 2) * 0.8 + [2, 2]
X_class1 = np.random.randn(80, 2) * 0.8 + [5, 5]

X_train = np.vstack([X_class0, X_class1])
y_train = np.array([0] * 80 + [1] * 80)

print(f"  类别 0（蓝色）：{len(X_class0)} 个样本，中心 (2, 2)")
print(f"  类别 1（红色）：{len(X_class1)} 个样本，中心 (5, 5)")


# ========== 2. 训练模型 ==========

print(f"\n--- 2. 训练逻辑回归模型 ---")

model = LogisticRegression()
model.fit(X_train, y_train)

w = model.coef_[0]       # 权重向量
b = model.intercept_[0]  # 偏置

print(f"  权重向量 w = [{w[0]:.4f}, {w[1]:.4f}]")
print(f"  偏置 b = {b:.4f}")
print(f"  决策边界方程：{w[0]:.4f} * x1 + {w[1]:.4f} * x2 + {b:.4f} = 0")


# ========== 3. 选择测试点 ==========

print(f"\n--- 3. 选择测试点 ---")

# 选一个在类别 1 区域内部、但离决策边界不远的点
# 注意：不能选 (3.5, 3.5) 这种正好压在边界上的点
# 否则原始预测可能就是类别 0，攻击就没有"翻转"效果了
x_test = np.array([3.8, 3.8])
score = model.decision_function([x_test])[0]  # 正 = 类别1, 负 = 类别0
pred = model.predict([x_test])[0]

print(f"  测试点 x = {x_test}")
print(f"  决策分数 = {score:.4f}  （正数 = 类别 1）")
print(f"  预测结果 = 类别 {pred}")

# 确保原始预测是类别 1
if pred != 1:
    print(f"  [WARN] 测试点未被预测为类别 1，请调整 x_test 的位置")
else:
    print(f"  [PASS] 测试点正确预测为类别 1")

# 计算到决策边界的距离
distance_to_boundary = abs(score) / np.linalg.norm(w)
print(f"  到决策边界的距离 = {distance_to_boundary:.4f}")
print(f"  距离不算远，所以可以被对抗扰动翻转")


# ========== 4. 生成对抗样本（FGSM）==========

print(f"\n--- 4. 生成对抗样本（FGSM） ---")


def fgsm_attack(x, y_true, model, epsilon):
    """
    FGSM 对抗攻击

    原理：沿着使模型损失增大的方向，加一个大小为 ε 的扰动
    公式：x_adv = x + ε * sign(∇ₓ L(x, y))

    对于逻辑回归：
      损失梯度 ∝ (预测概率 - 真实标签) * 权重向量
    """
    # 获取模型对类别 1 的预测概率
    prob = model.predict_proba([x])[0][1]

    # 计算损失对输入的梯度（简化版）
    grad = (prob - y_true) * w

    # 生成扰动
    delta = epsilon * np.sign(grad)

    return x + delta, delta


# 尝试不同大小的扰动
epsilons = [0.1, 0.3, 0.5, 1.0]
y_true = 1  # 真实标签是类别 1

print(f"\n  原始点 x = {x_test}，真实类别 = {y_true}")
print(f"  决策分数 = {score:.4f}，预测 = 类别 {pred}")
print()

attack_results = []
for eps in epsilons:
    x_adv, delta = fgsm_attack(x_test, y_true, model, eps)
    pred_adv = model.predict([x_adv])[0]
    score_adv = model.decision_function([x_adv])[0]

    changed = "[翻转]" if pred_adv != pred else "[未变]"

    print(f"  ε = {eps:.1f}")
    print(f"    扰动 delta = [{delta[0]:+.4f}, {delta[1]:+.4f}]  "
          f"（扰动幅度 = {np.linalg.norm(delta):.4f}）")
    print(f"    对抗样本 = [{x_adv[0]:.4f}, {x_adv[1]:.4f}]")
    print(f"    决策分数 = {score_adv:.4f}  预测 = 类别 {pred_adv}  {changed}")
    print()

    attack_results.append({
        "epsilon": eps,
        "x_adv": x_adv,
        "delta": delta,
        "pred": pred_adv,
        "score": score_adv,
    })


# ========== 5. 可视化 ==========

print(f"\n--- 5. 可视化对抗攻击过程 ---")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ===== 左图：决策边界 + 数据点 + 对抗样本 =====
ax = axes[0]

# 画决策边界背景
xx, yy = np.meshgrid(np.linspace(-1, 8, 300), np.linspace(-1, 8, 300))
Z = model.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.4)
ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2, linestyles="--")

# 训练数据
ax.scatter(X_class0[:, 0], X_class0[:, 1], c="#3498DB", s=20,
           alpha=0.5, label="类别 0（训练）")
ax.scatter(X_class1[:, 0], X_class1[:, 1], c="#E74C3C", s=20,
           alpha=0.5, label="类别 1（训练）")

# 原始测试点
ax.scatter(x_test[0], x_test[1], c="#2ECC71", s=150, marker="*",
           edgecolors="black", linewidths=1.5, zorder=10,
           label=f"原始点（类别 {pred}）")

# 各个对抗样本
colors_adv = ["#F39C12", "#9B59B6", "#E67E22", "#1ABC9C"]
for i, result in enumerate(attack_results):
    x_adv = result["x_adv"]
    ax.scatter(x_adv[0], x_adv[1], s=100, marker="X",
               c=colors_adv[i], edgecolors="black", linewidths=1,
               zorder=10,
               label=f"ε={result['epsilon']} → 类别 {result['pred']}")

    # 画扰动方向箭头
    ax.annotate("", xy=x_adv, xytext=x_test,
                arrowprops=dict(arrowstyle="->", color=colors_adv[i],
                                lw=1.5, alpha=0.7))

ax.set_xlabel("特征 x1", fontsize=12)
ax.set_ylabel("特征 x2", fontsize=12)
ax.set_title("对抗攻击可视化\n（虚线 = 决策边界，X = 对抗样本）",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

# ===== 右图：决策分数变化 =====
ax2 = axes[1]

eps_labels = ["原始"] + [f"ε={r['epsilon']}" for r in attack_results]
scores = [score] + [r["score"] for r in attack_results]
preds = [pred] + [r["pred"] for r in attack_results]

# 颜色逻辑：预测被翻转的涂红，没翻转的涂绿
colors_bar = ["#2ECC71"] + [
    "#E74C3C" if p != pred else "#2ECC71" for p in preds[1:]
]

bars = ax2.bar(eps_labels, scores, color=colors_bar,
               edgecolor="gray", alpha=0.8)

# 画零线（决策边界）
ax2.axhline(y=0, color="black", linewidth=2, linestyle="--",
            label="决策边界（0）")

# 标注每根柱子的分数和预测类别
for bar, s, p in zip(bars, scores, preds):
    # 修复：正数柱标注在柱子上方，负数柱标注在柱子下方
    if s >= 0:
        y_pos = bar.get_height() + 0.12
        va = "bottom"
    else:
        y_pos = bar.get_height() - 0.45
        va = "top"

    ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
             f"{s:.2f}\n类别{p}", ha="center", va=va, fontsize=10)

ax2.set_ylabel("决策分数（>0 = 类别1，<0 = 类别0）", fontsize=12)
ax2.set_title("不同扰动大小下的决策分数\n（柱子跨过零线 = 预测翻转）",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")

# 设置 y 轴范围，给标注留空间
y_min = min(scores) - 1.0
y_max = max(scores) + 0.5
ax2.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig("charts/adversarial_concept.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  对抗攻击可视化已保存到 charts/adversarial_concept.png")
print(f"\n  看图你能直观感受到：")
print(f"    1. 原始点在类别 1 区域，决策分数为正")
print(f"    2. 微小的扰动（ε=0.1 或 0.3）就能让决策分数变负")
print(f"    3. 柱子跨过零线，就说明预测被翻转了（红色柱子）")
print(f"    4. 扰动越大，离决策边界越远，对抗样本越容易被发现")