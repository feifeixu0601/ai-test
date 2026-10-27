"""
神经元工作原理演示
用手写数字识别的简化场景，理解单个神经元是如何做决定的
"""

import numpy as np

print("=" * 60)
print("单个神经元工作原理")
print("=" * 60)


# ========== 场景：判断"这个手写数字是不是 3" ==========

# 我们假设有 3 个特征（实际中会有 784 个像素特征）
# 特征1：图像中间竖线的强度（3 有一个弧形竖线）
# 特征2：图像右上角弧度的强度（3 的右上角是圆弧）
# 特征3：图像左下角弧度的强度（3 的左下角也是圆弧）

def neuron(inputs, weights, bias, activation="sigmoid"):
    """
    一个人工神经元的完整计算过程

    参数：
        inputs:  输入特征 [x1, x2, x3]
        weights: 权重 [w1, w2, w3]，表示每个特征的重要性
        bias:    偏置，调整激活的难易程度
        activation: 激活函数类型

    返回：
        神经元的输出值
    """
    # 第一步：每个输入乘以对应的权重
    weighted = [x * w for x, w in zip(inputs, weights)]

    # 第二步：加权求和
    total = sum(weighted) + bias

    # 第三步：通过激活函数
    if activation == "sigmoid":
        # Sigmoid 函数：把任意实数压缩到 0~1 之间
        # 就像把一个模糊的"程度"变成一个确定的"概率"
        output = 1 / (1 + np.exp(-total))
    elif activation == "relu":
        # ReLU 函数：负数变 0，正数不变
        # 就像一个单向阀，只让正信号通过
        output = max(0, total)
    else:
        output = total

    return output, total


# ========== 实验1：不同输入的输出 ==========

print("\n--- 实验1：同一个神经元，不同的输入 ---")

# 这个神经元学会了"识别数字 3"
weights = [0.5, 0.8, 0.6]    # 权重：特征2（右上角弧度）最重要
bias = -1.0                   # 偏置：让激活门槛高一点

test_cases = [
    ("数字 3（很像）",   [0.8, 0.9, 0.7]),
    ("数字 3（有点模糊）", [0.5, 0.6, 0.5]),
    ("数字 8（不太像）", [0.3, 0.2, 0.4]),
    ("数字 1（完全不像）", [0.1, 0.0, 0.1]),
]

for name, inputs in test_cases:
    output, raw_sum = neuron(inputs, weights, bias, activation="sigmoid")
    is_three = "是 3" if output > 0.5 else "不是 3"
    print(f"  {name}")
    print(f"    输入特征: {inputs}")
    print(f"    加权求和: {raw_sum:.3f}")
    print(f"    Sigmoid输出: {output:.4f}")
    print(f"    判断结果: {is_three}（置信度 {output:.1%}）")
    print()

    # ========== 激活函数可视化 ==========

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    print("\n--- 激活函数可视化 ---")

    x = np.linspace(-5, 5, 200)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))

    # ReLU
    relu = np.maximum(0, x)

    # Tanh
    tanh = np.tanh(x)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Sigmoid
    axes[0].plot(x, sigmoid, "b-", linewidth=2.5)
    axes[0].set_title("Sigmoid\n输出压缩到 0~1\n适合输出概率", fontsize=13)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.3, 1.3)

    # ReLU
    axes[1].plot(x, relu, "r-", linewidth=2.5)
    axes[1].set_title("ReLU\n负数变0，正数不变\n深度学习最常用", fontsize=13)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.5, 5)

    # Tanh
    axes[2].plot(x, tanh, "g-", linewidth=2.5)
    axes[2].set_title("Tanh\n输出压缩到 -1~1\n关于原点对称", fontsize=13)
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1.5, 1.5)

    plt.suptitle("三种常见激活函数", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/activation_functions.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  激活函数图表已保存到 charts/activation_functions.png")
