"""
神经网络完整工作过程演示
从单个神经元到多层网络，理解前向传播和训练过程
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("神经网络完整工作过程")
print("=" * 60)


# ========== 手写一个简单的两层神经网络 ==========

class SimpleNeuralNetwork:
    """
    一个简单的两层神经网络
    结构：输入层(2) → 隐藏层(4) → 输出层(1)

    用来做二分类：判断一个点属于红色还是蓝色
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.1):
        # 初始化权重和偏置（随机）
        np.random.seed(42)

        # 第一层：输入层 → 隐藏层
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))

        # 第二层：隐藏层 → 输出层
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Sigmoid 的导数（用于反向传播）"""
        return x * (1 - x)

    def forward(self, X):
        """
        前向传播：数据从输入流向输出

        过程：
          X → [加权求和] → [激活函数] → hidden
          hidden → [加权求和] → [激活函数] → output
        """
        # 第一层
        self.z1 = X.dot(self.W1) + self.b1     # 加权求和
        self.a1 = self.sigmoid(self.z1)          # 激活函数 → 隐藏层输出

        # 第二层
        self.z2 = self.a1.dot(self.W2) + self.b2  # 加权求和
        self.a2 = self.sigmoid(self.z2)            # 激活函数 → 最终输出

        return self.a2

    def backward(self, X, y, output):
        """
        反向传播：计算误差，更新权重

        核心思想：
          1. 计算预测和真实答案之间的差距（损失）
          2. 从输出层往回推，计算每层权重对误差的"贡献"
          3. 调整权重，让误差变小
        """
        m = X.shape[0]    # 样本数

        # 输出层误差
        output_error = y - output                    # 预测和真实的差距
        output_delta = output_error * self.sigmoid_derivative(output)

        # 隐藏层误差（从输出层的误差"传播"回来）
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        # 更新权重和偏置
        self.W2 += self.a1.T.dot(output_delta) * self.learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(hidden_delta) * self.learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=1000):
        """
        训练过程：重复"前向传播 → 计算误差 → 反向传播 → 更新权重"
        """
        losses = []

        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)

            # 计算损失（均方误差）
            loss = np.mean((y - output) ** 2)
            losses.append(loss)

            # 反向传播 + 更新权重
            self.backward(X, y, output)

            if (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1:>5d}/{epochs}  损失: {loss:.6f}")

        return losses


# ========== 训练一个二分类模型 ==========

print("\n--- 训练演示：二分类任务 ---")

# 生成训练数据：两个类别的点
np.random.seed(42)

# 类别 0（蓝色）：围绕 (2, 2) 的点
X_blue = np.random.randn(50, 2) * 0.8 + [2, 2]
y_blue = np.zeros((50, 1))

# 类别 1（红色）：围绕 (6, 6) 的点
X_red = np.random.randn(50, 2) * 0.8 + [6, 6]
y_red = np.ones((50, 1))

# 合并
X_train = np.vstack([X_blue, X_red])
y_train = np.vstack([y_blue, y_red])

print(f"  训练数据：{X_train.shape[0]} 个样本，{X_train.shape[1]} 个特征")

# 创建网络
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)

# 训练
print(f"\n  开始训练...")
losses = nn.train(X_train, y_train, epochs=1000)

# 画损失曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(losses, "b-", linewidth=1.5)
ax1.set_title("训练损失曲线\n（损失越来越小 = 模型在学习）", fontsize=13)
ax1.set_xlabel("训练轮数 (Epoch)", fontsize=11)
ax1.set_ylabel("损失 (Loss)", fontsize=11)
ax1.grid(True, alpha=0.3)

# 画决策边界
xx, yy = np.meshgrid(np.linspace(-1, 9, 200), np.linspace(-1, 9, 200))
grid_points = np.column_stack([xx.ravel(), yy.ravel()])
Z = nn.forward(grid_points).reshape(xx.shape)

ax2.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.6)
ax2.scatter(X_blue[:, 0], X_blue[:, 1], c="blue", s=30, edgecolors="gray",
            label="类别 0（蓝）", zorder=5)
ax2.scatter(X_red[:, 0], X_red[:, 1], c="red", s=30, edgecolors="gray",
            label="类别 1（红）", zorder=5)
ax2.set_title("模型学到的决策边界\n（蓝区预测0，红区预测1）", fontsize=13)
ax2.set_xlabel("特征 1", fontsize=11)
ax2.set_ylabel("特征 2", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("charts/nn_training.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  训练结果图已保存到 charts/nn_training.png")
