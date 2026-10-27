"""
过拟合与欠拟合可视化演示
用多项式回归来直观感受不同复杂度的模型表现
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 设置中文字体（Mac 用户改成 "PingFang SC"）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("过拟合与欠拟合可视化演示")
print("=" * 60)

# ========== 生成模拟数据 ==========
# 真实规律：y = sin(x) + 少量噪声
np.random.seed(42)
X = np.sort(np.random.uniform(0, 2 * np.pi, 20))   # 20 个训练点
y = np.sin(X) + np.random.normal(0, 0.2, X.shape)  # sin 函数 + 噪声

# 用于画曲线的密集 X 值
X_plot = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y_true = np.sin(X_plot.ravel())    # 真实曲线

# ========== 用不同复杂度的模型来拟合 ==========
# 复杂度由多项式的阶数控制：1 阶（直线）、3 阶（简单曲线）、15 阶（复杂曲线）

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

degrees = [1, 3, 15]
titles = ["欠拟合（1阶多项式）", "刚好（3阶多项式）", "过拟合（15阶多项式）"]

for ax, degree, title in zip(axes, degrees, titles):
    # 生成多项式特征
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    X_plot_poly = poly.transform(X_plot)

    # 训练模型
    model = LinearRegression()
    model.fit(X_poly, y)

    # 预测
    y_plot = model.predict(X_plot_poly)
    y_train_pred = model.predict(X_poly)

    # 计算训练误差
    train_mse = mean_squared_error(y, y_train_pred)

    # 画图
    ax.plot(X_plot, y_true, "k--", alpha=0.5, label="真实规律 (sin)")
    ax.scatter(X, y, c="red", s=60, zorder=5, label="训练数据")
    ax.plot(X_plot, y_plot, "b-", linewidth=2, label=f"模型预测")
    ax.set_title(f"{title}\n训练误差: {train_mse:.4f}", fontsize=13)
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("不同复杂度模型的拟合效果", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("overfit_demo.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n可视化图片已保存：overfit_demo.png")

# ========== 量化验证：训练误差 vs 测试误差 ==========
print("\n--- 量化验证：不同复杂度的训练/测试误差 ---")

# 生成更多数据用于测试
np.random.seed(99)
X_test = np.sort(np.random.uniform(0, 2 * np.pi, 100))
y_test = np.sin(X_test) + np.random.normal(0, 0.2, X_test.shape)

print(f"\n{'阶数':>4s} | {'训练误差':>10s} | {'测试误差':>10s} | {'状态':>10s}")
print("-" * 50)

for degree in [1, 2, 3, 5, 8, 15, 20]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_train_poly, y)

    train_mse = mean_squared_error(y, model.predict(X_train_poly))
    test_mse = mean_squared_error(y_test, model.predict(X_test_poly))

    # 判断拟合状态
    if train_mse > 0.08:
        status = "欠拟合"
    elif test_mse > train_mse * 1.5:
        status = "过拟合"
    else:
        status = "刚好"

    print(f"{degree:>4d} | {train_mse:>10.4f} | {test_mse:>10.4f} | {status:>10s}")
