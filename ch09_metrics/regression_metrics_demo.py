"""
回归任务评估指标
用模拟数据演示 MAE、MSE、RMSE、R² 的含义和区别
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置中文字体（Mac 用户改成 "PingFang SC"）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("回归任务评估指标")
print("=" * 60)

# ========== 场景：预测模型推理延迟 ==========
# 真实延迟（毫秒）
np.random.seed(42)
y_true = np.random.uniform(50, 200, 30)

# 模型A：预测比较准
y_pred_a = y_true + np.random.normal(0, 10, 30)

# 模型B：预测有系统性偏差（都偏高）
y_pred_b = y_true + np.random.normal(30, 15, 30)

# 模型C：预测波动大，偶尔有极端错误
y_pred_c = y_true + np.random.normal(0, 40, 30)
y_pred_c[0] = y_true[0] + 150    # 一个极端错误

# ========== 计算回归指标 ==========
def calc_regression_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"  MAE（平均绝对误差）:  {mae:.2f} ms")
    print(f"  MSE（均方误差）:      {mse:.2f}")
    print(f"  RMSE（均方根误差）:   {rmse:.2f} ms")
    print(f"  R²（决定系数）:       {r2:.4f}")

    return {"name": name, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

m_a = calc_regression_metrics(y_true, y_pred_a, "模型A：预测准确")
m_b = calc_regression_metrics(y_true, y_pred_b, "模型B：有系统偏差")
m_c = calc_regression_metrics(y_true, y_pred_c, "模型C：波动大+极端错误")

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, y_pred, m, title in zip(
    axes, [y_pred_a, y_pred_b, y_pred_c], [m_a, m_b, m_c],
    ["模型A：准确", "模型B：有偏差", "模型C：波动大"]
):
    ax.scatter(y_true, y_pred, c="#2E86C1", s=50, alpha=0.7, edgecolors="gray")

    # 完美预测线（y=x）
    min_val = min(y_true.min(), y_pred.min()) - 10
    max_val = max(y_true.max(), y_pred.max()) + 10
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="完美预测线")

    ax.set_xlabel("真实延迟 (ms)", fontsize=11)
    ax.set_ylabel("预测延迟 (ms)", fontsize=11)
    ax.set_title(f"{title}\nMAE={m['mae']:.1f}, R²={m['r2']:.3f}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("回归模型预测效果对比（越靠近红色虚线越好）", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/05_regression_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n对比图已保存到 charts/05_regression_metrics.png")
