"""
混淆矩阵与基础分类指标
从混淆矩阵出发，理解所有分类指标的来源
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 设置中文字体（Mac 用户改成 "PingFang SC"）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("混淆矩阵演示")
print("=" * 60)

# ========== 场景：癌症筛查模型 ==========
# 1000 个患者，20 个有病，980 个没病

# 真实标签：1 = 有病，0 = 没病
np.random.seed(42)
y_true = np.array([1] * 20 + [0] * 980)

# ========== 模型A：偷懒模型（全部预测"没病"）==========
y_pred_lazy = np.zeros(1000, dtype=int)    # 全部预测 0（没病）

# ========== 模型B：一个合理的模型 ==========
y_pred_smart = np.zeros(1000, dtype=int)
# 正确识别出 16 个有病的，漏掉 4 个
y_pred_smart[:16] = 1    # 16 个真正例 TP
# 误报了 8 个没病的
false_pos_indices = np.random.choice(range(20, 1000), 8, replace=False)
y_pred_smart[false_pos_indices] = 1


def show_confusion_matrix(y_true, y_pred, model_name, filename):
    """计算并显示混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} 的混淆矩阵 ---")
    print(f"              预测:没病  预测:有病")
    print(f"  实际:没病    TN={cm[0,0]:>4d}   FP={cm[0,1]:>4d}")
    print(f"  实际:有病    FN={cm[1,0]:>4d}   TP={cm[1,1]:>4d}")

    # 提取四个值
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  TP（正确识别有病）: {tp}")
    print(f"  TN（正确识别没病）: {tn}")
    print(f"  FP（误报，没病说成有病）: {fp}")
    print(f"  FN（漏报，有病说成没病）: {fn}")

    # 可视化混淆矩阵
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["没病", "有病"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"{model_name}\n混淆矩阵", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"charts/{filename}", dpi=150, bbox_inches="tight")
    plt.close()

    return tn, fp, fn, tp


# 展示两个模型的混淆矩阵
tn_a, fp_a, fn_a, tp_a = show_confusion_matrix(
    y_true, y_pred_lazy, "模型A（偷懒模型）", "01_cm_lazy.png")

tn_b, fp_b, fn_b, tp_b = show_confusion_matrix(
    y_true, y_pred_smart, "模型B（合理模型）", "02_cm_smart.png")

print("\n图片已保存到 charts/ 目录")
