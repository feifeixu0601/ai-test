"""
CV 核心概念演示
卷积操作的直觉理解与可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("CV 核心概念演示")
print("=" * 60)


# ========== 1. 图片 = 数字矩阵 ==========

print("\n--- 1. 图片就是数字矩阵 ---")

# 创建一个简单的 8x8 图片（模拟一个简单的"笑脸"）
# 0 = 黑色，1 = 白色
image = np.zeros((8, 8))
image[1:7, 1:7] = 1       # 一个白色方块
image[3, 3] = 0           # 左眼（黑点）
image[3, 4] = 0           # 右眼（黑点）
image[5, 2:6] = 0          # 嘴巴（黑色横线）

print(f"  图片尺寸: {image.shape}")
print(f"  图片数据（0=黑, 1=白）:")
for row in image:
    print(f"    {' '.join('█' if v > 0.5 else '·' for v in row)}")


# ========== 2. 卷积操作 ==========

print(f"\n--- 2. 卷积操作演示 ---")
print(f"用一个 3x3 的卷积核在图片上滑动，提取局部特征")

def convolve2d(image, kernel):
    """
    手写一个 2D 卷积操作
    理解卷积到底在做什么
    """
    kh, kw = kernel.shape
    ih, iw = image.shape
    oh, ow = ih - kh + 1, iw - kw + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            # 取出当前位置的 3x3 小区域
            region = image[i:i+kh, j:j+kw]
            # 加权求和
            output[i, j] = np.sum(region * kernel)

    return output


# 定义三种不同的卷积核
kernels = {
    "水平边缘检测": np.array([[-1, -1, -1],
                              [ 0,  0,  0],
                              [ 1,  1,  1]]),

    "垂直边缘检测": np.array([[-1,  0,  1],
                              [-1,  0,  1],
                              [-1,  0,  1]]),

    "锐化": np.array([[ 0, -1,  0],
                      [-1,  5, -1],
                      [ 0, -1,  0]]),
}

print(f"\n  定义了 {len(kernels)} 种卷积核，分别提取不同的特征")

# 对图片应用每种卷积核
feature_maps = {}
for name, kernel in kernels.items():
    feature_map = convolve2d(image, kernel)
    feature_maps[name] = feature_map
    print(f"\n  [{name}]")
    print(f"  卷积核:")
    for row in kernel:
        print(f"    {' '.join(f'{v:>3d}' for v in row)}")
    print(f"  特征图（提取结果）:")
    for row in feature_map:
        print(f"    {' '.join(f'{v:>5.1f}' for v in row)}")


# ========== 3. 可视化：原图和特征图 ==========

print(f"\n--- 3. 可视化 ---")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 原图
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("原图（8x8 灰度图）", fontsize=12)
axes[0].axis("off")

# 各个特征图
for idx, (name, fmap) in enumerate(feature_maps.items(), 1):
    axes[idx].imshow(fmap, cmap="RdBu_r")
    axes[idx].set_title(f"特征图：{name}", fontsize=12)
    axes[idx].axis("off")

plt.suptitle("卷积操作：从原图中提取不同特征", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/cv_convolution.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  卷积可视化已保存到 charts/cv_convolution.png")


# ========== 4. 特征层级概念 ==========

print(f"\n--- 4. 特征层级（从简单到复杂） ---")
print(f"""
  CNN 的深层结构就像一个"特征组装工厂"：

  第 1 层（低级特征）：
    提取边缘、线条、颜色块
    → 相当于看到图片中最基本的视觉元素

  第 2 层（中级特征）：
    把边缘组合成纹理、形状
    → 毛茸茸的纹理、圆形的眼睛、三角形的耳朵

  第 3 层（高级特征）：
    把形状组合成物体部件
    → 猫脸、猫爪、猫尾巴

  第 4 层（最终判断）：
    把部件组合成完整概念
    → "这是一只猫"，置信度 92%

  每一层提取的特征越来越抽象，越来越接近人类理解的"概念"
""")
