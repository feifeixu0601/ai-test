"""
Matplotlib 基础知识练习
每段代码可以独立运行
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体（Windows 用 Microsoft YaHei，Mac 用 PingFang SC 或 Arial Unicode MS）
# 如果不设置，中文会显示成方块
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ========== 第一张图：最简单的折线图 ==========
'''
x = [1, 2, 3, 4, 5]
y = [10, 25, 15, 30, 20]

plt.figure(figsize=(8, 5))      # 创建画布，指定尺寸 8x5 英寸
plt.plot(x, y, marker="o")      # 画折线图，marker="o" 表示数据点用圆点标记
plt.title("我的第一张图")         # 标题
plt.xlabel("X 轴")               # X 轴标签
plt.ylabel("Y 轴")               # Y 轴标签
plt.grid(True)                   # 显示网格线
plt.tight_layout()               # 自动调整布局，防止标签被裁切
plt.savefig("chart_demo_1.png", dpi=150)   # 保存为图片
plt.show()                       # 显示图片（在弹出的窗口中查看）


# ========== 中文字体设置说明 ==========

# Windows 系统用 SimHei（黑体）
# matplotlib.rcParams["font.sans-serif"] = ["SimHei"]

# Mac 系统用 PingFang SC 或 Arial Unicode MS
# matplotlib.rcParams["font.sans-serif"] = ["PingFang SC"]

# Linux 系统用 WenQuanYi Zen Hei
# matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]

# 如果你不确定系统有什么字体，可以用下面的代码查看
from matplotlib import font_manager
fonts = set(f.name for f in font_manager.fontManager.ttflist)
chinese_fonts = [f for f in fonts if any(kw in f for kw in ["SimHei", "PingFang", "WenQuanYi", "Microsoft YaHei", "SimSun"])]
print(f"系统中可用的中文字体：{chinese_fonts}")


# ========== 柱状图 ==========

print("\n--- 柱状图演示 ---")

skills = ["Python", "机器学习", "大模型", "pytest", "NLP", "Docker"]
counts = [10, 5, 3, 2, 2, 1]

plt.figure(figsize=(10, 6))

# 画柱状图
bars = plt.bar(skills, counts, color="steelblue", edgecolor="white", linewidth=0.5)

# 在柱子顶部显示数值（让图表更精确）
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,   # x 坐标：柱子中心
        bar.get_height() + 0.1,               # y 坐标：柱子顶部往上一点
        str(count),                            # 显示的文字
        ha="center",                           # 水平居中
        fontsize=12
    )

plt.title("AI 测试岗位技能需求", fontsize=16)
plt.xlabel("技能", fontsize=12)
plt.ylabel("出现次数", fontsize=12)
plt.ylim(0, max(counts) + 1)    # Y 轴范围，留一点空间给顶部文字
plt.tight_layout()
plt.savefig("chart_demo_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# 横向柱状图
plt.figure(figsize=(10, 6))
plt.barh(skills, counts, color="coral")    # barh 就是 bar horizontal
plt.xlabel("出现次数")
plt.title("AI 测试岗位技能需求（横向）")
plt.tight_layout()
plt.savefig("chart_demo_barh.png", dpi=150, bbox_inches="tight")
plt.show()


# ========== 饼图 ==========

print("\n--- 饼图演示 ---")

levels = ["初级(<25K)", "中级(25-40K)", "高级(40-55K)", "专家(55K+)"]
level_counts = [2, 5, 3, 2]
colors = ["#AED6F1", "#5DADE2", "#2E86C1", "#1B4F72"]  # 由浅到深的蓝色系

plt.figure(figsize=(8, 8))

# 画饼图
wedges, texts, autotexts = plt.pie(
    level_counts,
    labels=levels,
    colors=colors,
    autopct="%1.1f%%",          # 显示百分比，保留一位小数
    startangle=90,              # 起始角度（90 表示从顶部开始）
    explode=(0, 0, 0, 0.05),    # 最后一块稍微弹出一点（可选，突出重点）
    textprops={"fontsize": 12}  # 文字大小
)

# 设置百分比文字的样式
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("bold")

plt.title("AI 测试岗位薪资分层", fontsize=16)
plt.tight_layout()
plt.savefig("chart_demo_pie.png", dpi=150, bbox_inches="tight")
plt.show()


# ========== 直方图 ==========

print("\n--- 直方图演示 ---")

# 模拟一批薪资数据（正态分布）
np.random.seed(42)
salaries = np.random.normal(loc=38000, scale=12000, size=100).astype(int)
# 裁剪到合理范围
salaries = salaries[(salaries > 10000) & (salaries < 80000)]

plt.figure(figsize=(10, 6))

# 画直方图
plt.hist(salaries, bins=15, color="steelblue", edgecolor="white", alpha=0.8)

# 添加均值线
mean_salary = salaries.mean()
plt.axvline(mean_salary, color="red", linestyle="--", linewidth=2, label=f"均值: {mean_salary:,.0f}")

plt.title("AI 测试岗位薪资分布", fontsize=16)
plt.xlabel("月薪（元）", fontsize=12)
plt.ylabel("岗位数量", fontsize=12)
plt.legend()   # 显示图例
plt.tight_layout()
plt.savefig("chart_demo_hist.png", dpi=150, bbox_inches="tight")
plt.show()


# ========== 散点图 ==========

print("\n--- 散点图演示 ---")

# 模拟数据：经验年限 vs 月薪
np.random.seed(42)
experience = np.random.uniform(1, 10, 50)          # 1~10 年经验
salary = 15000 + experience * 5000 + np.random.normal(0, 8000, 50)  # 和经验正相关 + 随机波动

plt.figure(figsize=(10, 6))

# 画散点图
plt.scatter(
    experience, salary,
    c=salary,             # 点的颜色根据薪资变化
    cmap="YlOrRd",        # 颜色映射：黄-橙-红（越高薪资越红）
    s=80,                 # 点的大小
    alpha=0.7,            # 透明度
    edgecolors="gray",    # 点的边框
    linewidth=0.5
)

# 添加颜色条
plt.colorbar(label="月薪（元）")

# 添加趋势线（线性回归）
z = np.polyfit(experience, salary, 1)
p = np.poly1d(z)
x_line = np.linspace(experience.min(), experience.max(), 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, label="趋势线")

plt.title("经验 vs 薪资", fontsize=16)
plt.xlabel("经验（年）", fontsize=12)
plt.ylabel("月薪（元）", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("chart_demo_scatter.png", dpi=150, bbox_inches="tight")
plt.show()


# ========== 分组柱状图（对比分析） ==========

print("\n--- 分组柱状图演示 ---")

cities = ["北京", "上海", "杭州", "深圳"]
avg_salary = [52000, 45000, 38000, 35000]
job_count = [5, 3, 2, 2]

x = np.arange(len(cities))    # [0, 1, 2, 3]
width = 0.35                   # 柱子宽度

fig, ax1 = plt.subplots(figsize=(10, 6))

# 左 Y 轴：平均薪资（柱状图）
bars1 = ax1.bar(x - width/2, avg_salary, width, label="平均薪资", color="steelblue")
ax1.set_ylabel("平均薪资（元/月）", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(cities, fontsize=12)

# 右 Y 轴：岗位数量（折线图）
ax2 = ax1.twinx()    # 创建共享 X 轴的第二个 Y 轴
line = ax2.plot(x + width/2, job_count, "ro-", linewidth=2, markersize=8, label="岗位数量")
ax2.set_ylabel("岗位数量", fontsize=12)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)

plt.title("各城市薪资与岗位数量对比", fontsize=16)
plt.tight_layout()
plt.savefig("chart_demo_group.png", dpi=150, bbox_inches="tight")
plt.show()


'''
# ========== 子图 ==========

print("\n--- 子图演示 ---")

# 创建 2x2 的子图布局
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图 1：柱状图
skills = ["Python", "ML", "LLM", "NLP", "Docker"]
counts = [10, 5, 3, 2, 1]
axes[0, 0].bar(skills, counts, color="steelblue")
axes[0, 0].set_title("技能需求", fontsize=13)
axes[0, 0].set_ylabel("次数")

# 子图 2：饼图
labels = ["初级", "中级", "高级", "专家"]
sizes = [2, 5, 3, 2]
axes[0, 1].pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#AED6F1", "#5DADE2", "#2E86C1", "#1B4F72"])
axes[0, 1].set_title("薪资分层", fontsize=13)

# 子图 3：直方图
data = np.random.normal(35000, 10000, 80)
axes[1, 0].hist(data, bins=12, color="coral", edgecolor="white")
axes[1, 0].set_title("薪资分布", fontsize=13)
axes[1, 0].set_xlabel("月薪")

# 子图 4：折线图
months = ["1月", "2月", "3月", "4月", "5月", "6月"]
trend = [12, 15, 18, 22, 28, 35]
axes[1, 1].plot(months, trend, "go-", linewidth=2, markersize=8)
axes[1, 1].set_title("岗位需求趋势", fontsize=13)
axes[1, 1].set_ylabel("岗位数")

plt.suptitle("AI 测试岗位分析总览", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("chart_demo_subplots.png", dpi=150, bbox_inches="tight")
plt.show()
