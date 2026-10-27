"""
AI 测试岗位数据可视化
把第 06 篇的分析结果做成可视化图表

数据来源：job_data.json（第 04 篇爬虫生成）
输出目录：charts/（自动生成）
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter

# 设置中文字体（Mac 用户改成 "PingFang SC"）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 常量配置
INPUT_JSON = "../data/job_data.json"
OUTPUT_DIR = "charts"

# 自定义配色方案
COLORS = {
    "primary": "#2E86C1",
    "secondary": "#E74C3C",
    "accent": "#F39C12",
    "bg": "#F8F9FA",
    "palette": ["#2E86C1", "#E74C3C", "#F39C12", "#27AE60", "#8E44AD", "#1ABC9C", "#E67E22", "#34495E"],
}

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("AI 测试岗位数据可视化启动")
print(f"图表输出目录：{OUTPUT_DIR}/")


def load_data():
    """
    加载岗位数据并做基础处理
    """
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"数据加载成功：{len(data)} 条岗位")
        return data
    except FileNotFoundError:
        print(f"数据文件不存在：{INPUT_JSON}")
        print("请先运行第 04 篇的爬虫代码生成数据")
        return None


def chart_skills(data):
    """
    图表1：技能需求 Top 10 柱状图
    这是最重要的图表——直接回答"该学什么"
    """
    # 统计所有技能标签
    all_tags = []
    for job in data:
        all_tags.extend(job.get("tags", []))
    tag_counter = Counter(all_tags)
    top_tags = tag_counter.most_common(10)

    if not top_tags:
        print("没有技能标签数据，跳过")
        return

    labels = [t[0] for t in top_tags]
    counts = [t[1] for t in top_tags]

    fig, ax = plt.subplots(figsize=(12, 7))

    # 画柱状图，颜色由深到浅（出现次数越多颜色越深）
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)

    # 在柱子顶部标注数值
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(count),
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    ax.set_title("AI 测试岗位技能需求 Top 10", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("技能", fontsize=13)
    ax.set_ylabel("出现次数", fontsize=13)
    ax.set_ylim(0, max(counts) + 1.5)
    ax.spines["top"].set_visible(False)    # 去掉上边框
    ax.spines["right"].set_visible(False)  # 去掉右边框

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_skills_top10.png"), dpi=150, bbox_inches="tight")
    plt.close()    # 关闭画布，释放内存
    print("  图表1（技能需求Top10）完成")


def chart_salary_levels(data):
    """
    图表2：薪资分层饼图
    展示不同薪资段的岗位占比
    """
    # 按薪资分层统计
    levels = {"初级(<25K)": 0, "中级(25-40K)": 0, "高级(40-55K)": 0, "专家(55K+)": 0, "未统计": 0}

    for job in data:
        low = job.get("salary_low")
        high = job.get("salary_high")
        salary_type = job.get("salary_type", "")

        if salary_type != "K" or not low or not high:
            levels["未统计"] += 1
            continue

        avg = (low + high) / 2
        if avg < 25000:
            levels["初级(<25K)"] += 1
        elif avg < 40000:
            levels["中级(25-40K)"] += 1
        elif avg < 55000:
            levels["高级(40-55K)"] += 1
        else:
            levels["专家(55K+)"] += 1

    # 过滤掉数量为 0 的
    levels = {k: v for k, v in levels.items() if v > 0}

    if not levels:
        print("没有薪资数据，跳过")
        return

    labels = list(levels.keys())
    sizes = list(levels.values())
    colors = COLORS["palette"][:len(labels)]

    fig, ax = plt.subplots(figsize=(9, 9))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.03] * len(labels),
        textprops={"fontsize": 12},
        pctdistance=0.75
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)

    ax.set_title("AI 测试岗位薪资分层", fontsize=18, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_salary_levels.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  图表2（薪资分层饼图）完成")


def chart_locations(data):
    """
    图表3：地区分布与薪资对比
    双 Y 轴：左边柱状图是岗位数，右边折线图是平均薪资
    """
    # 按城市统计
    location_jobs = {}
    for job in data:
        loc = job.get("location", "未知")
        if loc not in location_jobs:
            location_jobs[loc] = []
        location_jobs[loc].append(job)

    # 计算各城市数据
    cities = []
    job_counts = []
    avg_salaries = []

    for loc, jobs in sorted(location_jobs.items(), key=lambda x: len(x[1]), reverse=True):
        cities.append(loc)
        job_counts.append(len(jobs))

        # 计算月薪岗位的平均薪资
        monthly = [
            (j["salary_low"] + j["salary_high"]) / 2
            for j in jobs
            if j.get("salary_type") == "K" and j.get("salary_low") and j.get("salary_high")
        ]
        avg_salaries.append(sum(monthly) / len(monthly) if monthly else 0)

    x = np.arange(len(cities))

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 左 Y 轴：岗位数量（柱状图）
    bars = ax1.bar(x, job_counts, 0.5, color=COLORS["primary"], alpha=0.7, label="岗位数量")
    ax1.set_xlabel("城市", fontsize=13)
    ax1.set_ylabel("岗位数量", fontsize=13, color=COLORS["primary"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(cities, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])

    # 在柱子上标注数量
    for bar, count in zip(bars, job_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(count), ha="center", fontsize=11, fontweight="bold")

    # 右 Y 轴：平均薪资（折线图）
    ax2 = ax1.twinx()
    line = ax2.plot(x, avg_salaries, "o-", color=COLORS["secondary"],
                    linewidth=2.5, markersize=10, label="平均薪资")
    ax2.set_ylabel("平均薪资（元/月）", fontsize=13, color=COLORS["secondary"])
    ax2.tick_params(axis="y", labelcolor=COLORS["secondary"])

    # 在折线点上标注薪资
    for i, salary in enumerate(avg_salaries):
        if salary > 0:
            ax2.annotate(f"{salary/1000:.0f}K",
                        xy=(i, salary), xytext=(0, 12),
                        textcoords="offset points",
                        ha="center", fontsize=10, color=COLORS["secondary"])

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)

    plt.title("各城市岗位数量与薪资对比", fontsize=18, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_locations.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  图表3（地区分布与薪资对比）完成")


def chart_experience_salary(data):
    """
    图表4：经验与薪资关系散点图
    直观展示"经验越多薪资越高"的规律
    """
    # 提取有薪资数据的岗位
    points = []
    for job in data:
        if (job.get("salary_type") == "K"
                and job.get("salary_low") and job.get("salary_high")):
            avg_salary = (job["salary_low"] + job["salary_high"]) / 2
            exp = job.get("experience", "")
            # 把经验字符串转成数字
            exp_years = {
                "在校生": 0.5, "1-3年": 2, "3-5年": 4, "5-10年": 7.5
            }.get(exp, 3)
            points.append((exp_years, avg_salary, job.get("title", ""), exp))

    if len(points) < 3:
        print("数据不足，跳过散点图")
        return

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    titles = [p[2] for p in points]
    exp_labels = [p[3] for p in points]

    fig, ax = plt.subplots(figsize=(12, 7))

    # 散点图，颜色按薪资映射
    scatter = ax.scatter(x, y, c=y, cmap="YlOrRd", s=120, alpha=0.8,
                         edgecolors="gray", linewidth=0.5, zorder=5)

    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("平均薪资（元/月）", fontsize=11)

    # 添加趋势线
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
        ax.plot(x_line, p_line(x_line), "--", color="gray", linewidth=1.5,
                alpha=0.7, label="趋势线")

    # 标注每个点的岗位名
    for xi, yi, title, exp in points:
        ax.annotate(title, xy=(xi, yi), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, alpha=0.7)

    ax.set_title("经验 vs 薪资", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("经验（年）", fontsize=13)
    ax.set_ylabel("平均月薪（元）", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_experience_salary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  图表4（经验与薪资关系）完成")


def chart_demographics(data):
    """
    图表5：学历和经验分布（子图，一排两图）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- 左图：学历分布 ----
    edu_counter = Counter(job.get("education", "未知") for job in data)
    edu_labels = list(edu_counter.keys())
    edu_counts = list(edu_counter.values())
    edu_colors = COLORS["palette"][:len(edu_labels)]

    ax1.bar(edu_labels, edu_counts, color=edu_colors, edgecolor="white", linewidth=0.8)
    for i, (label, count) in enumerate(zip(edu_labels, edu_counts)):
        ax1.text(i, count + 0.1, str(count), ha="center", fontsize=12, fontweight="bold")
    ax1.set_title("学历要求分布", fontsize=14, fontweight="bold")
    ax1.set_ylabel("岗位数量", fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ---- 右图：经验分布 ----
    exp_counter = Counter(job.get("experience", "未知") for job in data)
    exp_labels = list(exp_counter.keys())
    exp_counts = list(exp_counter.values())
    exp_colors = COLORS["palette"][:len(exp_labels)]

    ax2.bar(exp_labels, exp_counts, color=exp_colors, edgecolor="white", linewidth=0.8)
    for i, (label, count) in enumerate(zip(exp_labels, exp_counts)):
        ax2.text(i, count + 0.1, str(count), ha="center", fontsize=12, fontweight="bold")
    ax2.set_title("经验要求分布", fontsize=14, fontweight="bold")
    ax2.set_ylabel("岗位数量", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("岗位要求分布", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_demographics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  图表5（学历与经验分布）完成")


def chart_skill_radar(data):
    """
    图表6：技能类别雷达图
    展示各技能类别的综合需求强度
    """
    # 统计技能类别
    skill_categories = {
        "编程语言": ["Python"],
        "AI/ML技术": ["机器学习", "深度学习", "大模型", "LLM", "NLP", "计算机视觉", "TensorFlow", "PyTorch"],
        "测试技能": ["自动化测试", "pytest", "测试框架", "模型测试", "测试开发", "测试基础", "数据标注"],
        "评测方法": ["LLM评测", "模型评测", "Prompt Engineering", "RAG", "BLEU", "ROUGE", "文本评测"],
        "数据技能": ["数据分析", "Pandas", "SQL", "数据清洗", "数据标注"],
        "工程技能": ["Docker", "CI/CD", "Selenium", "Red Team", "对抗攻击", "大模型安全", "质量保障", "A/B测试"],
    }

    all_tags = []
    for job in data:
        all_tags.extend(job.get("tags", []))
    tag_counter = Counter(all_tags)

    categories = []
    values = []
    for category, keywords in skill_categories.items():
        count = sum(tag_counter.get(kw, 0) for kw in keywords)
        categories.append(category)
        values.append(count)

    if not any(values):
        print("没有技能类别数据，跳过")
        return

    # 雷达图需要闭合数据（首尾相接）
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    # 画雷达图
    ax.fill(angles_closed, values_closed, color=COLORS["primary"], alpha=0.25)
    ax.plot(angles_closed, values_closed, color=COLORS["primary"], linewidth=2.5, marker="o", markersize=8)

    # 设置每个轴的标签
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=12)

    # 在每个数据点旁标注数值
    for angle, value, category in zip(angles, values, categories):
        if value > 0:
            ax.annotate(str(value), xy=(angle, value), xytext=(5, 5),
                        textcoords="offset points", fontsize=11, fontweight="bold",
                        color=COLORS["secondary"])

    ax.set_title("技能类别需求强度", fontsize=18, fontweight="bold", pad=30)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_skill_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  图表6（技能类别雷达图）完成")


def chart_skill_checklist(data):
    """
    图表7：技能需求横向条形图
    实用性最强的一张图——可以直接当学习清单
    """
    all_tags = []
    for job in data:
        all_tags.extend(job.get("tags", []))
    tag_counter = Counter(all_tags)
    all_tags_sorted = tag_counter.most_common()    # 全部技能

    if not all_tags_sorted:
        return

    labels = [t[0] for t in all_tags_sorted]
    counts = [t[1] for t in all_tags_sorted]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.5)))

    # 横向柱状图
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(labels)))
    bars = ax.barh(labels, counts, color=colors, edgecolor="white", linewidth=0.5)

    # 标注数值
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), va="center", fontsize=11, fontweight="bold")

    ax.set_title("AI 测试岗位全部技能需求清单", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("出现次数", fontsize=13)
    ax.invert_yaxis()    # 翻转 Y 轴，让最多的技能在最上面
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_skill_checklist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  图表7（技能清单）完成")


# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载数据
    data = load_data()
    if not data:
        exit()

    # 生成所有图表
    print(f"\n开始生成图表...")

    chart_skills(data)              # 图表1：技能需求 Top 10
    chart_salary_levels(data)       # 图表2：薪资分层饼图
    chart_locations(data)           # 图表3：地区分布与薪资对比
    chart_experience_salary(data)   # 图表4：经验与薪资散点图
    chart_demographics(data)        # 图表5：学历与经验分布
    chart_skill_radar(data)         # 图表6：技能类别雷达图
    chart_skill_checklist(data)     # 图表7：技能清单

    print(f"\n全部图表生成完毕！")
    print(f"输出目录：{OUTPUT_DIR}/")
    print(f"\n生成的图表文件：")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"  {f}  ({size:.0f} KB)")
