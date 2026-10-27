"""
AI 测试岗位数据分析
读取爬取的 JD 数据，进行清洗、转换、分析，输出分析报告

数据来源：job_data.json（第04篇爬虫生成）
输出文件：analysis_report.txt
"""

import pandas as pd
import json
from collections import Counter


# 常量配置
INPUT_JSON = "../data/job_data.json"
OUTPUT_REPORT = "../data/analysis_report.txt"

print("AI 测试岗位数据分析启动")
print("=" * 60)


def load_data():
    """
    读取岗位数据，返回 DataFrame
    """
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"数据加载成功：{len(df)} 条岗位记录")
        return df
    except FileNotFoundError:
        print(f"数据文件不存在：{INPUT_JSON}")
        print("请先运行第04篇的爬虫代码生成数据")
        return None


def clean_data(df):
    """
    数据清洗：检查质量、处理缺失值、统一格式
    """
    print(f"\n--- 数据清洗 ---")

    # 1. 查看数据基本信息
    print(f"原始数据量：{len(df)} 条")
    print(f"列名：{list(df.columns)}")

    # 2. 检查缺失值
    print(f"\n缺失值统计：")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"  {col}: {count} 个缺失值")

    # 3. 处理缺失值：字符串列填"未知"，数值列填 None 保持缺失
    str_cols = ["title", "company", "location", "experience", "education", "description"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("未知")

    # 4. 确保 tags 列是列表类型
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])

    # 5. 删除完全重复的行
    before_count = len(df)
    df = df.drop_duplicates(subset=["title", "company", "location", "salary_raw"])
    after_count = len(df)
    if before_count != after_count:
        print(f"  去重：删除了 {before_count - after_count} 条重复数据")

    # 6. 标准化城市名（去掉多余空格）
    if "location" in df.columns:
        df["location"] = df["location"].str.strip()

    print(f"清洗后数据量：{len(df)} 条")
    return df


def transform_data(df):
    """
    数据转换：薪资解析、技能标签展开、增加分析维度
    """
    print(f"\n--- 数据转换 ---")

    # 1. 计算平均薪资（取上下限的中间值）
    # 只对月薪岗位（salary_type == "K"）计算
    def calc_avg_salary(row):
        # 只计算月薪岗位（salary_type == "K"），排除日薪等其他类型
        if row.get("salary_type") == "K" and row.get("salary_low") and row.get("salary_high"):
            return (row["salary_low"] + row["salary_high"]) / 2
        return None

    df["avg_salary"] = df.apply(calc_avg_salary, axis=1)
    monthly_count = df["avg_salary"].notna().sum()
    print(f"  计算了 {monthly_count} 条月薪岗位的平均薪资")

    # 2. 增加薪资分层
    def salary_level(salary):
        if pd.isna(salary):
            return "未统计"
        if salary < 25000:
            return "初级（<25K）"
        elif salary < 40000:
            return "中级（25-40K）"
        elif salary < 55000:
            return "高级（40-55K）"
        else:
            return "专家（55K+）"

    df["salary_level"] = df["avg_salary"].apply(salary_level)
    print(f"  完成薪资分层")

    # 3. 展开技能标签（为技能分析准备）
    # 每条岗位可能有多个 tag，我们把它们全部收集起来
    all_tags = []
    for tags in df["tags"]:
        if isinstance(tags, list):
            all_tags.extend(tags)

    tag_counter = Counter(all_tags)
    print(f"  技能标签展开：共 {len(all_tags)} 个标签，去重后 {len(tag_counter)} 个")

    # 4. 增加技能数量列（每条岗位要求的技能数）
    df["tag_count"] = df["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    return df, tag_counter


def analyze_data(df, tag_counter):
    """
    多维度数据分析
    返回分析结果字典
    """
    results = {}

    # ---- 1. 薪资分析 ----
    monthly = df[df["avg_salary"].notna()]
    results["salary"] = {
        "月薪岗位数": len(monthly),
        "平均薪资": monthly["avg_salary"].mean(),
        "最高薪资": monthly["avg_salary"].max(),
        "最低薪资": monthly["avg_salary"].min(),
        "中位数薪资": monthly["avg_salary"].median(),
    }

    # 按薪资分层统计
    results["salary_level_dist"] = df["salary_level"].value_counts().to_dict()

    print(f"\n薪资分析完成")

    # ---- 2. 技能需求分析（最核心） ----
    results["top_skills"] = tag_counter.most_common(15)

    # 技能分类统计
    skill_categories = {
        "编程语言": ["Python"],
        "AI/ML技术": ["机器学习", "深度学习", "大模型", "LLM", "NLP", "计算机视觉", "TensorFlow", "PyTorch"],
        "测试技能": ["自动化测试", "pytest", "测试框架", "模型测试", "测试开发", "测试基础", "数据标注"],
        "评测方法": ["LLM评测", "模型评测", "Prompt Engineering", "RAG", "BLEU", "ROUGE", "文本评测"],
        "数据技能": ["数据分析", "Pandas", "SQL", "数据清洗", "数据标注"],
        "工程技能": ["Docker", "CI/CD", "Selenium", "Red Team", "对抗攻击", "大模型安全", "质量保障", "A/B测试"],
    }

    category_counts = {}
    for category, keywords in skill_categories.items():
        count = sum(tag_counter.get(kw, 0) for kw in keywords)
        if count > 0:
            category_counts[category] = count

    results["skill_categories"] = category_counts
    print(f"技能分析完成")

    # ---- 3. 地区分析 ----
    results["location_dist"] = df["location"].value_counts().to_dict()

    # 按城市分组的平均薪资
    location_salary = monthly.groupby("location")["avg_salary"].agg(["mean", "count"])
    results["location_salary"] = location_salary.to_dict("index")
    print(f"地区分析完成")

    # ---- 4. 经验要求分析 ----
    results["experience_dist"] = df["experience"].value_counts().to_dict()

    # 按经验分组的平均薪资
    exp_salary = monthly.groupby("experience")["avg_salary"].agg(["mean", "count"])
    results["experience_salary"] = exp_salary.to_dict("index")
    print(f"经验分析完成")

    # ---- 5. 学历要求分析 ----
    results["education_dist"] = df["education"].value_counts().to_dict()
    print(f"学历分析完成")

    # ---- 6. 薪资与技能关联分析 ----
    # 找出高薪岗位（平均薪资前 25%）的技能共性
    if len(monthly) > 4:
        threshold = monthly["avg_salary"].quantile(0.75)
        high_salary_jobs = df[df["avg_salary"] >= threshold]
        high_salary_tags = []
        for tags in high_salary_jobs["tags"]:
            if isinstance(tags, list):
                high_salary_tags.extend(tags)
        results["high_salary_skills"] = Counter(high_salary_tags).most_common(10)
        results["high_salary_threshold"] = threshold
        results["high_salary_count"] = len(high_salary_jobs)
        print(f"高薪岗位技能分析完成")

    return results


def generate_report(results, df):
    """
    生成分析报告，保存到文本文件并打印到终端
    """
    lines = []

    def add(text=""):
        lines.append(text)

    add("=" * 60)
    add("AI 测试岗位技能需求分析报告")
    add("=" * 60)
    add(f"\n数据范围：{len(df)} 条 AI 测试相关岗位")
    add(f"分析维度：薪资、技能、地区、经验、学历、高薪岗位")

    # ---- 薪资分析 ----
    s = results.get("salary", {})
    add(f"\n{'─' * 60}")
    add(f"一、薪资分析")
    add(f"{'─' * 60}")
    add(f"  月薪岗位数：{s.get('月薪岗位数', 0)}")
    add(f"  平均薪资：{s.get('平均薪资', 0):,.0f} 元/月")
    add(f"  中位数薪资：{s.get('中位数薪资', 0):,.0f} 元/月")
    add(f"  薪资范围：{s.get('最低薪资', 0):,.0f} ~ {s.get('最高薪资', 0):,.0f} 元/月")

    add(f"\n  薪资分层分布：")
    for level, count in results.get("salary_level_dist", {}).items():
        bar = "█" * (count * 3)
        add(f"    {level:<16s} {count:>2d} 个岗位  {bar}")

    # ---- 技能分析 ----
    add(f"\n{'─' * 60}")
    add(f"二、技能需求分析（最重要的部分）")
    add(f"{'─' * 60}")

    add(f"\n  技能需求 Top 15：")
    for i, (tag, count) in enumerate(results.get("top_skills", []), 1):
        bar = "█" * (count * 2)
        add(f"    {i:>2d}. {tag:<24s} {count:>2d} 次  {bar}")

    add(f"\n  按技能类别统计：")
    for category, count in sorted(results.get("skill_categories", {}).items(),
                                   key=lambda x: x[1], reverse=True):
        bar = "█" * (count * 2)
        add(f"    {category:<12s} {count:>3d} 次  {bar}")

    # ---- 地区分析 ----
    add(f"\n{'─' * 60}")
    add(f"三、地区分布")
    add(f"{'─' * 60}")

    add(f"\n  岗位地区分布：")
    for location, count in results.get("location_dist", {}).items():
        bar = "█" * (count * 3)
        add(f"    {location:<8s} {count:>2d} 个岗位  {bar}")

    loc_salary = results.get("location_salary", {})
    if loc_salary:
        add(f"\n  各城市平均薪资：")
        for loc, stats in sorted(loc_salary.items(), key=lambda x: x[1].get("mean", 0), reverse=True):
            mean_salary = stats.get("mean", 0)
            count = stats.get("count", 0)
            add(f"    {loc:<8s} {mean_salary:>10,.0f} 元/月  ({int(count)} 个岗位)")

    # ---- 经验分析 ----
    add(f"\n{'─' * 60}")
    add(f"四、经验要求分析")
    add(f"{'─' * 60}")

    add(f"\n  经验要求分布：")
    for exp, count in results.get("experience_dist", {}).items():
        bar = "█" * (count * 3)
        add(f"    {exp:<10s} {count:>2d} 个岗位  {bar}")

    exp_salary = results.get("experience_salary", {})
    if exp_salary:
        add(f"\n  各经验等级平均薪资：")
        for exp, stats in exp_salary.items():
            mean_salary = stats.get("mean", 0)
            add(f"    {exp:<10s} {mean_salary:>10,.0f} 元/月")

    # ---- 学历分析 ----
    add(f"\n{'─' * 60}")
    add(f"五、学历要求分析")
    add(f"{'─' * 60}")

    add(f"\n  学历要求分布：")
    for edu, count in results.get("education_dist", {}).items():
        bar = "█" * (count * 3)
        add(f"    {edu:<8s} {count:>2d} 个岗位  {bar}")

    # ---- 高薪岗位分析 ----
    if "high_salary_skills" in results:
        add(f"\n{'─' * 60}")
        add(f"六、高薪岗位技能分析")
        add(f"{'─' * 60}")
        threshold = results.get("high_salary_threshold", 0)
        count = results.get("high_salary_count", 0)
        add(f"\n  高薪门槛：平均薪资 >= {threshold:,.0f} 元/月（前 25%）")
        add(f"  高薪岗位数：{count}")

        add(f"\n  高薪岗位最看重的技能 Top 10：")
        for i, (tag, freq) in enumerate(results.get("high_salary_skills", []), 1):
            bar = "█" * freq
            add(f"    {i:>2d}. {tag:<24s} {freq:>2d} 次  {bar}")

    # ---- 总结建议 ----
    add(f"\n{'─' * 60}")
    add(f"七、学习建议总结")
    add(f"{'─' * 60}")
    add(f"")
    add(f"  根据以上数据，AI 测试工程师的技能优先级建议如下：")
    add(f"")
    add(f"  第一优先级（必备）：")
    add(f"    Python 编程")
    add(f"    基本测试方法论")
    add(f"    数据分析基础")
    add(f"")
    add(f"  第二优先级（核心竞争力）：")
    add(f"    AI/ML 基础知识")
    add(f"    模型评测方法")
    add(f"    自动化测试框架")
    add(f"")
    add(f"  第三优先级（差异化优势）：")
    add(f"    大模型评测（LLM）")
    add(f"    Prompt Engineering")
    add(f"    RAG 系统测试")
    add(f"    安全测试")

    add(f"\n{'=' * 60}")
    add(f"报告生成完毕")

    # 输出报告
    report_text = "\n".join(lines)

    # 打印到终端
    print(report_text)

    # 保存到文件
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存到 {OUTPUT_REPORT}")

    return report_text


# ========== 主程序 ==========
if __name__ == "__main__":
    # 第一步：读取数据
    df = load_data()
    if df is None:
        exit()

    # 第二步：数据清洗
    df = clean_data(df)

    # 第三步：数据转换
    df, tag_counter = transform_data(df)

    # 第四步：数据分析
    results = analyze_data(df, tag_counter)

    # 第五步：生成报告
    generate_report(results, df)

    print(f"\n全部完成！")
