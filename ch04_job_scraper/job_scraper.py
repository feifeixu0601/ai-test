"""
AI测试岗位JD数据爬虫
从招聘页面中爬取、清洗、存储、分析岗位数据

流程：读取页面 → 解析数据 → 清洗数据 → 保存文件 → 统计分析
"""

import json
import csv
import re
from collections import Counter
from bs4 import BeautifulSoup


# 常量配置
HTML_FILE = "../data/mock_jobs.html"     # 数据来源（本地模拟页面）
OUTPUT_JSON = "../data/job_data.json"    # 输出 JSON 文件
OUTPUT_CSV = "../data/job_data.csv"      # 输出 CSV 文件

print("AI测试岗位JD数据爬虫启动")


def read_local_page(file_path):
    """
    从本地 HTML 文件读取页面内容
    用于学习阶段（数据来源是 mock_jobs.html）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
    print(f"读取本地页面：{file_path}")
    return BeautifulSoup(html, "lxml")


def read_remote_page(url):
    """
    从远程网页读取页面内容（真实爬虫使用）
    用于生产环境（数据来源是真实招聘网站）
    需要安装 requests：pip install requests
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        print(f"请求远程页面成功：{url}")
        return BeautifulSoup(response.text, "lxml")
    except Exception as e:
        print(f"请求远程页面失败：{e}")
        return None


def parse_jobs(soup):
    """
    从 BeautifulSoup 对象中提取职位数据
    返回原始数据列表（未清洗）
    """
    jobs = []
    job_items = soup.find_all("div", class_="job-item")
    print(f"找到 {len(job_items)} 条职位信息")

    for item in job_items:
        # 提取职位名称
        title_el = item.find("a", class_="job-title")
        title = title_el.get_text(strip=True) if title_el else "未知"

        # 提取薪资
        salary_el = item.find("span", class_="salary")
        salary = salary_el.get_text(strip=True) if salary_el else "未知"

        # 提取基本信息（公司、城市、经验、学历）
        company_el = item.find("span", class_="company")
        company = company_el.get_text(strip=True) if company_el else "未知"

        location_el = item.find("span", class_="location")
        location = location_el.get_text(strip=True) if location_el else "未知"

        experience_el = item.find("span", class_="experience")
        experience = experience_el.get_text(strip=True) if experience_el else "未知"

        education_el = item.find("span", class_="education")
        education = education_el.get_text(strip=True) if education_el else "未知"

        # 提取技能标签（可能有多个）
        tag_els = item.find_all("span", class_="tag")
        tags = [tag.get_text(strip=True) for tag in tag_els]

        # 提取岗位描述
        desc_el = item.find("p", class_="description")
        description = desc_el.get_text(strip=True) if desc_el else ""

        jobs.append({
            "title": title,
            "salary_raw": salary,
            "company": company,
            "location": location,
            "experience": experience,
            "education": education,
            "tags": tags,
            "description": description
        })

    return jobs


def parse_salary(salary_str):
    """
    把薪资字符串解析成数字

    处理格式：
        "25-40K"       → (25000, 40000, "K")
        "200-300元/天"  → (200, 300, "天")
        "未知"          → (None, None, "")

    返回元组：(最低薪资, 最高薪资, 薪资类型)
    """
    if not salary_str or salary_str == "未知":
        return None, None, ""

    # 判断是月薪（K）还是日薪（天）
    if "元/天" in salary_str:
        salary_type = "天"
        # 提取 "200-300元/天" 中的数字
        match = re.search(r'(\d+)-(\d+)', salary_str)
        if match:
            low = int(match.group(1))
            high = int(match.group(2))
            return low, high, salary_type
    elif "K" in salary_str:
        salary_type = "K"
        # 提取 "25-40K" 中的数字
        match = re.search(r'(\d+)-(\d+)', salary_str)
        if match:
            low = int(match.group(1)) * 1000   # K 转换成元
            high = int(match.group(2)) * 1000
            return low, high, salary_type

    return None, None, ""


def clean_jobs(raw_jobs):
    """
    清洗原始数据，增加解析后的薪资字段
    """
    clean_data = []

    for job in raw_jobs:
        salary_low, salary_high, salary_type = parse_salary(job["salary_raw"])

        clean_job = {
            "title": job["title"],
            "salary_raw": job["salary_raw"],
            "salary_low": salary_low,
            "salary_high": salary_high,
            "salary_type": salary_type,
            "company": job["company"],
            "location": job["location"],
            "experience": job["experience"],
            "education": job["education"],
            "tags": job["tags"],
            "description": job["description"]
        }
        clean_data.append(clean_job)

    print(f"数据清洗完成，共 {len(clean_data)} 条")
    return clean_data


def save_json(jobs, file_path):
    """保存为 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    print(f"JSON 数据已保存：{file_path}")


def save_csv(jobs, file_path):
    """保存为 CSV 文件"""
    if not jobs:
        print("没有数据，跳过 CSV 保存")
        return

    # 定义 CSV 列顺序
    fieldnames = [
        "title", "salary_raw", "salary_low", "salary_high", "salary_type",
        "company", "location", "experience", "education", "tags", "description"
    ]

    # encoding="utf-8-sig" 让 Excel 打开不乱码
    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for job in jobs:
            # CSV 里标签字段用逗号连接成字符串
            row = job.copy()
            row["tags"] = ", ".join(job["tags"])
            writer.writerow(row)

    print(f"CSV 数据已保存：{file_path}")


def analyze_jobs(jobs):
    """
    对爬取的岗位数据做基础统计分析
    """
    if not jobs:
        print("没有数据可供分析")
        return

    print("\n" + "=" * 60)
    print("AI 测试岗位数据统计报告")
    print("=" * 60)

    # ---- 1. 总量统计 ----
    print(f"\n【总量】")
    print(f"  岗位总数：{len(jobs)}")

    # ---- 2. 薪资统计（只统计月薪岗位）----
    print(f"\n【薪资分析（月薪岗位）】")
    monthly_jobs = [j for j in jobs if j["salary_type"] == "K"]
    daily_jobs = [j for j in jobs if j["salary_type"] == "天"]

    if monthly_jobs:
        avg_low = sum(j["salary_low"] for j in monthly_jobs) / len(monthly_jobs)
        avg_high = sum(j["salary_high"] for j in monthly_jobs) / len(monthly_jobs)
        max_high = max(j["salary_high"] for j in monthly_jobs)
        min_low = min(j["salary_low"] for j in monthly_jobs)

        print(f"  月薪岗位数：{len(monthly_jobs)}")
        print(f"  平均薪资范围：{avg_low:.0f} - {avg_high:.0f} 元/月")
        print(f"  最高薪资上限：{max_high} 元/月")
        print(f"  最低薪资下限：{min_low} 元/月")

    if daily_jobs:
        print(f"  日薪岗位数（实习）：{len(daily_jobs)}")

    # ---- 3. 技能标签统计（最重要的分析）----
    print(f"\n【技能需求 Top 10】")
    all_tags = []
    for job in jobs:
        all_tags.extend(job["tags"])

    tag_counter = Counter(all_tags)
    for tag, count in tag_counter.most_common(10):
        bar = "█" * (count * 3)
        print(f"  {tag:<20s} {count:>2d} 次  {bar}")

    # ---- 4. 地区分布 ----
    print(f"\n【地区分布】")
    location_counter = Counter(job["location"] for job in jobs)
    for location, count in location_counter.most_common():
        print(f"  {location}：{count} 个岗位")

    # ---- 5. 经验要求分布 ----
    print(f"\n【经验要求分布】")
    exp_counter = Counter(job["experience"] for job in jobs)
    for exp, count in exp_counter.most_common():
        print(f"  {exp}：{count} 个岗位")

    # ---- 6. 学历要求分布 ----
    print(f"\n【学历要求分布】")
    edu_counter = Counter(job["education"] for job in jobs)
    for edu, count in edu_counter.most_common():
        print(f"  {edu}：{count} 个岗位")

    print("\n" + "=" * 60)


# ========== 主程序 ==========
if __name__ == "__main__":
    # 第一步：读取页面
    soup = read_local_page(HTML_FILE)
    # 如需爬取真实网站，换成下面这行：
    # soup = read_remote_page("https://xxx.com/jobs?keyword=AI测试")

    if not soup:
        print("页面读取失败，程序退出")
        exit()

    # 第二步：解析职位数据
    raw_jobs = parse_jobs(soup)

    # 第三步：数据清洗
    clean_jobs_data = clean_jobs(raw_jobs)

    # 第四步：保存数据
    save_json(clean_jobs_data, OUTPUT_JSON)
    save_csv(clean_jobs_data, OUTPUT_CSV)

    # 第五步：统计分析
    analyze_jobs(clean_jobs_data)

    print(f"\n全部完成！")
    print(f"  JSON 文件：{OUTPUT_JSON}")
    print(f"  CSV 文件：{OUTPUT_CSV}")
    print(f"  可以用 Excel 打开 CSV 查看完整数据")
