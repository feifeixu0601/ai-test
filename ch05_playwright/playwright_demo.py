"""
Playwright 基础练习
演示动态页面和静态页面的区别
"""

import os
import pathlib
from bs4 import BeautifulSoup


# ========== 实验一：用传统方式读取动态页面 ==========

print("=" * 50)
print("实验一：传统方式（模拟 requests 的行为）")
print("=" * 50)

with open("dynamic_page_demo.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")
items = soup.find_all("div", class_="job-item")

print(f"找到 {len(items)} 个职位")  # 输出 0！
print(f"job-list 容器内容：")
job_list = soup.find("div", id="job-list")
print(f"  {job_list}")  # <div id="job-list"></div>，空的！



# ========== 实验二：用 Playwright 渲染动态页面 ==========

print("\n" + "=" * 50)
print("实验二：Playwright 渲染后提取")
print("=" * 50)

from playwright.sync_api import sync_playwright

# 获取 HTML 文件的绝对路径（Playwright 需要完整路径）
file_path = pathlib.Path("dynamic_page_demo.html").resolve()
file_url = file_path.as_uri()

with sync_playwright() as p:
    # 启动浏览器（headless=True 表示无界面模式，不会弹出浏览器窗口）
    browser = p.chromium.launch(headless=True)

    # 新建一个浏览器标签页
    page = browser.new_page()

    # 打开本地 HTML 文件
    page.goto(file_url)

    # 等待 JavaScript 执行完成，页面中出现 .job-item 元素
    page.wait_for_selector(".job-item")

    # 获取渲染后的完整 HTML（JavaScript 已执行，DOM 已更新）
    rendered_html = page.content()

    # 关闭浏览器
    browser.close()

# 用 BeautifulSoup 解析渲染后的 HTML
soup = BeautifulSoup(rendered_html, "lxml")
items = soup.find_all("div", class_="job-item")

print(f"找到 {len(items)} 个职位")  # 输出 3！

for item in items:
    title = item.find("h3").text
    salary = item.find("span", class_="salary").text
    company = item.find("span", class_="company").text
    print(f"  {title} | {salary} | {company}")
