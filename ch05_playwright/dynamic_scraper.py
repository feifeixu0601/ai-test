"""
动态页面爬虫实战
用 Playwright 渲染 JavaScript 动态页面，提取名言数据

目标：https://quotes.toscrape.com/js/
特点：数据由 JavaScript 动态加载，HTML 源码中没有数据
"""

import json
import time
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin



def scrape_quotes_with_playwright(base_url, max_pages=5):
    """
    用 Playwright 爬取动态加载的名言数据

    参数：
        base_url: 目标网址
        max_pages: 最大爬取页数

    返回：
        所有名言数据的列表
    """
    all_quotes = []

    with sync_playwright() as p:
        # 启动无界面浏览器
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        current_url = base_url
        page_num = 1

        while current_url and page_num <= max_pages:
            print(f"\n[第 {page_num} 页] {current_url}")

            # 打开页面
            page.goto(current_url)

            # 等待 JavaScript 渲染完成，页面中出现 .quote 元素
            page.wait_for_selector(".quote", timeout=10000)

            # 获取渲染后的 HTML
            rendered_html = page.content()

            # 用 BeautifulSoup 解析渲染后的 HTML
            soup = BeautifulSoup(rendered_html, "lxml")
            quote_divs = soup.find_all("div", class_="quote")
            print(f"  找到 {len(quote_divs)} 条名言")

            for div in quote_divs:
                text_el = div.find("span", class_="text")
                author_el = div.find("small", class_="author")
                tag_els = div.find_all("a", class_="tag")

                all_quotes.append({
                    "text": text_el.get_text(strip=True) if text_el else "",
                    "author": author_el.get_text(strip=True) if author_el else "未知",
                    "tags": [t.get_text(strip=True) for t in tag_els]
                })

            # 查找下一页按钮
            next_btn = soup.find("li", class_="next")
            if next_btn:
                next_a = next_btn.find("a")
                if next_a and next_a.get("href"):
                    # 用 page.url（当前页面实际 URL）作为基准
                    # 并确保以 / 结尾，这样相对路径才能正确解析
                    base = page.url if page.url.endswith("/") else page.url + "/"
                    current_url = urljoin(base, next_a["href"])
                    print(f"  下一页: {current_url}")
                else:
                    current_url = None
            else:
                current_url = None

            page_num += 1
            if current_url:
                time.sleep(1)

        # 关闭浏览器
        browser.close()

    return all_quotes


# ========== 主程序 ==========
if __name__ == "__main__":
    BASE_URL = "https://quotes.toscrape.com/js"

    print("=" * 60)
    print("动态页面爬虫启动（Playwright 版）")
    print("=" * 60)

    quotes = scrape_quotes_with_playwright(BASE_URL, max_pages=5)

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"爬取完成！共获取 {len(quotes)} 条名言")
    print(f"{'=' * 60}")

    for i, q in enumerate(quotes[:5], 1):
        print(f"\n  [{i}] {q['text'][:70]}...")
        print(f"      作者：{q['author']}")
        print(f"      标签：{', '.join(q['tags'])}")

    # 保存数据
    with open("quotes_dynamic.json", "w", encoding="utf-8") as f:
        json.dump(quotes, f, ensure_ascii=False, indent=2)
    print(f"\n数据已保存到 quotes_dynamic.json")
