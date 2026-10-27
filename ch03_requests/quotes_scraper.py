"""
实战项目：爬取 quotes.toscrape.com 的名人名言
逐步构建，最终形成完整的爬虫程序
"""

import requests
from bs4 import BeautifulSoup
import json
import time


def get_page(url):
    """
    请求单个页面，返回 BeautifulSoup 对象
    请求失败返回 None
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except requests.exceptions.RequestException as e:
            print(f"  [重试 {attempt}/3] 请求失败: {e}")
            time.sleep(2)

    print(f"  [放弃] {url}")
    return None


def parse_quotes(soup):
    """
    从 BeautifulSoup 对象中提取当前页的所有名言数据
    返回一个列表，每个元素是一条名言的字典
    """
    quotes = []
    quote_divs = soup.find_all("div", class_="quote")

    for div in quote_divs:
        # 提取名言文本
        text_el = div.find("span", class_="text")
        if not text_el:
            continue
        text = text_el.get_text(strip=True)

        # 提取作者
        author_el = div.find("small", class_="author")
        author = author_el.get_text(strip=True) if author_el else "未知"

        # 提取所有标签
        tag_els = div.find_all("a", class_="tag")
        tags = [tag.get_text(strip=True) for tag in tag_els]

        quotes.append({
            "text": text,
            "author": author,
            "tags": tags
        })

    return quotes


def get_next_url(soup, base_url):
    """
    查找下一页的链接
    有下一页返回完整 URL，没有返回 None
    """
    next_li = soup.find("li", class_="next")
    if next_li:
        next_a = next_li.find("a")
        if next_a and next_a.get("href"):
            return base_url.rstrip("/") + next_a["href"]
    return None


def scrape_all_quotes(base_url, max_pages=10):
    """
    主函数：爬取所有页面的名言数据
    """
    all_quotes = []
    current_url = base_url
    page = 1

    while current_url and page <= max_pages:
        print(f"\n[第 {page} 页] {current_url}")

        soup = get_page(current_url)
        if not soup:
            break

        # 提取当前页数据
        quotes = parse_quotes(soup)
        all_quotes.extend(quotes)
        print(f"  提取到 {len(quotes)} 条名言，累计 {len(all_quotes)} 条")

        # 查找下一页
        current_url = get_next_url(soup, base_url)

        # 礼貌等待，避免请求过快
        if current_url:
            time.sleep(1)

        page += 1

    return all_quotes


# ========== 主程序 ==========
if __name__ == "__main__":
    BASE_URL = "https://quotes.toscrape.com"

    print("=" * 60)
    print("开始爬取 quotes.toscrape.com")
    print("=" * 60)

    quotes = scrape_all_quotes(BASE_URL, max_pages=5)

    # 打印结果摘要
    print(f"\n{'=' * 60}")
    print(f"爬取完成！共获取 {len(quotes)} 条名言")
    print(f"{'=' * 60}")

    # 打印前 5 条
    print("\n前 5 条名言预览：")
    for i, q in enumerate(quotes[:5], 1):
        print(f"\n  [{i}] {q['text'][:70]}...")
        print(f"      作者：{q['author']}")
        print(f"      标签：{', '.join(q['tags'])}")

    # 统计信息
    authors = {}
    all_tags = []
    for q in quotes:
        authors[q["author"]] = authors.get(q["author"], 0) + 1
        all_tags.extend(q["tags"])

    print(f"\n作者数量：{len(authors)} 位")
    print(f"标签数量：{len(set(all_tags))} 个")

    # 出现最多的作者 Top 5
    top_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n出现最多的作者 Top 5：")
    for author, count in top_authors:
        print(f"  {author}: {count} 条")

    # 保存到 JSON 文件
    output_file = "quotes.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(quotes, f, ensure_ascii=False, indent=2)
    print(f"\n数据已保存到 {output_file}")
