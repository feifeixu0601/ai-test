from bs4 import BeautifulSoup

# 定义一段 HTML 字符串，后面所有代码示例都基于它
HTML = """
<div class="job-list">
    <div class="job-item">
        <h3>AI测试工程师</h3>
        <span class="company">某科技公司</span>
        <span class="salary">25-40K</span>
    </div>
    <div class="job-item">
        <h3>高级AI测试开发</h3>
        <span class="company">某互联网大厂</span>
        <span class="salary">35-60K</span>
    </div>
</div>
"""

# 把 HTML 字符串传入 BeautifulSoup，指定解析器为 lxml
soup = BeautifulSoup(HTML, "lxml")

# soup 对象就代表了整个 HTML 文档
# 你可以对它调用各种查找方法
print("BeautifulSoup 对象创建成功！")


# ========== find() 找第一个匹配的元素 ==========

print("\n--- find() 演示 ---")

# 按标签名查找
first_h3 = soup.find("h3")
print("第一个 h3:", first_h3.text)         # AI测试工程师

# 按 class 查找（注意用 class_ 带下划线，因为 class 是 Python 关键字）
first_company = soup.find("span", class_="company")
print("第一个公司:", first_company.text)    # 某科技公司

# 按多个条件组合查找
first_salary = soup.find("span", class_="salary")
print("第一个薪资:", first_salary.text)     # 25-40K


# ========== find_all() 找所有匹配的元素 ==========

print("\n--- find_all() 演示 ---")

# 查找所有 h3 标签
all_h3 = soup.find_all("h3")
print("找到", len(all_h3), "个 h3 标签")

for h3 in all_h3:
    print("  h3 文本:", h3.text)

# 查找所有 class="salary" 的 span
all_salary = soup.find_all("span", class_="salary")
for s in all_salary:
    print("  薪资:", s.text)


# ========== select() CSS 选择器查找 ==========

print("\n--- select() 演示 ---")

# 选择 class="job-item" 下的所有 h3
titles = soup.select(".job-item h3")
for t in titles:
    print("  职位:", t.text)

# 选择 class="job-item" 下的所有 class="salary"
salaries = soup.select(".job-item .salary")
for s in salaries:
    print("  薪资:", s.text)

# ========== 4.5 提取属性值 ==========

print("\n--- 提取属性值演示 ---")

html2 = """
<a href="/job/001" data-id="1001">AI测试工程师</a>
<img src="https://example.com/logo.png" alt="公司logo">
"""
soup2 = BeautifulSoup(html2, "lxml")

# 提取 a 标签的 href 属性
link = soup2.find("a")
print("href:", link["href"])             # /job/001
print("href (get方法):", link.get("href")) # /job/001
print("data-id:", link.get("data-id"))    # 1001

# 提取 img 标签的 src 属性
img = soup2.find("img")
print("src:", img["src"])               # https://example.com/logo.png
print("alt:", img.get("alt"))           # 公司logo

# 获取所有属性（返回字典）
print("所有属性:", link.attrs)
# {'href': '/job/001', 'data-id': '1001'}

# ========== 4.6 嵌套查找（核心模式） ==========

print("\n--- 嵌套查找演示 ---")

html3 = """
<div class="job-list">
    <div class="job-item">
        <h3>AI测试工程师</h3>
        <div class="info">
            <span class="company">某科技公司</span>
            <span class="salary">25-40K</span>
        </div>
        <p class="desc">负责AI模型的功能测试、性能测试、鲁棒性测试</p>
        <div class="tags">
            <span class="tag">Python</span>
            <span class="tag">机器学习</span>
            <span class="tag">自动化测试</span>
        </div>
    </div>
    <div class="job-item">
        <h3>大模型测试专家</h3>
        <div class="info">
            <span class="company">某互联网大厂</span>
            <span class="salary">40-70K</span>
        </div>
        <p class="desc">负责大语言模型评测体系建设，设计自动化评测方案</p>
        <div class="tags">
            <span class="tag">LLM</span>
            <span class="tag">NLP</span>
            <span class="tag">Prompt Engineering</span>
        </div>
    </div>
</div>
"""

soup3 = BeautifulSoup(html3, "lxml")

# 第一步：找到所有大容器
items = soup3.find_all("div", class_="job-item")

jobs = []
for item in items:
    # 第二步：从当前容器中查找子元素
    title   = item.find("h3").text.strip()
    company = item.find("span", class_="company").text.strip()
    salary  = item.find("span", class_="salary").text.strip()
    desc    = item.find("p", class_="desc").text.strip()

    # 提取所有标签（一个职位可能有多个 tag）
    tag_elements = item.find_all("span", class_="tag")
    tags = [tag.text for tag in tag_elements]

    # 第三步：组装成字典存入列表
    jobs.append({
        "职位": title,
        "公司": company,
        "薪资": salary,
        "描述": desc,
        "技能要求": tags
    })

# 打印结果
for job in jobs:
    print(f"\n  职位：{job['职位']}")
    print(f"  公司：{job['公司']}")
    print(f"  薪资：{job['薪资']}")
    print(f"  描述：{job['描述']}")
    print(f"  技能：{'、'.join(job['技能要求'])}")

# ========== 4.7 .text 和 .string 的区别 ==========

print("\n--- .text 和 .string 的区别 ---")

html4 = "<p>  价格：<span>99元</span>  </p>"
soup4 = BeautifulSoup(html4, "lxml")
p = soup4.find("p")

# .text（或 .get_text()）—— 获取所有子节点的文本，拼接在一起
print("text:", repr(p.text))                    # '  价格：99元  '
print("get_text(strip=True):", repr(p.get_text(strip=True)))  # '价格：99元'

# .string —— 只有当标签内只有纯文本时才有值
print("string:", p.string)    # None（因为 p 里还有子标签 span，不是纯文本）

# ========== 4.8 常见坑与调试技巧 ==========

print("\n--- 常见坑演示 ---")

# 坑1：class 属性有多个值
html5 = '<div class="job-item featured hot">好职位</div>'
soup5 = BeautifulSoup(html5, "lxml")

# class_ 是"包含匹配"，不是精确匹配
result = soup5.find("div", class_="job-item")
print("包含匹配能找到:", result.text)  # 好职位

# 也可以用 CSS 选择器精确匹配多个 class
result = soup5.select_one("div.job-item.featured.hot")
print("CSS 选择器也能找到:", result.text)  # 好职位

# 坑2：find 找不到元素返回 None
item = soup.find("div", class_="不存在的class")
print("找不到时返回:", item)  # None

# 错误写法：直接调用 .text 会报 AttributeError
# print(item.text)    # None 没有 .text 属性，程序崩溃

# 正确写法：先判断是否找到
if item:
    print(item.text)
else:
    print("元素未找到，程序没有崩溃")

# 调试技巧：如果你不确定 HTML 结构，用 prettify() 格式化输出
print("\n--- prettify() 调试技巧 ---")
print(soup.prettify())
