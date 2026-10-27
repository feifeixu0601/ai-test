import requests

# requests.get() 就相当于浏览器发送了一个 GET 请求
response = requests.get("https://httpbin.org/get")

# response 是一个 Response 对象，包含了服务器返回的所有信息

# 状态码 —— 服务器告诉你请求的结果
print(response.status_code)       # 200，表示成功

# 响应头 —— 服务器返回的附加信息
print(response.headers["Content-Type"])  # application/json

# 响应体 —— 你要的数据
print(response.text)              # 字符串形式的响应内容

# 如果响应是 JSON 格式，可以直接转成 Python 字典
print(response.json())            # dict 类型


# ========== 2.3 带参数的 GET 请求 ==========

# 方式一：直接拼在 URL 里（不推荐）
# 问题：中文需要手动 URL 编码，参数多了容易拼错
response = requests.get("https://httpbin.org/get?name=张三&age=25")

# 方式二：用 params 参数（推荐）
params = {
    "name": "张三",
    "age": 25
}
response = requests.get("https://httpbin.org/get", params=params)

# 看看实际请求的 URL 是什么样的
print(response.url)
# https://httpbin.org/get?name=%E5%BC%A0%E4%B8%89&age=25
#
# requests 自动帮你做了 URL 编码：
# "张三" 被编码成了 "%E5%BC%A0%E4%B8%89"
# 这就是为什么推荐用 params 而不是手动拼 URL

# ========== 2.4 请求头 ==========

# 不加请求头试试（有些网站会拒绝）
# response = requests.get("https://www.zhihu.com")
# print("不带头:", response.status_code)    # 大概率 403，被拒绝了

# 加上请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
# response = requests.get("https://www.zhihu.com", headers=headers)
# print("带头:", response.status_code)      # 200，成功了

# 用 httpbin 演示请求头的效果
response = requests.get("https://httpbin.org/headers", headers=headers)
print("\n服务器看到的请求头：")
print(response.text)


# ========== 2.5 POST 请求 ==========

# 发送表单数据（类似你在网页上填写表单然后点提交）
form_data = {
    "username": "test_user",
    "password": "123456"
}
response = requests.post("https://httpbin.org/post", data=form_data, headers=headers)
print("\n表单数据结果：")
print(response.json()["form"])
# {'username': 'test_user', 'password': '123456'}

# 发送 JSON 数据（前后端分离的网站常用）
json_data = {
    "model": "gpt-4",
    "prompt": "你好",
    "temperature": 0.7
}
response = requests.post("https://httpbin.org/post", json=json_data, headers=headers)
print("\nJSON数据结果：")
print(response.json()["json"])
# {'model': 'gpt-4', 'prompt': '你好', 'temperature': 0.7}


# ========== 2.6 Response 对象详解 ==========

response = requests.get("https://httpbin.org/get", headers=headers)

print("\n--- Response 对象属性 ---")

# 状态码
print("状态码:", response.status_code)         # 200

# 是否成功（状态码在 200-400 之间返回 True）
print("是否成功:", response.ok)                # True

# 响应头（字典类型）
print("Content-Type:", response.headers["Content-Type"])

# 响应体 —— 三种获取方式
print("text 类型:", type(response.text))       # <class 'str'>
print("content 类型:", type(response.content)) # <class 'bytes'>
print("json 类型:", type(response.json()))     # <class 'dict'>

# 编码
print("编码:", response.encoding)              # utf-8

# 实际请求的 URL（处理重定向后的真实地址）
print("URL:", response.url)

# 请求耗时
print("耗时:", response.elapsed.total_seconds(), "秒")

# 看看 requests 实际发出去的请求头（调试用）
print("发出的请求头:", response.request.headers)


# ========== 2.7 超时与异常处理 ==========

def safe_request(url, max_retries=3, timeout=10):
    """
    安全的请求函数，带重试和异常处理
    这个函数会在第五部分的实战项目中用到

    参数：
        url: 请求地址
        max_retries: 最大重试次数
        timeout: 超时时间（秒）

    返回：
        成功返回 Response 对象，失败返回 None
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            print(f"  [重试 {attempt}/{max_retries}] 请求超时: {url}")

        except requests.exceptions.ConnectionError:
            print(f"  [重试 {attempt}/{max_retries}] 连接失败: {url}")

        except requests.exceptions.HTTPError as e:
            print(f"  [放弃] HTTP 错误 {e.response.status_code}: {url}")
            return None

        except requests.exceptions.TooManyRedirects:
            print(f"  [放弃] 重定向次数过多: {url}")
            return None

        except Exception as e:
            print(f"  [重试 {attempt}/{max_retries}] 未知错误: {e}")

    print(f"  [放弃] 达到最大重试次数: {url}")
    return None


# 测试 safe_request 函数
print("\n--- safe_request 测试 ---")
result = safe_request("https://httpbin.org/get")
if result:
    print(f"请求成功，状态码: {result.status_code}")
else:
    print("请求失败")
