"""
API 连通性测试
跑通了再批量使用，没跑通就别往下走了

踩坑记录：
  1. kimi-k2.6 是推理模型，会把 token 花在"思考"上
     → max_tokens 必须调大（建议 2000+），否则 content 为空
  2. kimi-k2.6 只允许 temperature=1 → 不要传 temperature 参数
  3. 回答在 msg.content 里，思考过程在 msg.reasoning_content 里
"""

import os

try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai 库：pip install openai")
    exit()

api_key = os.environ.get("LLM_API_KEY")
if api_key is None:
    print("[错误] 未设置环境变量 LLM_API_KEY")
    print("Windows PowerShell 执行：$env:LLM_API_KEY = 'sk-你的密钥'")
    exit()

BASE_URL = "https://api.moonshot.cn/v1"
MODEL = "kimi-k2.6"

print("正在调用 Kimi API...")
print(f"API Key: {api_key[:8]}****（已隐藏）")
print(f"base_url: {BASE_URL}")
print(f"model: {MODEL}")
print()

client = OpenAI(api_key=api_key, base_url=BASE_URL)

# 第 0 步：列出账号可用的模型
try:
    print("可用模型列表：")
    for m in client.models.list().data:
        print("  ", m.id)
    print()
except Exception as e:
    print(f"[WARN] 拉取模型列表失败（不影响下面测试）：{e}")

# 第 1 步：测试 API 连通性
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}],
        max_tokens=2000,  # 必须调大！推理模型思考很吃 token
    )

    choice = response.choices[0]
    msg = choice.message

    print(f"finish_reason: {choice.finish_reason}")
    print("=" * 50)

    # 推理模型的思考过程（可能在 reasoning_content 里）
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        print("【思考过程（前 200 字）】:")
        print(reasoning[:200])
        print("=" * 50)

    # 最终回答
    if msg.content:
        print("[PASS] API 调用成功！")
        print(f"【回答】: {msg.content}")
    else:
        if choice.finish_reason == "length":
            print("[FAIL] 回答为空，原因：max_tokens 不够，模型思考把 token 用完了")
            print("       解决：把 max_tokens 调大（如 4000）")
        else:
            print("[FAIL] API 返回空内容")

except Exception as e:
    print(f"[FAIL] API 调用失败：{e}")
    print()
    print("排查口诀：")
    print("  404 + 'Not found the model XXX' → 模型名不对，用上面列表里的名字")
    print("  401 → API Key 不对")
    print("  429 → 限流（RPM=3），等 1 分钟再试")
    print("  content 为空 + finish_reason=length → max_tokens 太小，调大")