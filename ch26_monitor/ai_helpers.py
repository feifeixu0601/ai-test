"""AI 测试工具：API 调用"""
import time, os

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("pip install openai")

_MIN_INTERVAL = 21
_last_call_time = [0.0]

def call_simple(question, model="kimi-k2.6", base_url="https://api.moonshot.cn/v1", **kw):
    """单轮调用（含全局限流）"""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 LLM_API_KEY")

    elapsed = time.time() - _last_call_time[0]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=6000,
        )
        _last_call_time[0] = time.time()
        return resp.choices[0].message.content if resp.choices else None
    except Exception as e:
        _last_call_time[0] = time.time()
        print(f"[API 错误] {e}")
        return None