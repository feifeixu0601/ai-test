"""
多轮对话系统（带上下文管理）

核心机制：
  每次用户发言时，把完整对话历史拼进 messages 发给 LLM
  LLM 基于历史上下文生成回答
  回答后更新对话历史

测试价值：
  - 可以观察 LLM 在不同上下文长度下的表现
  - 可以验证记忆、连贯、一致、状态等维度
  - 支持自定义对话序列（测试用例驱动）
"""

import json
import os
import time
from datetime import datetime

print("=" * 60)
print("多轮对话系统（带上下文管理）")
print("=" * 60)


# ========== 1. 连接 Kimi API ==========

print("\n--- 1. 连接 Kimi API ---")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("  [错误] 未安装 openai 库，请执行：pip install openai")

# 全局限流（多轮对话一次任务内连发多条，限流必须放在 API 层内部）
_MIN_INTERVAL = 21
_last_call_time = [0.0]


def call_llm_api(messages, api_key=None,
                 base_url="https://api.moonshot.cn/v1",
                 model="kimi-k2.6"):
    """
    调用 Kimi API（多轮对话版）

    参数：
        messages: 消息列表，格式 [{"role": "user/assistant", "content": "..."}]
    """
    if not HAS_OPENAI:
        return None

    if api_key is None:
        api_key = os.environ.get("LLM_API_KEY")
    if api_key is None:
        print("  [错误] 未配置 API Key")
        return None

    elapsed = time.time() - _last_call_time[0]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=6000,
        )
        _last_call_time[0] = time.time()
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return None
    except Exception as e:
        _last_call_time[0] = time.time()
        print(f"  [错误] API 调用失败: {e}")
        return None


print(f"  API 连接就绪")


# ========== 2. 对话管理器 ==========

print(f"\n--- 2. 对话管理器 ---")

SYSTEM_PROMPT = """你是一个友好的对话助手。
请根据对话历史回答用户的问题。
如果用户的问题不清晰，请主动追问澄清。
保持回答简洁、准确、连贯。
注意记住用户之前说过的信息。"""


class DialogManager:
    """
    多轮对话管理器

    核心功能：
      1. 维护对话历史（role + content 的消息列表）
      2. 每次用户发言时拼接历史发给 LLM
      3. 记录完整的对话轨迹（用于测试分析）
    """

    def __init__(self, llm_func):
        self.llm_func = llm_func
        self.messages = []      # 对话历史（发给 API 的格式）
        self.dialog_log = []    # 完整对话日志（用于测试分析）

    def send(self, user_message):
        """发送用户消息，获取 AI 回答"""
        self.messages.append({"role": "user", "content": user_message})

        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.messages

        ai_response = self.llm_func(api_messages)

        if ai_response:
            self.messages.append({"role": "assistant", "content": ai_response})
            self.dialog_log.append({
                "turn": len(self.dialog_log) + 1,
                "user": user_message,
                "assistant": ai_response,
            })
        else:
            self.dialog_log.append({
                "turn": len(self.dialog_log) + 1,
                "user": user_message,
                "assistant": "[API 调用失败]",
            })

        return ai_response or "[API 调用失败]"

    def reset(self):
        """重置对话（开始新对话，避免上下文污染）"""
        self.messages = []
        self.dialog_log = []

    def get_context_length(self):
        """获取当前上下文长度（字符数）"""
        return sum(len(m["content"]) for m in self.messages)


dialog_mgr = DialogManager(call_llm_api)
print(f"  对话管理器就绪")
print(f"  支持多轮对话、上下文管理、对话日志")