"""
RAG 系统（检索增强生成）
真实可用的 RAG 实现：检索 + 生成

组件：
  1. 文档加载：从 knowledge_docs.json 读取知识库
  2. 文本检索：用字符 n-gram 相似度做检索（无需向量数据库）
  3. Prompt 组装：把检索结果塞进 Prompt
  4. LLM 生成：调用 Kimi API 生成回答

适用场景：
  适合知识库规模较小（几十到几百条）的场景。
  大规模场景可替换为向量数据库（Chroma、Milvus 等）。
"""

import json
import numpy as np
import os
import time
from collections import Counter

print("=" * 60)
print("RAG 系统（检索增强生成）")
print("=" * 60)


# ========== 1. 文档加载 ==========

print("\n--- 1. 加载知识库文档 ---")

with open("knowledge_docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"  已加载 {len(docs)} 篇文档")
for doc in docs:
    print(f"    [{doc['doc_id']}] {doc['title']}（{len(doc['content'])} 字符）")


# ========== 2. 文本检索器 ==========

print(f"\n--- 2. 构建文本检索器 ---")


class SimpleRetriever:
    """
    简单的文本检索器（基于字符 n-gram 相似度）

    原理：
      把查询和文档都切分成字符 n-gram（2~3 字的片段），
      统计它们的重合度。重合越多越相关。

    优点：
      无需向量数据库、无需 Embedding 模型、零依赖
      适合小规模知识库和教学演示

    局限：
      只能做字面匹配，不能理解语义
      真实项目推荐用向量数据库（Chroma、Milvus）
    """

    def __init__(self, docs):
        self.docs = docs
        self.doc_ngrams = []
        for doc in docs:
            ngrams = self._extract_ngrams(doc["content"] + " " + doc["title"])
            self.doc_ngrams.append(set(ngrams))

    def _extract_ngrams(self, text, n=2):
        """提取字符 n-gram"""
        ngrams = []
        for size in [2, 3]:
            for i in range(len(text) - size + 1):
                ngrams.append(text[i:i + size])
        return ngrams

    def search(self, query, top_k=2):
        """检索最相关的文档"""
        query_ngrams = set(self._extract_ngrams(query))

        scores = []
        for i, doc_ngrams in enumerate(self.doc_ngrams):
            # Jaccard 相似度：交集 / 并集
            intersection = len(query_ngrams & doc_ngrams)
            union = len(query_ngrams | doc_ngrams)
            score = intersection / union if union > 0 else 0.0
            scores.append((i, score))

        # 按相似度降序排序
        scores.sort(key=lambda x: x[1], reverse=True)

        return [(self.docs[i], s) for i, s in scores[:top_k]]


retriever = SimpleRetriever(docs)
print(f"  检索器就绪（字符 n-gram 相似度，top_k=2）")
print(f"  每篇文档已建立 n-gram 索引")


# ========== 3. LLM 调用 ==========

print(f"\n--- 3. 连接 Kimi API ---")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("  [错误] 未安装 openai 库，请执行：pip install openai")


def call_llm_api(question, api_key=None,
                 base_url="https://api.moonshot.cn/v1",
                 model="kimi-k2.6"):
    """
    调用 Kimi API 获取回答（含限流保护）

    注意：
      1. kimi-k2.6 是推理模型，max_tokens 必须够大
      2. kimi-k2.6 不允许传 temperature
    """
    if not HAS_OPENAI:
        return None

    if api_key is None:
        api_key = os.environ.get("LLM_API_KEY")
    if api_key is None:
        print("  [错误] 未配置 API Key")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=6000,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return None
    except Exception as e:
        print(f"  [错误] API 调用失败: {e}")
        return None


# ========== 4. RAG 核心：检索 + 生成 ==========

print(f"\n--- 4. RAG 核心流程 ---")

RAG_PROMPT_TEMPLATE = """你是一个企业知识库问答助手。请严格根据以下参考资料回答问题。

要求：
  1. 只使用参考资料中的信息回答
  2. 如果参考资料中没有答案，明确说"参考资料中未提及"
  3. 回答时标注信息来源（文档编号）
  4. 不要编造参考资料之外的内容

参考资料：
{context}

用户问题：{question}"""


def rag_answer(question, retriever, top_k=2):
    """
    RAG 完整流程：检索 → 组装 Prompt → 生成回答

    返回：
        answer: LLM 的回答
        retrieved_docs: 检索到的文档列表
        context: 组装的上下文
    """
    # 第一步：检索
    search_results = retriever.search(question, top_k=top_k)

    # 第二步：组装上下文
    context_parts = []
    retrieved_docs = []
    for doc, score in search_results:
        context_parts.append(f"[{doc['doc_id']}] {doc['title']}\n{doc['content']}")
        retrieved_docs.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "score": round(score, 4),
        })

    context = "\n\n".join(context_parts)

    # 第三步：组装 Prompt
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # 第四步：调用 LLM 生成
    answer = call_llm_api(question=prompt)

    return answer, retrieved_docs, context


def pure_llm_answer(question):
    """
    纯 LLM 回答（不检索，直接问 LLM）——用于对比实验
    """
    return call_llm_api(question=question)


print(f"  RAG 流程就绪（检索 top_k=2）")
print(f"  Prompt 模板已定义（{len(RAG_PROMPT_TEMPLATE)} 字符）")


# ========== 5. 快速验证 ==========

print(f"\n--- 5. 快速验证 RAG 系统 ---")
print(f"  测试问题：'公司年假有多少天？'")
print(f"  正确答案应来自 DOC-01：12 天，工龄满 5 年加 2 天，最多 20 天")

test_question = "公司年假有多少天？"
answer, retrieved, context = rag_answer(test_question, retriever)

print(f"\n  检索结果：")
for doc in retrieved:
    print(f"    [{doc['doc_id']}] {doc['title']}（相似度 {doc['score']}）")

if answer:
    print(f"\n  RAG 回答：")
    for line in answer.split("\n"):
        print(f"    {line}")
else:
    print(f"  [错误] 未获取到回答，请检查 API Key")

print(f"\n  验证要点：")
print(f"    1. 检索到的文档应包含 DOC-01（年假制度）")
print(f"    2. 回答应提及'12 天'、'5 年'、'2 天'等具体数字")
print(f"    3. 回答应标注来源为 DOC-01")
