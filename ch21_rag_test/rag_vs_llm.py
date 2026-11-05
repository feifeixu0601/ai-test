"""
RAG vs 纯 LLM 对比实验
同样的问题，对比两种方式的回答质量

实验设计：
  用知识库中有答案的问题（RAG-01~06）
  分别用 RAG 和纯 LLM 回答
  对比回答中的关键信息是否正确
"""

import numpy as np
import json
import time
import os
from datetime import datetime

print("=" * 60)
print("RAG vs 纯 LLM 对比实验")
print("=" * 60)


# ========== 1. 导入 RAG 系统 ==========

print("\n--- 1. 加载 RAG 系统 ---")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from rag_system import retriever, rag_answer, pure_llm_answer

print(f"  RAG 系统就绪")


# ========== 2. 对比用例 ==========

print(f"\n--- 2. 对比用例（用知识库中有标准答案的问题）---")

with open("rag_compare_cases.json", "r", encoding="utf-8") as f:
    compare_cases = json.load(f)

print(f"  对比用例数：{len(compare_cases)}")
print(f"  说明：这些问题的标准答案在知识库中（虚构公司制度）")
print(f"        纯 LLM 不知道这些公司内部规定，大概率会编造或用通用知识回答")


# ========== 3. 执行对比 ==========

print(f"\n--- 3. 执行对比实验 ---")
print(f"  限流控制：RAG 和纯 LLM 各调 1 次 API，每次间隔 21 秒")
print(f"  预计总耗时：约 2 分钟")

compare_results = []

for tc in compare_cases:
    print(f"\n  [{tc['id']}] 问题：{tc['question']}")

    # RAG 回答
    print(f"    正在获取 RAG 回答...")
    rag_ans, retrieved, _ = rag_answer(tc["question"], retriever)
    print(f"    RAG 检索到：{[d['doc_id'] for d in retrieved]}")
    time.sleep(21)

    # 纯 LLM 回答
    print(f"    正在获取纯 LLM 回答...")
    pure_ans = pure_llm_answer(tc["question"])
    time.sleep(21)

    # 分析对比
    # 按事实点评分：同义写法归为同一个事实点，命中任意变体算命中该事实点
    rag_points_hit = sum(
        1 for fp in tc["fact_points"]
        if rag_ans and any(v in rag_ans for v in fp["variants"])
    )
    pure_points_hit = sum(
        1 for fp in tc["fact_points"]
        if pure_ans and any(v in pure_ans for v in fp["variants"])
    )

    total_points = len(tc["fact_points"])
    rag_score = round(rag_points_hit / max(total_points, 1), 2)
    pure_score = round(pure_points_hit / max(total_points, 1), 2)

    compare_results.append({
        "id": tc["id"],
        "question": tc["question"],
        "rag_score": rag_score,
        "pure_score": pure_score,
        "rag_points": rag_points_hit,
        "pure_points": pure_points_hit,
        "total_points": total_points,
        "rag_answer": rag_ans[:80] if rag_ans else "无回答",
        "pure_answer": pure_ans[:80] if pure_ans else "无回答",
    })

    # 打印对比
    winner = "RAG" if rag_score > pure_score else ("纯LLM" if pure_score > rag_score else "持平")
    print(f"\n    RAG 回答（前80字）：{rag_ans[:80] if rag_ans else '无'}...")
    print(f"    纯LLM 回答（前80字）：{pure_ans[:80] if pure_ans else '无'}...")
    print(f"    事实点命中：RAG {rag_points_hit}/{total_points} vs "
          f"纯LLM {pure_points_hit}/{total_points} → {winner}胜")

print(f"\n  对比实验完成！")


# ========== 4. 对比报告 ==========

print(f"\n{'=' * 60}")
print(f"RAG vs 纯 LLM 对比报告")
print(f"{'=' * 60}")

print(f"\n  {'用例':<8s} {'RAG得分':>8s} {'纯LLM得分':>10s} {'优势方':>10s}")
print(f"  {'─' * 40}")

for r in compare_results:
    winner = "RAG" if r["rag_score"] > r["pure_score"] else ("纯LLM" if r["pure_score"] > r["rag_score"] else "持平")
    print(f"  {r['id']:<8s} {r['rag_score']:>8.2f} {r['pure_score']:>10.2f} {winner:>10s}")

avg_rag = np.mean([r["rag_score"] for r in compare_results])
avg_pure = np.mean([r["pure_score"] for r in compare_results])

print(f"\n  平均得分：RAG {avg_rag:.2f} vs 纯 LLM {avg_pure:.2f}")
print(f"  提升幅度：{(avg_rag - avg_pure) * 100:+.1f}%")

print(f"\n  结论：")
if avg_rag > avg_pure + 0.1:
    print(f"    RAG 显著优于纯 LLM——正确地检索到了公司制度并用在回答中")
    print(f"    纯 LLM 无法回答公司内部规定（训练数据中没有），只能给出通用答案或编造")
elif avg_rag > avg_pure:
    print(f"    RAG 略优于纯 LLM")
else:
    print(f"    两者表现接近（可能知识库内容和通用知识重叠较多）")

print(f"\n  教学价值：")
print(f"    1. 公司内部规定（年假 12 天、餐补 150 元）是 LLM 训练数据中没有的")
print(f"    2. 纯 LLM 只能给出通用答案（'年假 5~15 天'）或者直接说不知道")
print(f"    3. RAG 通过检索知识库，找到了准确答案并标注来源")
print(f"    4. 这就是 RAG 在企业场景中的核心价值：让 AI 知道'你们公司'的规定")

# 保存报告
report = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "kimi-k2.6",
    "avg_rag": round(float(avg_rag), 2),
    "avg_pure": round(float(avg_pure), 2),
    "results": compare_results,
}

with open("rag_vs_llm_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n  对比报告已保存到 rag_vs_llm_report.json")
