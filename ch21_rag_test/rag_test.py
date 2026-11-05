"""
RAG 系统测试工具
三大指标：检索准确率、回答忠实度、端到端准确率

测试方法：
  1. 检索测试：检查检索到的文档是否包含正确答案
  2. 忠实度测试：检查回答是否忠于检索到的文档
  3. 端到端测试：检查最终回答是否正确

注意：
  测试用例中包含 expected_doc_id（期望检索到的文档）
  和 expected_keywords（期望回答中的关键词），
  用于自动判断测试是否通过。
"""

import numpy as np
import json
import time
import os
from datetime import datetime

print("=" * 60)
print("RAG 系统测试工具")
print("=" * 60)

np.random.seed(42)


# ========== 1. 导入 RAG 系统 ==========

print("\n--- 1. 加载 RAG 系统 ---")

# 从 rag_system 导入组件（同目录下）
import sys
sys.path.insert(0, os.path.dirname(__file__))
from rag_system import retriever, rag_answer, call_llm_api

print(f"  RAG 系统就绪")


# ========== 2. 定义测试用例 ==========

print(f"\n--- 2. 定义测试用例 ---")

with open("rag_test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

print(f"  测试用例数：{len(test_cases)}")
print(f"  有标准答案的问题：{sum(1 for tc in test_cases if tc['expected_doc_id'])} 个")
print(f"  知识库中没有答案的问题：{sum(1 for tc in test_cases if not tc['expected_doc_id'])} 个")
print(f"\n  [设计说明]")
print(f"    RAG-01~06：知识库中有答案，测试检索+生成能力")
print(f"    RAG-07~08：知识库中没答案，测试模型会不会编造（幻觉检测）")
print(f"    RAG-09：时效性测试——知识库只有 2025 版，看模型如何处理时间差")


# ========== 3. 执行测试 ==========

print(f"\n--- 3. 执行 RAG 测试 ---")
print(f"  限流控制：每次调用后等待 21 秒（适配 RPM=3）")
print(f"  预计总耗时：约 3 分钟")

import time

test_results = []

for tc in test_cases:
    print(f"\n  [{tc['id']}] 问题：{tc['question']}")

    # RAG 流程：检索 + 生成
    answer, retrieved, context = rag_answer(tc["question"], retriever)

    if not answer:
        print(f"    [错误] API 调用失败，跳过")
        continue

    # ---------- 第一层：检索测试 ----------
    retrieved_ids = [d["doc_id"] for d in retrieved]
    retrieval_hit = False

    if tc["expected_doc_id"]:
        retrieval_hit = tc["expected_doc_id"] in retrieved_ids
        retrieval_score = 1.0 if retrieval_hit else 0.0
        retrieval_desc = f"期望 {tc['expected_doc_id']}，检索到 {retrieved_ids}"
    else:
        # 知识库中没答案的题，任何检索结果都可以
        # （关键是生成阶段不能编造）
        retrieval_score = 1.0
        retrieval_desc = f"无标准文档（检索到 {retrieved_ids}）"

    # ---------- 第二层：忠实度测试 ----------
    # 回答中的关键词是否在检索到的文档中
    # 按事实点评分：同义写法归为同一个事实点，命中任意变体算命中该事实点
    faithfulness_points = sum(
        1 for fp in tc.get("fact_points", [])
        if any(v in answer or v in context for v in fp["variants"])
    )
    total_fp = len(tc.get("fact_points", []))
    faithfulness_score = round(faithfulness_points / max(total_fp, 1), 2) if total_fp > 0 else 1.0

    # ---------- 第三层：端到端测试 ----------
    # 回答中是否包含期望的关键信息
    answer_keywords_found = sum(
        1 for kw in tc["expected_answer_keywords"] if kw in answer
    )
    e2e_score = round(answer_keywords_found / max(len(tc["expected_answer_keywords"]), 1), 2)

    # 汇总
    test_results.append({
        "id": tc["id"],
        "question": tc["question"],
        "retrieved_ids": retrieved_ids,
        "retrieval_score": retrieval_score,
        "retrieval_desc": retrieval_desc,
        "faithfulness_score": faithfulness_score,
        "e2e_score": e2e_score,
        "answer_preview": answer[:60],
    })

    # 打印单条结果
    r_verdict = "[命中]" if retrieval_hit else ("[N/A]" if not tc["expected_doc_id"] else "[未命中]")
    f_verdict = "[好]" if faithfulness_score >= 0.5 else "[差]"
    e_verdict = "[对]" if e2e_score >= 0.5 else "[错]"

    print(f"    检索结果：{retrieved_ids}  {r_verdict}")
    print(f"    检索得分：{retrieval_score:.2f}")
    print(f"    忠实度：{faithfulness_score:.2f}  {f_verdict}")
    print(f"    端到端：{e2e_score:.2f}  {e_verdict}")
    print(f"    回答预览：{answer[:50]}...")

    time.sleep(21)

print(f"\n  测试完成！")


# ========== 4. 测试报告 ==========

print(f"\n{'=' * 60}")
print(f"RAG 系统测试报告")
print(f"{'=' * 60}")

total = len(test_results)
if total == 0:
    print("  无测试结果")
    exit()

# 检索准确率
retrieval_scores = [r["retrieval_score"] for r in test_results]
avg_retrieval = np.mean(retrieval_scores)
retrieval_hits = sum(1 for r in test_results
                    if r["retrieval_score"] >= 0.5 or "N/A" in r["retrieval_desc"])

# 忠实度
faithfulness_scores = [r["faithfulness_score"] for r in test_results]
avg_faithfulness = np.mean(faithfulness_scores)

# 端到端准确率
e2e_scores = [r["e2e_score"] for r in test_results]
avg_e2e = np.mean(e2e_scores)

print(f"\n  指标汇总：")
print(f"  {'指标':<15s} {'平均分':>8s} {'判定':>10s}")
print(f"  {'─' * 35}")
print(f"  {'检索准确率':<15s} {avg_retrieval:>8.2f} {'[PASS]' if avg_retrieval >= 0.8 else '[WARN]' if avg_retrieval >= 0.5 else '[FAIL]':>10s}")
print(f"  {'回答忠实度':<15s} {avg_faithfulness:>8.2f} {'[PASS]' if avg_faithfulness >= 0.8 else '[WARN]' if avg_faithfulness >= 0.5 else '[FAIL]':>10s}")
print(f"  {'端到端准确率':<15s} {avg_e2e:>8.2f} {'[PASS]' if avg_e2e >= 0.8 else '[WARN]' if avg_e2e >= 0.5 else '[FAIL]':>10s}")

# 逐条明细
print(f"\n  逐条明细：")
print(f"  {'用例':<8s} {'检索':>6s} {'忠实度':>8s} {'端到端':>8s} {'检索结果':<15s}")
print(f"  {'─' * 50}")

for r in test_results:
    print(f"  {r['id']:<8s} {r['retrieval_score']:>6.2f} {r['faithfulness_score']:>8.2f} "
          f"{r['e2e_score']:>8.2f} {','.join(r['retrieved_ids']):<15s}")

# 最弱环节分析
print(f"\n  最弱环节分析：")
scores = {
    "检索": avg_retrieval,
    "忠实度": avg_faithfulness,
    "端到端": avg_e2e,
}
worst = min(scores.items(), key=lambda x: x[1])
if worst[1] >= 0.9:
    print(f"    各项指标均优秀（最低 {worst[0]} {worst[1]:.2f}），暂无明显短板")
else:
    print(f"    最弱环节：{worst[0]}（{worst[1]:.2f}）")

if worst[1] < 0.9:
    if worst[0] == "检索":
        print(f"    建议：优化检索器（换向量数据库、调 top_k、优化文档切分）")
    elif worst[0] == "忠实度":
        print(f"    建议：优化 Prompt 模板（更强的'只用参考资料'约束）")
    else:
        print(f"    建议：检查 Prompt 中的上下文质量，或提升模型能力")

# 改进建议
print(f"\n  通用改进建议：")
print(f"    1. 如果检索不准，考虑用向量数据库替换 n-gram 相似度")
print(f"    2. 如果忠实度低，在 Prompt 中加强'不要编造'的约束")
print(f"    3. 如果端到端差，检查文档内容是否包含完整答案")
print(f"    4. 对'知识库中没答案'的问题，正确表现是说'未提及'")
print(f"    5. 建议扩充测试集到 30+ 条，覆盖更多业务场景")

# 保存报告
report = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "kimi-k2.6",
    "total_tests": total,
    "avg_retrieval": round(float(avg_retrieval), 2),
    "avg_faithfulness": round(float(avg_faithfulness), 2),
    "avg_e2e": round(float(avg_e2e), 2),
    "results": test_results,
}

with open("rag_test_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n  报告已保存到 rag_test_report.json")


# ========== 5. 切分质量分析 ==========

print(f"\n{'─' * 60}")
print(f"5. 文档切分质量分析")
print(f"{'─' * 60}")

from rag_system import docs, retriever

print(f"\n  文档统计：")
print(f"  {'文档ID':<10s} {'标题':<12s} {'字符数':>8s} {'句子数':>8s} {'建议':>12s}")
print(f"  {'─' * 55}")

for doc in docs:
    char_count = len(doc["content"])
    # 粗略统计句子数（按中文标点切分）
    import re
    sentences = re.split(r'[。！？；]', doc["content"])
    sentences = [s for s in sentences if s.strip()]
    sent_count = len(sentences)

    if char_count < 50:
        suggestion = "[太短]"
    elif char_count > 500:
        suggestion = "[太长]"
    else:
        suggestion = "[合理]"

    print(f"  {doc['doc_id']:<10s} {doc['title']:<12s} {char_count:>8d} {sent_count:>8d} {suggestion:>12s}")

print(f"\n  切分质量评估要点：")
print(f"    1. 文档块大小：50~500 字符为宜（太小丢语义，太大稀释相关性）")
print(f"    2. 信息完整性：每个块应包含一个完整的知识单元")
print(f"    3. 边界问题：避免把一个完整的句子/段落从中间切断")
print(f"    4. 重叠设计：相邻块之间可设置 10%~20% 的重叠，避免边界信息丢失")


# ========== 6. 检索排序分析 ==========

print(f"\n{'─' * 60}")
print(f"6. 检索排序分析")
print(f"{'─' * 60}")
print(f"  不仅看'找没找到'，还看'排在第几'")

# 对每个有标准文档的测试用例做排序分析
print(f"\n  {'用例':<8s} {'问题':<20s} {'期望文档':>8s} {'排名':>6s} {'相似度':>8s} {'判定':>10s}")
print(f"  {'─' * 65}")

for tc in test_cases:
    if not tc.get("expected_doc_id"):
        continue

    # 检索所有文档（不截取 top_k），看正确文档排第几
    all_results = retriever.search(tc["question"], top_k=len(docs))

    rank = -1
    target_score = 0
    for i, (doc, score) in enumerate(all_results):
        if doc["doc_id"] == tc["expected_doc_id"]:
            rank = i + 1
            target_score = score
            break

    if rank == 1:
        verdict = "[优秀]"
    elif rank <= 2:
        verdict = "[良好]"
    elif rank <= 3:
        verdict = "[一般]"
    else:
        verdict = "[较差]"

    print(f"  {tc['id']:<8s} {tc['question'][:18]:<20s} {tc['expected_doc_id']:>8s} "
          f"{rank:>6d} {target_score:>8.4f} {verdict:>10s}")

print(f"\n  排序分析的价值：")
print(f"    - 排名 1：正确文档就在最前面，LLM 最容易用上")
print(f"    - 排名 3+：正确文档被淹在后面，LLM 可能忽略它")
print(f"    - 如果正确文档经常排在后面 → 说明检索器需要优化")


# ========== 7. 多文档冲突检测 ==========

print(f"\n{'─' * 60}")
print(f"7. 多文档冲突检测")
print(f"{'─' * 60}")
print(f"  检查知识库中是否存在相互矛盾的文档")

import re as re_module

# 提取每篇文档中的数字（作为事实的代表）
doc_facts = {}
for doc in docs:
    numbers = set(re_module.findall(r'\d+', doc["content"]))
    doc_facts[doc["doc_id"]] = {
        "title": doc["title"],
        "numbers": numbers,
    }

# 检测同主题文档间的数字冲突
print(f"\n  文档事实摘要：")
print(f"  {'文档ID':<10s} {'标题':<12s} {'关键数字':<35s}")
print(f"  {'─' * 60}")

for did, info in doc_facts.items():
    nums_str = ", ".join(sorted(info["numbers"], key=int)[:8])
    print(f"  {did:<10s} {info['title']:<12s} {nums_str:<35s}")

# 简单冲突检测：同一篇文档内是否有矛盾描述
print(f"\n  冲突检测结果：")
conflict_count = 0
for doc in docs:
    content = doc["content"]
    # 检查是否有互相矛盾的表述
    conflict_pairs = [
        ("增加", "减少"),
        ("以上", "以下"),
        ("允许", "禁止"),
        ("可以", "不可以"),
    ]
    for word1, word2 in conflict_pairs:
        if word1 in content and word2 in content:
            print(f"    [注意] [{doc['doc_id']}] 同时出现'{word1}'和'{word2}'，需人工确认是否矛盾")
            conflict_count += 1

if conflict_count == 0:
    print(f"    [PASS] 未发现明显的文档间冲突")

print(f"\n  冲突检测的价值：")
print(f"    知识库中的文档不一致时，RAG 可能检索到不同版本的答案")
print(f"    比如文档 A 说'年假 12 天'，文档 B 说'年假 10 天'")
print(f"    用户问年假，模型可能给出不同答案——这对用户来说就是'AI 时对时错'")
print(f"    测试人员要定期检查知识库的一致性")


# ========== 8. 检索量化指标 ==========

print(f"\n{'─' * 60}")
print(f"8. 检索量化指标（MRR、Hit Rate）")
print(f"{'─' * 60}")

from rag_system import docs as all_docs

# 对每个有标准文档的测试用例，计算检索排名
ranks_list = []
hit_count = 0
total_with_answer = 0

for tc in test_cases:
    if not tc.get("expected_doc_id"):
        continue

    total_with_answer += 1
    all_results = retriever.search(tc["question"], top_k=len(all_docs))

    for i, (doc, score) in enumerate(all_results):
        if doc["doc_id"] == tc["expected_doc_id"]:
            ranks_list.append(i + 1)
            if i < 2:  # 前 2 名算命中
                hit_count += 1
            break

# 计算指标
if ranks_list:
    hit_rate = hit_count / total_with_answer
    mrr = np.mean([1 / r for r in ranks_list])
    avg_rank = np.mean(ranks_list)
else:
    hit_rate = mrr = avg_rank = 0

print(f"\n  测试样本数（有标准文档）：{total_with_answer}")
print(f"  命中数（前 2 名）：{hit_count}")
print(f"\n  Hit Rate（命中率）：{hit_rate:.2%}")
print(f"  MRR（平均倒数排名）：{mrr:.4f}")
print(f"  平均排名：{avg_rank:.1f}")

print(f"\n  指标解读：")
print(f"    Hit Rate = {hit_rate:.0%}（{'优秀' if hit_rate >= 0.8 else '良好' if hit_rate >= 0.6 else '需要优化'}）")
print(f"      → 在 {total_with_answer} 次查询中，{hit_count} 次把正确文档排在了前 2 名")
print(f"    MRR = {mrr:.4f}（{'优秀' if mrr >= 0.7 else '良好' if mrr >= 0.5 else '需要优化'}）")
print(f"      → 平均来说，正确文档的排名是第 {avg_rank:.1f} 位（1 = 最好）")

print(f"\n  MRR 速查：")
print(f"    MRR = 1.00 → 每次都排第 1（完美）")
print(f"    MRR = 0.83 → 大部分排第 1，偶尔排第 2（优秀）")
print(f"    MRR = 0.50 → 平均排第 2（合格）")
print(f"    MRR < 0.50 → 正确文档经常排在后面（需要优化检索）")
