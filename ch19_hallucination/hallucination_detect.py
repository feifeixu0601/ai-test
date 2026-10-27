"""
幻觉检测工具
用三种方法检测大模型回答中的幻觉

方法：
  1. 知识库验证：和已知事实对比
  2. 交叉验证：多次调用看一致性
  3. 置信度分析：通过措辞判断可信度

数据来源：
  从 test_data.json 读取 LLM 回答
  （可手动替换为真实 LLM 输出，或用 API 自动获取）
"""

import numpy as np
import json
import re
from datetime import datetime

print("=" * 60)
print("大模型幻觉检测工具")
print("=" * 60)

np.random.seed(42)


# ========== 1. 加载测试数据 ==========

print("\n--- 1. 加载测试数据 ---")

with open("test_data.json", "r", encoding="utf-8") as f:
    test_responses = json.load(f)

print(f"  已加载 {len(test_responses)} 条 LLM 回答")

for resp in test_responses:
    print(f"    [{resp['id']}] {resp['question'][:20]}...  来源：{resp['source']}")


# ========== 2. 构建知识库 ==========

print(f"\n--- 2. 构建知识库 ---")

# 知识库：事实性问题的标准答案（键 = 问题关键词，值 = 事实列表）
knowledge_base = {
    "鲁迅": {
        "facts": ["周树人", "1881", "浙江绍兴", "狂人日记", "阿Q正传"],
        "description": "鲁迅相关事实",
    },
    "地球": {
        "facts": ["太阳系", "第三颗行星", "6371", "71%", "46亿"],
        "description": "地球相关事实",
    },
    "水": {
        "facts": ["H2O", "100", "0", "氢", "氧"],
        "description": "水的相关事实",
    },
    "Python": {
        "facts": ["1991", "Guido", "解释型", "高级语言", "胶水语言"],
        "description": "Python 语言相关事实",
    },
    "长城": {
        "facts": ["中国", "军事", "防御", "世界文化遗产", "秦始皇"],
        "description": "长城相关事实",
    },
}

print(f"  知识库条目数：{len(knowledge_base)}")
for key, info in knowledge_base.items():
    print(f"    {key}：{len(info['facts'])} 个已知事实")


# ========== 3. 方法一：知识库验证 ==========

print(f"\n{'─' * 60}")
print(f"3. 方法一：知识库验证")
print(f"{'─' * 60}")


def verify_with_kb(response):
    """
    知识库验证

    原理：检查回答中的关键词是否和知识库中的事实一致

    评分逻辑：
      - 找到匹配的事实 → 每个 +1 分
      - 回答中的数字和知识库不一致 → 扣分（疑似幻觉）
      - 总分 / 可匹配事实数 = 可信度
    """
    keyword = response["context_keyword"]
    answer = response["answer"]

    if keyword not in knowledge_base:
        return 0.5, "知识库中无相关信息"

    kb_facts = knowledge_base[keyword]["facts"]

    # 匹配正确的事实
    matched_facts = [f for f in kb_facts if f in answer]

    # 检查回答中的数字是否和知识库冲突
    answer_numbers = set(re.findall(r'\d+', answer))
    kb_numbers = set(re.findall(r'\d+', ' '.join(kb_facts)))

    # 回答中有数字但不在知识库中 → 可疑
    suspicious_numbers = answer_numbers - kb_numbers

    # 计算可信度
    match_ratio = len(matched_facts) / max(len(kb_facts), 1)

    if len(suspicious_numbers) > 0 and len(matched_facts) < len(kb_facts) / 2:
        score = 0.3
        desc = f"匹配 {len(matched_facts)} 个事实，发现 {len(suspicious_numbers)} 个可疑数字：{suspicious_numbers}"
    elif len(suspicious_numbers) > 0:
        score = 0.3    # 发现错误数字是强幻觉信号，重罚
        desc = f"匹配 {len(matched_facts)} 个事实，但有 {len(suspicious_numbers)} 个未验证数字"
    elif match_ratio >= 0.6:
        score = 1.0
        desc = f"匹配 {len(matched_facts)} 个事实，无冲突"
    elif match_ratio >= 0.3:
        score = 0.7
        desc = f"匹配 {len(matched_facts)} 个事实（部分覆盖）"
    else:
        score = 0.4
        desc = f"仅匹配 {len(matched_facts)} 个事实（信息不足）"

    return score, desc


print(f"\n  {'用例':<8s} {'可信度':>8s} {'验证结果':<55s}")
print(f"  {'─' * 75}")

kb_results = {}
for resp in test_responses:
    score, desc = verify_with_kb(resp)
    kb_results[resp["id"]] = score
    marker = "[可信]" if score >= 0.7 else ("[存疑]" if score >= 0.4 else "[疑似幻觉]")
    print(f"  {resp['id']:<8s} {score:>8.2f} {desc:<55s} {marker}")


# ========== 4. 方法二：交叉验证 ==========

print(f"\n{'─' * 60}")
print(f"4. 方法二：交叉验证")
print(f"{'─' * 60}")
print(f"  同一问题多次询问，看回答是否一致")
print(f"  （这里用启发式模拟多次调用，实际使用时调用真实 API）")


def cross_verify(response):
    """
    交叉验证：对同一问题模拟多次调用，看关键信息是否一致

    原理：如果 3 次回答中关键事实一致，大概率是真信息
          如果各说各的，至少有一次是幻觉

    评分逻辑：
      - 所有回答的关键信息一致 → 1.0
      - 大部分一致 → 0.6
      - 完全不一致 → 0.2
    """
    keyword = response["context_keyword"]
    base_answer = response["answer"]

    # 模拟多次调用的结果
    # 启发式逻辑：如果原回答包含幻觉（数字和知识库冲突），模拟结果不一致
    if keyword not in knowledge_base:
        simulated_runs = [base_answer, base_answer, base_answer]
    else:
        kb_numbers = set(re.findall(r'\d+', ' '.join(knowledge_base[keyword]["facts"])))
        answer_numbers = set(re.findall(r'\d+', base_answer))
        suspicious = answer_numbers - kb_numbers

        if suspicious:
            # 含幻觉的回答：3 次结果不一致
            simulated_runs = [
                base_answer,
                base_answer.replace(list(suspicious)[0], "【数字不一致】") if suspicious else base_answer,
                "【回答与前两次不同，数字有出入】",
            ]
        else:
            # 正确回答：3 次结果基本一致
            simulated_runs = [base_answer, base_answer, base_answer]

    # 提取每次回答中的数字（作为关键信息的代表）
    run_numbers = [set(re.findall(r'\d+', run)) for run in simulated_runs]

    # 计算一致性（Jaccard 相似度的平均值）
    similarities = []
    for i in range(len(run_numbers)):
        for j in range(i + 1, len(run_numbers)):
            intersection = len(run_numbers[i] & run_numbers[j])
            union = len(run_numbers[i] | run_numbers[j])
            sim = intersection / union if union > 0 else 1.0
            similarities.append(sim)

    avg_sim = float(np.mean(similarities)) if similarities else 1.0

    if avg_sim >= 0.8:
        score = 1.0
        desc = f"3 次回答高度一致（相似度 {avg_sim:.2f}）"
    elif avg_sim >= 0.4:
        score = 0.6
        desc = f"3 次回答部分一致（相似度 {avg_sim:.2f}）"
    else:
        score = 0.2
        desc = f"3 次回答严重不一致（相似度 {avg_sim:.2f}），疑似幻觉"

    return score, desc


print(f"\n  {'用例':<8s} {'可信度':>8s} {'验证结果':<55s}")
print(f"  {'─' * 75}")

cv_results = {}
for resp in test_responses:
    score, desc = cross_verify(resp)
    cv_results[resp["id"]] = score
    marker = "[可信]" if score >= 0.7 else ("[存疑]" if score >= 0.4 else "[疑似幻觉]")
    print(f"  {resp['id']:<8s} {score:>8.2f} {desc:<55s} {marker}")


# ========== 5. 方法三：置信度分析 ==========

print(f"\n{'─' * 60}")
print(f"5. 方法三：置信度分析")
print(f"{'─' * 60}")
print(f"  通过回答中的措辞判断模型的确信程度")


def confidence_analysis(response):
    """
    置信度分析：通过措辞判断可信度

    高置信表述（模型很确信）：
      "原名"、"是"、"确定"、"根据"、具体数字

    低置信表述（模型不太确定）：
      "可能"、"大约"、"我猜"、"不太确定"、"据说"

    逻辑：模型越自信，分数越高（注意：自信不代表真的！）
    """
    text = response["answer"]

    high_confidence_markers = ["原名", "是", "确定", "根据", "具体",
                               "1881", "1991", "6371", "100", "H2O"]
    low_confidence_markers = ["可能", "大约", "我猜", "不太确定",
                              "据说", "大概", "也许", "我不太清楚"]

    high_count = sum(1 for m in high_confidence_markers if m in text)
    low_count = sum(1 for m in low_confidence_markers if m in text)

    total = high_count + low_count

    if total == 0:
        return 0.5, "无法从措辞判断置信度"

    confidence = high_count / total

    if confidence >= 0.8:
        desc = f"高置信 {high_count} 个 / 低置信 {low_count} 个（模型很自信）"
    elif confidence >= 0.5:
        desc = f"高置信 {high_count} 个 / 低置信 {low_count} 个（置信度中等）"
    else:
        desc = f"高置信 {high_count} 个 / 低置信 {low_count} 个（模型不太自信）"

    return round(confidence, 2), desc


print(f"\n  {'用例':<8s} {'置信度':>8s} {'分析结果':<55s}")
print(f"  {'─' * 75}")

cf_results = {}
for resp in test_responses:
    score, desc = confidence_analysis(resp)
    cf_results[resp["id"]] = score
    print(f"  {resp['id']:<8s} {score:>8.2f} {desc:<55s}")

print(f"\n  [重要提示] 置信度高 ≠ 内容真！")
print(f"    含幻觉的回答（如 HR-02）的置信度也可能很高——模型用自信的语气说了假话。")
print(f"    这就是为什么置信度分析只能做初筛，不能单独用来判断幻觉。")


# ========== 6. 综合判定 ==========

print(f"\n{'=' * 60}")
print(f"6. 综合判定（三种方法加权）")
print(f"{'=' * 60}")

# 加权：知识库验证最重要，交叉验证次之，置信度分析辅助
WEIGHT_KB = 0.55
WEIGHT_CV = 0.35
WEIGHT_CF = 0.10

print(f"\n  权重分配：知识库 {WEIGHT_KB} + 交叉验证 {WEIGHT_CV} + 置信度 {WEIGHT_CF}")
print(f"\n  {'用例':<8s} {'KB':>6s} {'CV':>6s} {'CF':>6s} {'综合':>8s} {'判定':>14s}")
print(f"  {'─' * 52}")

final_results = []
for resp in test_responses:
    rid = resp["id"]
    kb = kb_results[rid]
    cv = cv_results[rid]
    cf = cf_results[rid]

    combined = kb * WEIGHT_KB + cv * WEIGHT_CV + cf * WEIGHT_CF

    if combined >= 0.7:
        verdict = "[可信]"
    elif combined >= 0.6:
        verdict = "[存疑-需审核]"
    else:
        verdict = "[疑似幻觉]"

    final_results.append({
        "id": rid,
        "kb": kb,
        "cv": cv,
        "cf": cf,
        "combined": round(combined, 2),
        "verdict": verdict,
    })

    print(f"  {rid:<8s} {kb:>6.2f} {cv:>6.2f} {cf:>6.2f} {combined:>8.2f} {verdict:>14s}")

# 验证检测效果
print(f"\n  检测效果验证（对比预埋的幻觉）：")

# 从数据中找出标注为"含幻觉"的条目
known_hallucination = [r["id"] for r in test_responses if "含幻觉" in r.get("source", "")]
print(f"    预埋含幻觉的回答：{known_hallucination}")

detected_hallucinations = [r["id"] for r in final_results if r["verdict"] == "[疑似幻觉]"]
print(f"    检测为疑似幻觉的：{detected_hallucinations}")

if set(known_hallucination) == set(detected_hallucinations):
    print(f"    [PASS] 所有预埋幻觉都被成功检测到！")
elif set(detected_hallucinations) & set(known_hallucination):
    overlap = set(detected_hallucinations) & set(known_hallucination)
    missed = set(known_hallucination) - set(detected_hallucinations)
    print(f"    [WARN] 检测到 {overlap}，但遗漏了 {missed}")
else:
    print(f"    [FAIL] 未能检测到任何预埋幻觉")
