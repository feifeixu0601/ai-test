"""
幻觉检测工具（真实 API 版）
用三种方法检测大模型回答中的幻觉

方法：
  1. 知识库验证：和已知事实对比
  2. 交叉验证：调用 3 次 API 看一致性
  3. 置信度分析：通过措辞判断可信度

数据来源：
  从 test_ai_data.json 读取问题，调用 Kimi API 获取真实回答
"""

import numpy as np
import json
import re
import time
import os
from datetime import datetime

print("=" * 60)
print("大模型幻觉检测工具（真实 API 版）")
print("=" * 60)

np.random.seed(42)


# ========== 1. 加载测试数据 ==========

print("\n--- 1. 加载测试数据 ---")

with open("test_ai_data.json", "r", encoding="utf-8") as f:
    test_responses = json.load(f)

print(f"  已加载 {len(test_responses)} 条 LLM 回答")

for resp in test_responses:
    print(f"    [{resp['id']}] {resp['question'][:30]}...  来源：{resp['source']}")


# ========== 1.5 调用 Kimi API 获取真实回答 ==========

print(f"\n--- 1.5 调用 Kimi API 获取真实回答 ---")
print(f"  限流控制：每次调用后等待 21 秒（适配 RPM=3）")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("  [提示] 未安装 openai 库，将使用 test_ai_data.json 中已有的数据")
    print("         如需 API 调用请执行：pip install openai")


def call_llm_api(question, api_key=None,
                 base_url="https://api.moonshot.cn/v1",
                 model="kimi-k2.6"):
    """
    调用 Kimi API 获取回答（含限流保护）

    注意：
      1. kimi-k2.6 是推理模型，max_tokens 必须够大（思考吃 token）
      2. kimi-k2.6 不允许传 temperature，用默认值
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
            max_tokens=4000,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        elif response.choices and response.choices[0].finish_reason == "length":
            print("  [警告] max_tokens 不够，回答被截断")
            return response.choices[0].message.content
        return None
    except Exception as e:
        print(f"  [错误] API 调用失败: {e}")
        return None


for resp in test_responses:
    if resp.get("answer", "").strip():
        print(f"  [{resp['id']}] 已有回答，跳过 API 调用")
        continue

    print(f"  [{resp['id']}] 正在调用 Kimi API...")
    real_answer = call_llm_api(question=resp["question"])

    if real_answer:
        resp["answer"] = real_answer
        resp["source"] = "Kimi 回答（kimi-k2.6）"
        print(f"  [{resp['id']}] 获取成功（{len(real_answer)} 字符）")

        # 打印完整回答，方便人工核对
        print(f"\n  ── AI 完整回答 ──")
        for line in real_answer.split("\n"):
            print(f"  {line}")
        print(f"  ── 回答结束 ──\n")
    else:
        print(f"  [{resp['id']}] 获取失败，请检查 API Key 或网络")

    time.sleep(21)

print(f"  API 调用完成！数据已就绪，开始检测...")


# ========== 2. 构建知识库 ==========

print(f"\n--- 2. 构建知识库 ---")

# 知识库：事实性问题的标准答案
# 注意：facts 里要包含中间计算可能出现的数字，减少误报
knowledge_base = {
    "三重积分": {
        "facts": ["3/2", "1.5", "1/2", "0.5", "单位", "立方体",
                  "积分", "0", "1", "3", "6", "4"],
        "description": "三重积分相关事实（单位立方体上的 ∭(x+y+z)dV = 3/2）",
    },
}

print(f"  知识库条目数：{len(knowledge_base)}")
for key, info in knowledge_base.items():
    print(f"    {key}：{len(info['facts'])} 个已知事实")
print(f"  [说明] 知识库中的数字包含中间计算值（3、6、4 等），避免误报")


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
        desc = f"匹配 {len(matched_facts)} 个事实，但有 {len(suspicious_numbers)} 个未验证数字：{suspicious_numbers}"
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


print(f"\n  {'用例':<10s} {'可信度':>8s} {'验证结果':<55s}")
print(f"  {'─' * 75}")

kb_results = {}
for resp in test_responses:
    score, desc = verify_with_kb(resp)
    kb_results[resp["id"]] = score
    marker = "[可信]" if score >= 0.7 else ("[存疑]" if score >= 0.4 else "[疑似幻觉]")
    print(f"  {resp['id']:<10s} {score:>8.2f} {desc:<55s} {marker}")


# ========== 4. 方法二：交叉验证 ==========

print(f"\n{'─' * 60}")
print(f"4. 方法二：交叉验证（真实 API）")
print(f"{'─' * 60}")
print(f"  同一问题调用 3 次 API，看回答是否一致")
print(f"  限流适配：每次调用后等 21 秒")


def cross_verify(response):
    """
    交叉验证：对同一问题调用 3 次 API，看回答是否一致

    限流适配：每次调用后等 21 秒，3 次调用约 63 秒
    """
    question = response["question"]

    runs = []
    for i in range(3):
        print(f"    交叉验证第 {i+1}/3 次调用...")
        answer = call_llm_api(question)
        if answer:
            runs.append(answer)
            print(f"      获取成功（{len(answer)} 字符）")
        else:
            print(f"      获取失败")
        if i < 2:
            time.sleep(21)

    if len(runs) < 2:
        return 0.5, "调用次数不足，无法交叉验证"

    # 提取每次回答中的数字，比较一致性
    run_numbers = [set(re.findall(r'\d+', run)) for run in runs]

    similarities = []
    for i in range(len(run_numbers)):
        for j in range(i + 1, len(run_numbers)):
            intersection = len(run_numbers[i] & run_numbers[j])
            union = len(run_numbers[i] | run_numbers[j])
            sim = intersection / union if union > 0 else 1.0
            similarities.append(sim)

    avg_sim = float(np.mean(similarities))

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


print(f"\n  {'用例':<10s} {'可信度':>8s} {'验证结果':<55s}")
print(f"  {'─' * 75}")

cv_results = {}
for resp in test_responses:
    score, desc = cross_verify(resp)
    cv_results[resp["id"]] = score
    marker = "[可信]" if score >= 0.7 else ("[存疑]" if score >= 0.4 else "[疑似幻觉]")
    print(f"  {resp['id']:<10s} {score:>8.2f} {desc:<55s} {marker}")


# ========== 5. 方法三：置信度分析 ==========

print(f"\n{'─' * 60}")
print(f"5. 方法三：置信度分析")
print(f"{'─' * 60}")
print(f"  通过回答中的措辞判断模型的确信程度")


def confidence_analysis(response):
    """
    置信度分析：通过措辞判断可信度

    数学/物理问题的关键词表
    """
    text = response["answer"]

    high_confidence_markers = [
        "等于", "公式", "定理", "结果是", "计算",
        "积分", "推导", "化简", "由...得", "= "
    ]
    low_confidence_markers = [
        "约等于", "大概", "估算", "可能", "我需要查一下",
        "近似", "不太确定", "仅供参考", "需要进一步验证"
    ]

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


print(f"\n  {'用例':<10s} {'置信度':>8s} {'分析结果':<55s}")
print(f"  {'─' * 75}")

cf_results = {}
for resp in test_responses:
    score, desc = confidence_analysis(resp)
    cf_results[resp["id"]] = score
    print(f"  {resp['id']:<10s} {score:>8.2f} {desc:<55s}")

print(f"\n  [重要提示] 置信度高 ≠ 内容真！")
print(f"    模型可能用自信的语气说了假话。")
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
print(f"\n  {'用例':<10s} {'KB':>6s} {'CV':>6s} {'CF':>6s} {'综合':>8s} {'判定':>14s}")
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

    print(f"  {rid:<10s} {kb:>6.2f} {cv:>6.2f} {cf:>6.2f} {combined:>8.2f} {verdict:>14s}")

# 人工确认提示
print(f"\n  最终判定：")
for r in final_results:
    print(f"    [{r['id']}] 综合得分 {r['combined']} → {r['verdict']}")

print(f"\n  请对照上面打印的 AI 完整回答，人工确认答案是否正确。")
print(f"    正确答案是 3/2（1.5）")
print(f"    如果 AI 答对了但被判'存疑' → 说明知识库/评分逻辑需要微调")
print(f"    如果 AI 答错了但被判'可信' → 说明检测工具有漏洞，需要加强")