"""
LLM-as-a-Judge 演示
用"AI 评委"对开放式回答打分

演示内容：
  1. 定义评分标准（Rubric）
  2. 构造评委 Prompt
  3. 模拟评委打分
  4. 评委可靠性校验（评委 vs 人工）
"""

import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 60)
print("LLM-as-a-Judge 演示")
print("=" * 60)

np.random.seed(42)


# ========== 1. 定义评分标准 ==========

print("\n--- 1. 定义评分标准（Rubric） ---")

rubric = {
    "task": "评价一段 AI 生成的文本回答质量",
    "dimensions": {
        "准确性": {
            "description": "内容是否正确、无事实错误",
            "scores": {
                5: "完全正确，无任何事实错误",
                4: "基本正确，有极小瑕疵但不影响理解",
                3: "大致正确，有少量事实错误",
                2: "有明显事实错误",
                1: "大部分内容错误",
            },
        },
        "完整性": {
            "description": "是否完整回答了问题",
            "scores": {
                5: "全面回答，覆盖所有要点",
                4: "回答了主要要点，个别细节缺失",
                3: "回答了部分要点",
                2: "回答了少数要点",
                1: "答非所问或严重不完整",
            },
        },
        "清晰度": {
            "description": "表达是否清晰易懂",
            "scores": {
                5: "表达清晰流畅，结构分明",
                4: "表达清楚，偶有不流畅",
                3: "基本能理解，但有些混乱",
                2: "表达晦涩，难以理解",
                1: "完全无法理解",
            },
        },
    },
}

print(f"  评分任务：{rubric['task']}")
print(f"  评分维度：{list(rubric['dimensions'].keys())}")
for dim_name, dim_info in rubric["dimensions"].items():
    print(f"    {dim_name}：{dim_info['description']}")


# ========== 2. 构造评委 Prompt ==========

print(f"\n--- 2. 构造评委 Prompt ---")


def build_judge_prompt(question, response, rubric):
    """
    构造给评委模型的 Prompt

    关键要素：
      1. 角色设定：你是专业的评测员
      2. 评分标准：详细的 Rubric
      3. 被评内容：问题 + 回答
      4. 输出格式：结构化的分数和理由
    """
    prompt = f"""你是一位专业的 AI 回答质量评测员。请根据以下评分标准，对给出的回答进行评分。

## 评分标准

{rubric['task']}

评分维度：
"""

    for dim_name, dim_info in rubric["dimensions"].items():
        prompt += f"\n### {dim_name}\n{dim_info['description']}\n评分档次：\n"
        for score, desc in sorted(dim_info["scores"].items(), reverse=True):
            prompt += f"  {score} 分：{desc}\n"

    prompt += f"""

## 被评内容

**问题**：{question}

**回答**：{response}

## 输出要求

请对每个维度打分（1-5 分），并给出理由。输出格式：

准确性：X 分
理由：...

完整性：X 分
理由：...

清晰度：X 分
理由：...
"""

    return prompt


# 构造一条评委 Prompt 示例
sample_question = "请介绍长城的历史"
sample_response = "长城是中国古代的军事防御工程，始建于春秋战国时期，秦始皇统一六国后将各段长城连接起来。明代是长城修建的最后一个高峰。长城全长超过两万里，是世界上最伟大的建筑奇迹之一，1987 年被列入世界文化遗产。"

judge_prompt = build_judge_prompt(sample_question, sample_response, rubric)

print(f"  评委 Prompt 已构造")
print(f"  Prompt 长度：{len(judge_prompt)} 字符")
print(f"  包含评分维度：准确性、完整性、清晰度")
print(f"  每个维度有 5 个评分档次")


# ========== 3. 模拟评委打分 ==========

print(f"\n--- 3. 模拟评委打分 ---")
print(f"  注意：实际使用时将此处替换为调用真实 LLM API")


def mock_judge_score(question, response):
    """
    模拟评委模型的打分输出
    实际使用时替换为：调用 GPT-4 / Claude / 其他强模型 API
    """
    scores = {}

    # 准确性：根据回答内容判断
    if any(kw in response for kw in ["错误", "不对", "并非"]):
        scores["准确性"] = 2
    elif len(response) > 100 and ("。" in response or "：" in response):
        scores["准确性"] = 5
    elif len(response) > 50:
        scores["准确性"] = 4
    else:
        scores["准确性"] = 3

    # 完整性：根据回答长度和要点覆盖
    if len(response) > 100:
        scores["完整性"] = 5
    elif len(response) > 50:
        scores["完整性"] = 4
    else:
        scores["完整性"] = 2

    # 清晰度：根据结构和标点
    if "。" in response and ("：" in response or "，" in response):
        scores["清晰度"] = 5
    elif "。" in response or "，" in response:
        scores["清晰度"] = 4
    else:
        scores["清晰度"] = 3

    # 理由也根据内容变化
    reasons = {
        "准确性": "内容详实，史实准确" if scores["准确性"] >= 5
                  else ("内容基本正确" if scores["准确性"] >= 4
                  else "回答简略，可能有遗漏"),
        "完整性": "覆盖全面，要点齐全" if scores["完整性"] >= 5
                  else ("覆盖了主要要点" if scores["完整性"] >= 4
                  else "回答太简短，要点缺失严重"),
        "清晰度": "表达流畅，结构分明" if scores["清晰度"] >= 5
                  else ("表达清楚" if scores["清晰度"] >= 4
                  else "表达较简略"),
    }

    return scores, reasons


# 测试用例：3 条不同质量的回答
judge_test_cases = [
    {
        "question": "请介绍长城的历史",
        "response": "长城是中国古代的军事防御工程，始建于春秋战国时期，秦始皇统一六国后将各段长城连接起来。明代是长城修建的最后一个高峰。长城全长超过两万里，是世界上最伟大的建筑奇迹之一，1987 年被列入世界文化遗产。",
    },
    {
        "question": "什么是机器学习？",
        "response": "机器学习是人工智能的一个分支。",
    },
    {
        "question": "推荐三本好书",
        "response": "推荐《百年孤独》、《1984》和《小王子》。《百年孤独》是魔幻现实主义代表作，《1984》描绘了极权社会，《小王子》是温暖的哲理童话。三本书风格各异，但都值得一读。",
    },
]

print(f"\n  待评回答数：{len(judge_test_cases)}")
print()

judge_results = []
for i, case in enumerate(judge_test_cases, 1):
    scores, reasons = mock_judge_score(case["question"], case["response"])
    avg_score = np.mean(list(scores.values()))

    judge_results.append({
        "case_id": i,
        "question": case["question"],
        "response_preview": case["response"][:30],
        "scores": scores,
        "reasons": reasons,
        "avg_score": avg_score,
    })

    print(f"  [用例 {i}] 问题：{case['question']}")
    print(f"    回答预览：{case['response'][:30]}...")
    for dim in ["准确性", "完整性", "清晰度"]:
        print(f"    {dim}：{scores[dim]} 分 — {reasons[dim]}")
    print(f"    平均分：{avg_score:.1f} / 5.0")
    print()


# ========== 4. 评委可靠性校验 ==========

print(f"--- 4. 评委可靠性校验 ---")
print(f"  评委打的分靠谱吗？用人工标注来检验！")

# 模拟人工评分（实际项目中由人工标注）
human_scores = [
    {"准确性": 5, "完整性": 5, "清晰度": 4},   # 用例1：长城介绍
    {"准确性": 3, "完整性": 1, "清晰度": 2},   # 用例2：人工觉得更差
    {"准确性": 5, "完整性": 5, "清晰度": 5},   # 用例3：推荐书籍
]

print(f"\n  {'用例':>4s} {'维度':>8s} {'评委分':>8s} {'人工分':>8s} {'差异':>8s} {'一致？':>8s}")
print(f"  {'─' * 50}")

diffs = []
for i, (jr, hr) in enumerate(zip(judge_results, human_scores), 1):
    for dim in ["准确性", "完整性", "清晰度"]:
        judge_s = jr["scores"][dim]
        human_s = hr[dim]
        diff = abs(judge_s - human_s)
        diffs.append(diff)
        consistent = "[一致]" if diff <= 1 else "[偏差]"

        print(f"  {i:>4d} {dim:>8s} {judge_s:>8d} {human_s:>8d} {diff:>8d} {consistent:>8s}")

# 计算评委可靠性指标
mean_diff = np.mean(diffs)
max_diff = max(diffs)
agreement_rate = sum(1 for d in diffs if d <= 1) / len(diffs) * 100

print(f"\n  评委可靠性指标：")
print(f"    平均差异：{mean_diff:.2f} 分（越小越好）")
print(f"    最大差异：{max_diff} 分")
print(f"    一致率（±1 分内）：{agreement_rate:.1f}%")

if agreement_rate >= 80:
    print(f"    [PASS] 评委可靠性良好，可替代人工评分")
elif agreement_rate >= 60:
    print(f"    [WARN] 评委可靠性一般，建议抽查校验")
else:
    print(f"    [FAIL] 评委可靠性不足，需要调整评分 Prompt 或更换评委模型")

print(f"\n  提升评委可靠性的方法：")
print(f"    1. 优化评分标准（Rubric 写得越详细，评委越稳定）")
print(f"    2. 提供评分示例（Few-shot：先给几个标注好的例子）")
print(f"    3. 多次评分取平均（降低随机波动）")
print(f"    4. 换更强的评委模型（能力越强，评得越准）")
print(f"    5. 定期抽查人工复核（防止评委标准漂移）")
