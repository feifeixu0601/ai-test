"""
Agent 功能测试
五大维度：规划能力、工具选择、参数准确、记忆能力、错误处理

评分口径（每项 0~1）：
  tool_score   工具选择：期望工具是否被调用
  param_score  参数准确：expected_params 是否真正被校验
  answer_score 回答质量：fact_points 逐点计分（全部命中才满分）
  memory_score 记忆能力：多轮用例看第二轮是否重复调用，单轮默认满分

综合得分 = tool*0.3 + param*0.2 + answer*0.3 + memory*0.2

测试方法：
  1. 发送任务 → Agent 自主规划和执行（多轮用例逐轮共享记忆）
  2. 检查执行步骤（工具选择、参数、步骤数）
  3. 检查最终回答（事实点逐点计分）
  4. 综合评分 + 报告

限流说明：
  限流已内置到 call_llm_api（全局限流 21 秒/次），
  一次任务内 Agent 的多次 LLM 调用同样受限，测试主循环无需再 sleep。
"""

import numpy as np
import json
import time
import os
from datetime import datetime

print("=" * 60)
print("Agent 功能测试")
print("=" * 60)


# ========== 1. 加载 Agent 系统 ==========

print("\n--- 1. 加载 Agent 系统 ---")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from agent_system import agent, TOOLS

print(f"  Agent 就绪")


# ========== 2. 加载测试用例 ==========

print(f"\n--- 2. 加载测试用例 ---")

with open("agent_test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

print(f"  测试用例数：{len(test_cases)}")

dimensions = {}
for tc in test_cases:
    dim = tc["dimension"]
    dimensions[dim] = dimensions.get(dim, 0) + 1

for dim, count in dimensions.items():
    print(f"    {dim}：{count} 个用例")


# ========== 3. 执行测试 ==========

print(f"\n--- 3. 执行 Agent 测试 ---")
print(f"  限流：21 秒/次（内置在 API 层，含 Agent 多步调用）")
print(f"  预计总耗时：约 6~8 分钟")

test_results = []

for tc in test_cases:
    print(f"\n  [{tc['id']}] 维度：{tc['dimension']}")
    task_desc = tc.get("task") or " / ".join(tc["turns"])
    print(f"    任务：{task_desc[:50]}...")

    # ---- 执行（多轮用例逐轮共享记忆）----
    if "turns" in tc:
        answers, tools_per_turn, all_steps = [], [], []
        for i, turn in enumerate(tc["turns"]):
            answer, steps = agent.run(turn, max_steps=5, reset=(i == 0))
            answers.append(answer)
            all_steps.extend(steps)
            turn_tools = [s.get("tool") for s in steps if s.get("action") == "call_tool"]
            tools_per_turn.append(turn_tools)
            print(f"      第 {i + 1} 轮工具：{turn_tools if turn_tools else '无'}")
        answer = answers[-1]
        steps = all_steps
        tools_used = [t for ts in tools_per_turn for t in ts]

        # 记忆评分：后续轮次不应重复调用 search_knowledge
        repeated = [t for t in tools_per_turn[-1]
                    if t == "search_knowledge" and t in tools_per_turn[0]]
        if repeated:
            memory_score = 0.0
            memory_desc = "第二轮重复搜索，未利用记忆"
        else:
            memory_score = 1.0
            memory_desc = "复用上一轮结果，未重复调用"
    else:
        answer, steps = agent.run(tc["task"], max_steps=5)
        tools_used = [s.get("tool") for s in steps if s.get("action") == "call_tool"]
        memory_score = 1.0
        memory_desc = "单轮任务"
        tools_per_turn = [tools_used]

    print(f"    执行步骤：{len(steps)} 步")
    for s in steps:
        if s["action"] == "call_tool":
            print(f"      Step {s['step']}: 调用 {s['tool']}({s.get('params', {})})")
        elif s["action"] == "answer":
            print(f"      Step {s['step']}: 给出答案")

    # ---- 评分 1：工具选择 ----
    expected_tools = set(tc.get("expected_tools", []))
    actual_tools = set(tools_used)
    if expected_tools:
        if expected_tools.issubset(actual_tools):
            tool_score = 1.0
        elif expected_tools & actual_tools:
            tool_score = 0.5
        else:
            tool_score = 0.0
        tool_desc = f"期望 {expected_tools}，实际 {actual_tools}"
    else:
        tool_score = 1.0
        tool_desc = "无期望工具"

    # ---- 评分 2：参数准确（真正校验 expected_params）----
    param_score, param_desc = 1.0, "无参数要求"
    if "expected_params" in tc:
        miss = []
        for exp_tool, exp_params in tc["expected_params"].items():
            matched = False
            for s in steps:
                if s.get("action") == "call_tool" and s.get("tool") == exp_tool:
                    actual = s.get("params", {})
                    if all(str(v) in str(actual.get(k, "")) for k, v in exp_params.items()):
                        matched = True
            if not matched:
                miss.append(exp_tool)
        param_score = 1.0 if not miss else 0.0
        param_desc = "参数校验通过" if not miss else f"参数未命中：{miss}"

    # ---- 评分 3：回答事实点（逐点计分，全部命中才满分）----
    fact_points = tc.get("fact_points", [])
    if fact_points:
        fp_hit = sum(1 for fp in fact_points if any(v in answer for v in fp["variants"]))
        answer_score = round(fp_hit / len(fact_points), 2)
        answer_desc = f"事实点命中 {fp_hit}/{len(fact_points)}"
    else:
        keywords = tc.get("expected_answer_keywords", [])
        if keywords:
            kw_hit = sum(1 for kw in keywords if kw in answer)
            answer_score = round(kw_hit / len(keywords), 2)
            answer_desc = f"关键词命中 {kw_hit}/{len(keywords)}"
        else:
            answer_score = 1.0
            answer_desc = "无回答要求"

    # ---- 评分 4：错误处理检测（异常步骤数/死循环）----
    if len(steps) >= 5 and steps[-1].get("action") != "answer":
        tool_score = min(tool_score, 0.5)
        tool_desc += "（达到最大步数，疑似死循环或无法收尾）"
    pair_seen = set()
    for s in steps:
        if s.get("action") == "call_tool":
            key = (s.get("tool"), json.dumps(s.get("params", {}), ensure_ascii=False))
            if key in pair_seen:
                tool_score = min(tool_score, 0.5)
                tool_desc += "（同一工具+参数重复调用）"
            pair_seen.add(key)

    # ---- 综合得分 ----
    combined = round(tool_score * 0.3 + param_score * 0.2
                     + answer_score * 0.3 + memory_score * 0.2, 2)

    test_results.append({
        "id": tc["id"],
        "dimension": tc["dimension"],
        "tool_score": tool_score,
        "param_score": param_score,
        "answer_score": answer_score,
        "memory_score": memory_score,
        "combined": combined,
        "tools_used": list(actual_tools),
        "steps_count": len(steps),
        "answer_preview": answer[:60],
    })

    verdict = "[PASS]" if combined >= 0.7 else ("[WARN]" if combined >= 0.4 else "[FAIL]")
    print(f"    工具选择：{tool_score:.2f}  {tool_desc}")
    print(f"    参数准确：{param_score:.2f}  {param_desc}")
    print(f"    回答质量：{answer_score:.2f}  {answer_desc}")
    print(f"    记忆能力：{memory_score:.2f}  {memory_desc}")
    print(f"    综合得分：{combined:.2f}  {verdict}")
    print(f"    回答预览：{answer[:50]}...")

print(f"\n  测试完成！")


# ========== 4. 测试报告 ==========

print(f"\n{'=' * 60}")
print(f"Agent 功能测试报告")
print(f"{'=' * 60}")

total = len(test_results)
avg = lambda key: np.mean([r[key] for r in test_results])
avg_tool, avg_param = avg("tool_score"), avg("param_score")
avg_answer, avg_memory = avg("answer_score"), avg("memory_score")
avg_combined = avg("combined")

pass_count = sum(1 for r in test_results if r["combined"] >= 0.7)
warn_count = sum(1 for r in test_results if 0.4 <= r["combined"] < 0.7)
fail_count = sum(1 for r in test_results if r["combined"] < 0.4)

print(f"\n  测试用例数：{total}")
print(f"  [PASS]：{pass_count} 个")
print(f"  [WARN]：{warn_count} 个")
print(f"  [FAIL]：{fail_count} 个")

print(f"\n  指标汇总：")
print(f"  {'指标':<15s} {'平均分':>8s} {'判定':>10s}")
print(f"  {'─' * 35}")
for label, val, thr in [("工具选择准确率", avg_tool, 0.8),
                        ("参数准确", avg_param, 0.8),
                        ("回答质量", avg_answer, 0.8),
                        ("记忆能力", avg_memory, 0.8),
                        ("综合得分", avg_combined, 0.7)]:
    status = "[PASS]" if val >= thr else "[WARN]"
    print(f"  {label:<15s} {val:>8.2f} {status:>10s}")

# 逐条明细
print(f"\n  逐条明细：")
print(f"  {'用例':<8s} {'维度':<14s} {'工具':>5s} {'参数':>5s} {'回答':>5s} {'记忆':>5s} {'综合':>5s} {'步数':>5s}")
print(f"  {'─' * 62}")

for r in test_results:
    print(f"  {r['id']:<8s} {r['dimension']:<14s} {r['tool_score']:>5.2f} "
          f"{r['param_score']:>5.2f} {r['answer_score']:>5.2f} {r['memory_score']:>5.2f} "
          f"{r['combined']:>5.2f} {r['steps_count']:>5d}")

# 最弱维度分析
print(f"\n  最弱维度分析：")
dim_scores = {}
for r in test_results:
    dim_scores.setdefault(r["dimension"], []).append(r["combined"])

for d, scores in dim_scores.items():
    a = np.mean(scores)
    status = "[PASS]" if a >= 0.7 else "[WARN]" if a >= 0.4 else "[FAIL]"
    print(f"    {d}：{a:.2f} {status}")

worst_dim = min(dim_scores.items(), key=lambda x: np.mean(x[1]))
if np.mean(worst_dim[1]) < 0.9:
    print(f"\n    最弱维度：{worst_dim[0]}（{np.mean(worst_dim[1]):.2f}）")

    advice = {
        "规划能力": "优化 Agent Prompt 中的任务拆解指引",
        "工具选择": "优化工具描述，让 LLM 更容易区分工具用途",
        "参数准确": "在 Prompt 中给出参数示例",
        "记忆能力": "强化 Prompt 中'不要重复调用'的约束，检查历史截断是否丢信息",
        "错误处理": "在 Prompt 中明确'工具报错时如实告知，不要编造'",
        "多工具协调": "在 Prompt 中给出多步任务的拆解示例",
        "安全边界-过度拒绝": "调整 Prompt 中的安全约束——不要过度拒绝正常任务",
    }
    print(f"    建议：{advice.get(worst_dim[0], '人工复核执行轨迹')}")

# 保存报告
report = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "kimi-k2.6",
    "total_tests": total,
    "avg_tool": round(float(avg_tool), 2),
    "avg_param": round(float(avg_param), 2),
    "avg_answer": round(float(avg_answer), 2),
    "avg_memory": round(float(avg_memory), 2),
    "avg_combined": round(float(avg_combined), 2),
    "results": test_results,
}

with open("agent_test_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n  报告已保存到 agent_test_report.json")
