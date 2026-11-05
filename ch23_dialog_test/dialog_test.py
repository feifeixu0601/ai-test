"""
多轮对话测试工具（含多次运行统计）
六大维度：上下文记忆、话题连贯性、回答一致性、对话状态管理、情绪感知、模糊澄清

评分口径：
  每个维度独立评分（0~1），子检查不满足条件时返回 None（跳过，不参与加权）
  综合得分 = Σ(维度得分 × 权重) / Σ(有用例维度的权重)

多次运行统计：
  每个用例跑 N_RUNS 次，取平均分、标准差、通过率
  稳定性判定：平均分 ≥ 0.8 且标准差 ≤ 0.15 → [PASS]

事实匹配：
  两级匹配：精确子串 → 数字归一化后匹配
  解决"八小时 vs 8 小时"的假阴性问题

限流说明：
  限流已内置到 call_llm_api（全局限流 21 秒/次），
  多轮对话内部的多次 LLM 调用同样受限，测试主循环无需再 sleep。
"""

import numpy as np
import json
import re
import os
from datetime import datetime

print("=" * 60)
print("多轮对话测试工具（含多次运行统计）")
print("=" * 60)


# ========== 1. 加载对话系统 ==========

print("\n--- 1. 加载对话系统 ---")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from dialog_system import dialog_mgr

print(f"  对话系统就绪")


# ========== 2. 加载测试用例 ==========

print(f"\n--- 2. 加载测试用例 ---")

with open("dialog_test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

print(f"  测试用例数：{len(test_cases)}")

dimensions = {}
for tc in test_cases:
    dim = tc["dimension"]
    dimensions[dim] = dimensions.get(dim, 0) + 1

for dim, count in dimensions.items():
    print(f"    {dim}：{count} 个用例")


# ========== 3. 六维权重 + 运行配置 ==========

WEIGHTS = {
    "上下文记忆": 0.20,
    "话题连贯性": 0.20,
    "回答一致性": 0.15,
    "对话状态管理": 0.20,
    "情绪感知": 0.15,
    "模糊澄清": 0.10,
}

N_RUNS = 3  # 每个用例跑几次（真实项目建议 3~5 次）


# ========== 4. 数字归一化（解决子串匹配假阴性） ==========

def normalize_numbers(text):
    """
    把中文数字转成阿拉伯数字

    注意：仅用于短事实变体匹配（如"八小时"→"8小时"），
    不要用于长文本事实抽取——"一些"会被误转成"1些"
    """
    cn_to_ar = {
        '二十': '20', '十九': '19', '十八': '18', '十七': '17',
        '十六': '16', '十五': '15', '十四': '14', '十三': '13',
        '十二': '12', '十一': '11', '十': '10',
        '九': '9', '八': '8', '七': '7', '六': '6',
        '五': '5', '四': '4', '三': '3', '二': '2', '一': '1', '零': '0',
    }
    result = text
    for cn, ar in sorted(cn_to_ar.items(), key=lambda x: -len(x[0])):
        result = result.replace(cn, ar)
    return result


def normalize_text(text):
    """标准化文本：去空格 + 中文数字转阿拉伯"""
    text = text.replace(' ', '').replace('　', '')
    text = normalize_numbers(text)
    return text


def match_fact(text, variants):
    """
    两级匹配：

    第一级：精确子串匹配（快速、零成本）
    第二级：归一化后匹配（解决"八小时 vs 8 小时"的假阴性）
    """
    # 第一级：精确子串
    for v in variants:
        if v in text:
            return True

    # 第二级：归一化后匹配
    norm_text = normalize_text(text)
    for v in variants:
        norm_v = normalize_text(v)
        if norm_v in norm_text:
            return True

    return False


print(f"  数字归一化模块就绪（支持中文数字 → 阿拉伯数字）")


# ========== 5. 各子检查函数 ==========

def score_fact_points(tc, log):
    """
    事实点评分：检查指定轮次的回答中是否包含期望事实

    使用两级匹配（精确子串 → 归一化匹配）
    """
    check_turn = tc.get("check_turn", len(log))
    check_resp = log[check_turn - 1]["assistant"] if check_turn <= len(log) else ""

    fact_points = tc.get("fact_points", [])
    if not fact_points:
        return None, "无事实点要求"

    fp_hit = sum(1 for fp in fact_points if match_fact(check_resp, fp["variants"]))
    score = round(fp_hit / len(fact_points), 2)
    desc = f"事实点命中 {fp_hit}/{len(fact_points)}（第{check_turn}轮）"
    return score, desc


def score_consistency(tc, log):
    """
    一致性检查：对比两个轮次回答中的关键事实

    只比 fact_points 相关数字（避免无关数字干扰）
    一致性只管"答得一不一致"，对错归事实点维度
    """
    if "consistency_check" not in tc:
        return None, "未配置一致性检查"

    cc = tc["consistency_check"]
    turn_nums = cc.get("turns", [])

    # 提取 fact_points 中的目标数字（归一化后提取）
    relevant_numbers = set()
    for fp in tc.get("fact_points", []):
        for v in fp["variants"]:
            norm_v = normalize_text(v)
            relevant_numbers.update(re.findall(r'\d+', norm_v))

    # 边界保护
    responses = [log[t - 1]["assistant"] for t in turn_nums if t <= len(log)]

    if len(responses) < 2 or not relevant_numbers:
        return None, "轮次不足或无目标事实，跳过一致性检查"

    # 只检查目标数字在各轮回答中是否存在
    facts_per_turn = []
    for r in responses:
        norm_r = normalize_text(r)
        turn_facts = frozenset(num for num in relevant_numbers if num in norm_r)
        facts_per_turn.append(turn_facts)

    if facts_per_turn[0] == facts_per_turn[1]:
        score = 1.0
        desc = f"第{','.join(map(str, turn_nums))}轮关键事实一致：{set(facts_per_turn[0])}"
    else:
        score = 0.0
        desc = f"第{','.join(map(str, turn_nums))}轮关键事实不一致：{set(facts_per_turn[0])} vs {set(facts_per_turn[1])}"

    return score, desc


def score_coherence(tc, log):
    """
    话题连贯检查（双向断言）

    负向：跑题轮不应包含旧话题词
    正向：回归轮应包含旧话题词
    """
    if "coherence_check" not in tc:
        return None, "未配置话题连贯检查"

    cc = tc["coherence_check"]

    # 边界保护
    off_resp = log[cc["off_topic_turn"] - 1]["assistant"] if cc["off_topic_turn"] <= len(log) else ""
    return_resp = log[cc["return_turn"] - 1]["assistant"] if cc["return_turn"] <= len(log) else ""

    # 负向断言
    leaked = [k for k in cc.get("off_topic_must_not_contain", []) if k in off_resp]
    no_topic_leak = len(leaked) == 0

    # 正向断言
    return_hit = any(k in return_resp for k in cc.get("return_must_contain_any", []))

    if no_topic_leak and return_hit:
        score = 1.0
        desc = "跑题轮正确识别新话题，回归轮正确接回旧话题"
    elif return_hit and not no_topic_leak:
        score = 0.5
        desc = f"回归轮正确，但跑题轮混入旧话题词：{leaked}"
    elif no_topic_leak and not return_hit:
        score = 0.5
        desc = "跑题轮正确，但回归轮没接回旧话题"
    else:
        score = 0.0
        desc = "跑题轮和回归轮都有问题"

    return score, desc


def score_state(tc, log):
    """
    状态覆盖检查（改口后旧值是否被新值覆盖）

    must_contain：新值必须出现
    must_not_contain：旧值不能残留
    """
    if "state_check" not in tc:
        return None, "未配置状态覆盖检查"

    sc = tc["state_check"]
    check_resp = log[sc["turn"] - 1]["assistant"] if sc["turn"] <= len(log) else ""

    must_have = [k for k in sc.get("must_contain", []) if k in check_resp]
    must_have_ok = len(must_have) == len(sc.get("must_contain", []))

    leaked = [k for k in sc.get("must_not_contain", []) if k in check_resp]
    no_leak = len(leaked) == 0

    if must_have_ok and no_leak:
        score = 1.0
        desc = "新值覆盖旧值，状态管理正确"
    elif must_have_ok and not no_leak:
        score = 0.5
        desc = f"新值正确但旧值残留：{leaked}"
    else:
        missing = set(sc.get("must_contain", [])) - set(must_have)
        score = 0.0
        desc = f"状态未正确更新（缺少：{missing}）"

    return score, desc


def score_emotion(tc, log):
    """情绪感知检查：指定轮次的回答是否识别了用户情绪"""
    if "emotion_check" not in tc:
        return None, "未配置情绪检查"

    ec = tc["emotion_check"]
    resp = log[ec["turn"] - 1]["assistant"] if ec["turn"] <= len(log) else ""

    signals_found = [s for s in ec.get("expected_signals", []) if s in resp]

    if signals_found:
        score = 1.0
        desc = f"识别情绪（{signals_found}）"
    else:
        score = 0.0
        desc = "未识别用户情绪"

    return score, desc


def score_clarify(tc, log):
    """
    模糊澄清检查：模糊指令是否触发了主动追问

    加 1 平滑（Bayesian smoothing），避免单条用例随机波动主导维度得分
    """
    if "clarify_check" not in tc:
        return None, "未配置澄清检查"

    cc = tc["clarify_check"]
    clarify_count = 0

    for t in cc["turns"]:
        if t <= len(log):
            resp = log[t - 1]["assistant"]
            if any(s in resp for s in cc.get("expected_signals", [])):
                clarify_count += 1

    total_turns = len(cc["turns"])

    if total_turns > 0:
        score = round((clarify_count + 1) / (total_turns + 2), 2)
    else:
        score = 1.0

    desc = f"主动澄清 {clarify_count}/{total_turns} 轮（加1平滑后 {score:.2f}）"
    return score, desc


# ========== 6. 统一评分流水线 ==========

def calculate_scores(tc, log):
    """
    统一评分流水线

    每个子检查有条件地写入自己的 key（有检查配置才执行）
    事实分映射到用例主维度，不覆盖维度专用检查的得分

    注意：当用例 dimension 与某个维度专用检查键名一致时（如 DLG-03
    维度是"回答一致性"），步骤 1 先写事实分，步骤 2 再用一致性分
    覆盖它。这是预期行为——专用检查比通用事实分更精准。
    """
    scores = {}
    descs = {}

    # 1. 事实点 → 映射到用例主维度
    s, d = score_fact_points(tc, log)
    descs["事实点"] = d
    if s is not None:
        scores[tc["dimension"]] = s

    # 2. 一致性（仅当配置了 consistency_check 才写入）
    s, d = score_consistency(tc, log)
    if s is not None:
        scores["回答一致性"] = s
        descs["一致性"] = d

    # 3. 话题连贯（仅当配置了 coherence_check 才写入）
    s, d = score_coherence(tc, log)
    if s is not None:
        scores["话题连贯性"] = s
        descs["话题连贯"] = d

    # 4. 状态覆盖（仅当配置了 state_check 才写入）
    s, d = score_state(tc, log)
    if s is not None:
        scores["对话状态管理"] = s
        descs["状态覆盖"] = d

    # 5. 情绪感知（仅当配置了 emotion_check 才写入）
    s, d = score_emotion(tc, log)
    if s is not None:
        scores["情绪感知"] = s
        descs["情绪感知"] = d

    # 6. 模糊澄清（仅当配置了 clarify_check 才写入）
    s, d = score_clarify(tc, log)
    if s is not None:
        scores["模糊澄清"] = s
        descs["模糊澄清"] = d

    # 综合加权（只统计有分数的维度，重归一化权重）
    active = {d: s for d, s in scores.items() if s is not None and d in WEIGHTS}
    if not active:
        combined = 0.0
    else:
        total_w = sum(WEIGHTS.get(d, 0) for d in active)
        if total_w > 0:
            combined = sum(s * (WEIGHTS.get(d, 0) / total_w) for d, s in active.items())
        else:
            combined = float(np.mean(list(active.values())))

    return scores, descs, round(combined, 2)


# ========== 7. 执行测试（多次运行） ==========

total_api_calls = sum(len(tc["user_messages"]) for tc in test_cases) * N_RUNS
est_minutes = total_api_calls * 21 / 60

print(f"\n--- 7. 执行对话测试（每条用例跑 {N_RUNS} 次）---")
print(f"  限流：21 秒/次（内置在 API 层）")
print(f"  预计总 API 调用：{total_api_calls} 次")
print(f"  预计总耗时：约 {est_minutes:.0f} 分钟")

test_results = []

for tc in test_cases:
    print(f"\n  [{tc['id']}] 维度：{tc['dimension']}")
    print(f"    对话轮数：{len(tc['user_messages'])} 轮 × {N_RUNS} 次运行")

    run_results = []

    for run_idx in range(N_RUNS):
        print(f"    ── 第 {run_idx + 1}/{N_RUNS} 次运行 ──")

        dialog_mgr.reset()

        for i, msg in enumerate(tc["user_messages"]):
            response = dialog_mgr.send(msg)
            if run_idx == 0:  # 第一次运行打印详细日志
                print(f"      用户[{i+1}]：{msg[:40]}...")
                print(f"      AI[{i+1}]：{response[:50]}...")

        log = dialog_mgr.dialog_log

        scores, descs, combined = calculate_scores(tc, log)

        run_results.append({
            "run": run_idx + 1,
            "scores": scores,
            "combined": combined,
        })

        if run_idx > 0:
            print(f"      综合得分：{combined:.2f}")

    # ---- 多次运行统计 ----
    combined_scores = [r["combined"] for r in run_results]
    mean_score = float(np.mean(combined_scores))
    std_score = float(np.std(combined_scores))
    min_score = float(np.min(combined_scores))
    max_score = float(np.max(combined_scores))
    pass_rate = sum(1 for s in combined_scores if s >= 0.8) / N_RUNS

    # 各维度统计
    dim_stats = {}
    for d in WEIGHTS:
        d_scores = [r["scores"].get(d) for r in run_results if r["scores"].get(d) is not None]
        if d_scores:
            dim_stats[d] = {
                "mean": round(float(np.mean(d_scores)), 2),
                "std": round(float(np.std(d_scores)), 2),
                "scores": d_scores,
            }

    # 稳定性判定
    if mean_score >= 0.8 and std_score <= 0.15:
        stability = "[PASS]"
    elif mean_score >= 0.8 and std_score > 0.15:
        stability = "[WARN]"
    elif mean_score < 0.8 and std_score <= 0.15:
        stability = "[WARN]"
    else:
        stability = "[FAIL]"

    test_results.append({
        "id": tc["id"],
        "dimension": tc["dimension"],
        "mean": round(mean_score, 2),
        "std": round(std_score, 2),
        "min": round(min_score, 2),
        "max": round(max_score, 2),
        "pass_rate": round(pass_rate, 2),
        "stability": stability,
        "dim_stats": dim_stats,
        "run_details": run_results,
        "turns": len(tc["user_messages"]),
    })

    # 打印统计
    print(f"\n    统计结果（{N_RUNS} 次运行）：")
    print(f"      平均分：{mean_score:.2f}  标准差：{std_score:.2f}  "
          f"最低：{min_score:.2f}  最高：{max_score:.2f}")
    print(f"      通过率：{pass_rate:.0%}  稳定性：{stability}")

print(f"\n  测试完成！共 {total_api_calls} 次 API 调用")


# ========== 8. 测试报告 ==========

print(f"\n{'=' * 60}")
print(f"多轮对话测试报告（{N_RUNS} 次运行统计）")
print(f"{'=' * 60}")

total = len(test_results)
avg_mean = np.mean([r["mean"] for r in test_results])
avg_std = np.mean([r["std"] for r in test_results])
avg_pass_rate = np.mean([r["pass_rate"] for r in test_results])

pass_count = sum(1 for r in test_results if r["stability"] == "[PASS]")
warn_count = sum(1 for r in test_results if r["stability"] == "[WARN]")
fail_count = sum(1 for r in test_results if r["stability"] == "[FAIL]")

print(f"\n  测试用例数：{total}（每条 {N_RUNS} 次运行）")
print(f"  [PASS]：{pass_count} 个")
print(f"  [WARN]：{warn_count} 个")
print(f"  [FAIL]：{fail_count} 个")

print(f"\n  整体统计：")
print(f"    平均分：{avg_mean:.2f} / 1.00")
print(f"    平均标准差：{avg_std:.2f}（越小越稳定）")
print(f"    平均通过率：{avg_pass_rate:.0%}")

# 各维度得分
print(f"\n  各维度得分（多次运行平均）：")
print(f"  {'维度':<14s} {'平均分':>8s} {'标准差':>8s} {'用例数':>6s} {'判定':>10s}")
print(f"  {'─' * 50}")

dim_scores = {}
for r in test_results:
    for d, stats in r["dim_stats"].items():
        dim_scores.setdefault(d, []).append(stats["mean"])

for d in WEIGHTS:
    scores_list = dim_scores.get(d, [])
    if scores_list:
        a = np.mean(scores_list)
        s = np.std(scores_list) if len(scores_list) > 1 else 0.0
        status = "[PASS]" if a >= 0.8 else "[WARN]" if a >= 0.5 else "[FAIL]"
        print(f"  {d:<14s} {a:>8.2f} {s:>8.2f} {len(scores_list):>6d} {status:>10s}")
    else:
        print(f"  {d:<14s} {'N/A':>8s} {'N/A':>8s} {0:>6d} {'[无用例]':>10s}")

# 逐条明细
print(f"\n  逐条明细（多次运行统计）：")
print(f"  {'用例':<8s} {'维度':<14s} {'轮数':>4s} {'平均':>6s} {'标准差':>8s} "
      f"{'最低':>6s} {'最高':>6s} {'通过率':>8s} {'稳定性':>10s}")
print(f"  {'─' * 80}")

for r in test_results:
    print(f"  {r['id']:<8s} {r['dimension']:<14s} {r['turns']:>4d} "
          f"{r['mean']:>6.2f} {r['std']:>8.2f} {r['min']:>6.2f} {r['max']:>6.2f} "
          f"{r['pass_rate']:>7.0%} {r['stability']:>10s}")

# 稳定性分析
print(f"\n  稳定性分析：")
unstable = [r for r in test_results if r["std"] > 0.15]
if unstable:
    print(f"    [关注] 以下用例运行不稳定（标准差 > 0.15）：")
    for r in unstable:
        print(f"      {r['id']}（{r['dimension']}）：std={r['std']:.2f}, "
              f"范围 [{r['min']:.2f}~{r['max']:.2f}]")
        print(f"        → LLM 输出随机性较大，建议增加运行次数或优化 Prompt 约束")
else:
    print(f"    [PASS] 所有用例运行稳定（标准差 ≤ 0.15）")

# 最弱维度
if dim_scores:
    worst_dim = min(
        [(d, np.mean(s)) for d, s in dim_scores.items()],
        key=lambda x: x[1]
    )
    if worst_dim[1] < 0.8:
        print(f"\n  最弱维度：{worst_dim[0]}（{worst_dim[1]:.2f}）")
        advice = {
            "上下文记忆": "优化上下文窗口管理，检查历史消息是否被截断",
            "话题连贯性": "在 Prompt 中增加'记住之前的话题'的约束",
            "回答一致性": "增加事实缓存机制，同一问题的回答保持一致",
            "对话状态管理": "引入结构化状态追踪，确保改口后旧值被覆盖",
            "情绪感知": "在 Prompt 中增加情绪识别指引",
            "模糊澄清": "在 Prompt 中明确'不确定时主动追问'",
        }
        print(f"    建议：{advice.get(worst_dim[0], '人工复核对话日志')}")

# 保存报告
report = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_runs": N_RUNS,
    "total_tests": total,
    "avg_mean": round(float(avg_mean), 2),
    "avg_std": round(float(avg_std), 2),
    "avg_pass_rate": round(float(avg_pass_rate), 2),
    "dim_scores": {d: round(float(np.mean(s)), 2) for d, s in dim_scores.items()},
    "results": [
        {"id": r["id"], "dimension": r["dimension"],
         "mean": r["mean"], "std": r["std"], "pass_rate": r["pass_rate"],
         "stability": r["stability"]}
        for r in test_results
    ],
}

with open("dialog_test_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n  报告已保存到 dialog_test_report.json")


# ========== 9. 可视化 ==========

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.makedirs("charts", exist_ok=True)

print(f"\n--- 9. 可视化 ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：各维度得分
if dim_scores:
    dims = [d for d in WEIGHTS if d in dim_scores]
    means = [np.mean(dim_scores[d]) for d in dims]
    colors = ["#2ECC71" if s >= 0.8 else "#F39C12" if s >= 0.5 else "#E74C3C" for s in means]

    bars = axes[0].bar(dims, means, color=colors, edgecolor="gray", alpha=0.85, width=0.5)
    for bar, score in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{score:.2f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

    axes[0].axhline(y=0.8, color="green", linestyle=":", alpha=0.5, label="PASS（0.8）")
    axes[0].axhline(y=0.5, color="orange", linestyle=":", alpha=0.5, label="WARN（0.5）")
    axes[0].set_ylabel("平均得分", fontsize=12)
    axes[0].set_title("各维度平均得分", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")

# 右图：各用例的稳定性（平均分 ± 标准差）
case_ids = [r["id"] for r in test_results]
means = [r["mean"] for r in test_results]
stds = [r["std"] for r in test_results]

x_pos = range(len(case_ids))
bars = axes[1].bar(x_pos, means, yerr=stds, capsize=5,
                   color=["#2ECC71" if r["stability"] == "[PASS]" else
                          "#F39C12" if r["stability"] == "[WARN]" else "#E74C3C"
                          for r in test_results],
                   edgecolor="gray", alpha=0.85, width=0.5)

axes[1].axhline(y=0.8, color="green", linestyle=":", alpha=0.5, label="PASS（0.8）")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(case_ids, fontsize=10)
axes[1].set_ylabel("综合得分（平均 ± 标准差）", fontsize=12)
axes[1].set_title(f"各用例稳定性（{N_RUNS} 次运行）", fontsize=13, fontweight="bold")
axes[1].set_ylim(0, 1.3)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis="y")

plt.suptitle("多轮对话测试报告",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/dialog_test.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  测试报告图已保存到 charts/dialog_test.png")
print(f"\n  从图上能看到：")
print(f"    1. 左图：各维度平均得分，颜色对应判定等级")
print(f"    2. 右图：各用例的得分和波动范围（误差棒 = 标准差）")
print(f"    3. 误差棒越长 = 运行越不稳定，LLM 输出随机性越大")
print(f"    4. 如果某个用例误差棒特别长，建议增加运行次数或优化 Prompt")
