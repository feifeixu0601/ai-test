"""
线上 AI 质量监控（最简版）

每天跑一次（cron 定时），输出质量指标，异常时发告警。

指标分两类：
  全量指标（延迟/成本/点踩）：读日志算数字，零 API 成本
  抽样指标（幻觉）：调裁判模型，每天抽样 5%

注意：
  1. 复用第 24 篇 ai_helpers.py 的 call_simple（API 调用）
  2. 裁判解析失败返回 None 并剔除分母，不兜默认值
     （避免把裁判故障伪装成"幻觉率正常"）
  3. 监控自身也要被监控：裁判大面积失败时发告警
  4. 调试时注意别手动连跑，否则 metrics_history.json 会写入重复日期记录
"""

import sys
import json
import os
import random
import numpy as np
from datetime import datetime

# 复用第 24 篇的 API 调用函数
# 独立运行时请把 call_simple 的实现复制过来，或确保 ai_helpers.py 在同目录
from ai_helpers import call_simple

# 成本换算（根据你的平台单价修改）
TOKEN_PRICE_PER_1K = 0.012  # 每 1000 token 的费用（元）
DAILY_BUDGET = 33           # 日预算（元），根据你的实际预算改

# 幻觉判定阈值：裁判 score < 0.5 算一条幻觉
HALLUC_THRESHOLD = 0.5


# ========== 1. 读取线上日志 ==========

def load_online_logs(log_file="online_logs.json"):
    """
    读取线上回答日志

    容错：
      - 文件损坏时返回空列表（不崩溃）
      - 字段缺失的记录跳过（不混入统计）
    """
    if not os.path.exists(log_file):
        return []

    try:
        with open(log_file, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  [警告] 日志文件损坏或无法读取：{e}")
        return []

    # 过滤掉格式不对的记录
    valid = [r for r in data if "question" in r and "answer" in r]
    if len(valid) < len(data):
        print(f"  [警告] {len(data) - len(valid)} 条日志格式异常，已跳过")
    return valid


# ========== 2. 计算全量指标（延迟/成本/点踩，零成本） ==========

def calc_metrics(responses):
    """
    计算全量指标：延迟、成本、用户反馈

    这些指标从全量日志计算，不需要采样（读日志零成本）。
    幻觉检测另走抽样（见 check_hallucination）。
    """
    if not responses:
        return None

    total = len(responses)

    # 延迟
    times = sorted(r.get("response_time", 0) for r in responses)
    p95_idx = min(int(total * 0.95), total - 1)
    p99_idx = min(int(total * 0.99), total - 1)

    # 成本（用 token 数 × 单价换算）
    total_tokens = sum(r.get("total_tokens", 0) for r in responses)
    est_cost = total_tokens / 1000 * TOKEN_PRICE_PER_1K

    return {
        "total": total,
        "avg_response_time": sum(times) / total,
        "p95_response_time": times[p95_idx],
        "p99_response_time": times[p99_idx],
        "dislike_rate": sum(1 for r in responses if r.get("user_dislike")) / total,
        "total_tokens": total_tokens,
        "est_cost": round(est_cost, 2),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }


# ========== 3. 幻觉检测（抽样，有 API 费用） ==========

def check_hallucination(question, answer, context=""):
    """
    线上幻觉检测：用裁判模型（LLM-as-judge）判定

    返回：
        score: 0~1（1.0=完全可信，0.0=明显幻觉）
        None: 裁判解析失败（不参与统计，不兜默认值）
    """
    if context:
        judge_prompt = f"""你是事实核查员。请判断以下回答是否忠于检索到的参考资料。

参考资料：
{context}

问题：{question}
回答：{answer}

判断标准：
- 回答中的事实是否忠于参考资料
- 是否存在参考资料中没有的编造内容
- 数字、日期、名称是否准确

只输出 JSON（不要输出其他内容）：
{{"score": 0.9, "reason": "一句话理由"}}"""
    else:
        judge_prompt = f"""你是事实核查员。请判断以下回答是否存在幻觉。

问题：{question}
回答：{answer}

判断标准：
- 回答中的事实是否可以被公认知识支持
- 是否存在编造的数字、日期、名称
- 是否存在自相矛盾

只输出 JSON（不要输出其他内容）：
{{"score": 0.9, "reason": "一句话理由"}}"""

    result = call_simple(judge_prompt)
    if not result:
        return None  # API 失败，不参与统计

    # 剥掉 ```json 围栏（LLM 常见输出格式）
    text = result.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        parsed = json.loads(text)
        score = parsed.get("score")
        if score is None:
            return None  # 裁判没按格式回，剔除
        return float(score)
    except (json.JSONDecodeError, ValueError):
        return None  # 解析失败，不参与统计


def smart_sample(responses, rate=0.05):
    """
    智能采样：随机抽样 + 用户反馈触发

    只用于幻觉检测（有 API 费用）。免费指标全量统计，不用采样。
    """
    sampled = [r for r in responses if r.get("user_dislike")]
    remaining = [r for r in responses if not r.get("user_dislike")]
    sample_size = max(1, int(len(remaining) * rate))
    sampled += random.sample(remaining, min(sample_size, len(remaining)))
    return sampled


# ========== 4. 异常检测（绝对值红线 + 变化率） ==========

def check_anomalies(today_metrics, halluc_rate=None, history_file="metrics_history.json"):
    """
    异常检测：绝对值红线 + 变化率

    两条规则独立、任一命中就告警：
      绝对值：超过阈值就告警（不需要历史，冷启动可用）
      变化率：突然变了才告警（全量维度 ≥7 天历史，幻觉维度 ≥3 天）
    """
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    alerts = []

    # ---- 绝对值红线（不需要历史，无条件执行）----

    if today_metrics["p95_response_time"] > 10:
        alerts.append({
            "level": "P1",
            "message": f"P95 延迟超红线：{today_metrics['p95_response_time']:.1f} 秒（阈值 10 秒）"
        })

    if today_metrics["p99_response_time"] > 30:
        alerts.append({
            "level": "P1",
            "message": f"P99 延迟超红线：{today_metrics['p99_response_time']:.1f} 秒（阈值 30 秒）"
        })

    if today_metrics["dislike_rate"] > 0.25:
        alerts.append({
            "level": "P1",
            "message": f"点踩率超红线：{today_metrics['dislike_rate']:.1%}（阈值 25%）"
        })

    if today_metrics["est_cost"] > DAILY_BUDGET * 1.5:
        alerts.append({
            "level": "P2",
            "message": f"日成本超预算 50%：¥{today_metrics['est_cost']:.2f}"
                       f"（预算 ¥{DAILY_BUDGET}，红线 ¥{DAILY_BUDGET * 1.5:.0f}）"
        })

    if halluc_rate is not None and halluc_rate > 0.15:
        alerts.append({
            "level": "P1",
            "message": f"幻觉率超红线：{halluc_rate:.1%}（阈值 15%）"
        })

    # ---- 变化率异常（需要历史数据）----

    # 全量维度：≥7 天历史（数据充足，变化率判定可靠）
    if len(history) >= 7:
        recent = history[-7:]

        avg_dislike = sum(m["dislike_rate"] for m in recent) / len(recent)
        if today_metrics["dislike_rate"] > avg_dislike * 2:
            alerts.append({
                "level": "P1",
                "message": f"点踩率突增：今天 {today_metrics['dislike_rate']:.1%}，"
                           f"近 7 天均值 {avg_dislike:.1%}"
            })

        avg_p95 = sum(m["p95_response_time"] for m in recent) / len(recent)
        if today_metrics["p95_response_time"] > avg_p95 * 2:
            alerts.append({
                "level": "P1",
                "message": f"P95 延迟突增：今天 {today_metrics['p95_response_time']:.1f}s，"
                           f"近 7 天均值 {avg_p95:.1f}s"
            })

    # 幻觉维度：≥3 天有效历史（抽样+剔除后有效天少，门槛降低）
    # 用 ×2.5 或绝对增量 >5pp 双条件（抽样方差大，×2 容易误报）
    if halluc_rate is not None:
        halluc_hist = [m["hallucination_rate"] for m in history[-7:]
                       if m.get("hallucination_rate") is not None]
        if len(halluc_hist) >= 3:
            avg_halluc = sum(halluc_hist) / len(halluc_hist)
            if avg_halluc > 0 and (halluc_rate > avg_halluc * 2.5
                                   or halluc_rate - avg_halluc > 0.05):
                alerts.append({
                    "level": "P1",
                    "message": f"幻觉率突增：今天 {halluc_rate:.1%}，"
                               f"近 {len(halluc_hist)} 天均值 {avg_halluc:.1%}"
                })

    # 保存历史（只保留 30 天；幻觉率 None 时不写入，避免污染下次均值）
    if halluc_rate is not None:
        today_metrics["hallucination_rate"] = halluc_rate
    history.append(today_metrics)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history[-30:], f, ensure_ascii=False, indent=2)

    return alerts


# ========== 5. 发送告警 ==========

def send_alerts(alerts):
    """按级别发送告警"""
    for a in alerts:
        if a["level"] == "P0":
            print(f"  🚨 P0 紧急：{a['message']}")
        elif a["level"] == "P1":
            print(f"  ⚠️ P1 高：{a['message']}")
        elif a["level"] == "P2":
            print(f"  📢 P2 中：{a['message']}")
        else:
            print(f"  ℹ️ {a['level']}：{a['message']}")


# ========== 6. 主流程 ==========

if __name__ == "__main__":
    print(f"📊 线上质量监控 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    responses = load_online_logs()
    if not responses:
        print("  无线上日志，跳过")
        sys.exit()

    # ---- 全量指标（延迟/成本/点踩，零成本）----
    metrics = calc_metrics(responses)
    print(f"  总回答数：{metrics['total']}")
    print(f"  平均延迟：{metrics['avg_response_time']:.1f}s")
    print(f"  P95 延迟：{metrics['p95_response_time']:.1f}s")
    print(f"  P99 延迟：{metrics['p99_response_time']:.1f}s")
    print(f"  点踩率：{metrics['dislike_rate']:.1%}")
    print(f"  总 token：{metrics['total_tokens']}")
    print(f"  预估成本：¥{metrics['est_cost']}")

    # ---- 幻觉检测（抽样，有 API 费用）----
    halluc_rate = None  # 先初始化，防止 NameError
    sample = smart_sample(responses, rate=0.05)
    print(f"\n  幻觉检测采样：{len(sample)}/{len(responses)} 条")

    halluc_scores = []
    for r in sample:
        score = check_hallucination(
            r["question"],
            r["answer"],
            context=r.get("context", "")  # RAG 产品传 context，纯 LLM 传空
        )
        halluc_scores.append(score)
        if score is not None and score < HALLUC_THRESHOLD:
            print(f"    ⚠️ 疑似幻觉（{score:.2f}）：{r['question'][:30]}...")

    # 用有效样本算幻觉率（解析失败的剔除分母）
    # failed_count 只在这里定义一次，后面的裁判失效块直接引用
    valid_scores = [s for s in halluc_scores if s is not None]
    failed_count = len(halluc_scores) - len(valid_scores)

    if valid_scores:
        halluc_count = sum(1 for s in valid_scores if s < HALLUC_THRESHOLD)
        halluc_rate = halluc_count / len(valid_scores)
        metrics["hallucination_rate"] = round(halluc_rate, 3)
        print(f"  幻觉率：{halluc_rate:.1%}（{halluc_count}/{len(valid_scores)}）")

        if failed_count > 0:
            print(f"  [提示] {failed_count} 条裁判解析失败，已剔除分母")

    # ---- 异常检测（先算好 alerts，裁判失效块往里 append）----
    alerts = check_anomalies(metrics, halluc_rate=halluc_rate)

    # ---- 裁判自身故障检测（监控也要被监控）----
    if halluc_scores:
        fail_ratio = failed_count / len(halluc_scores)
        if fail_ratio > 0.2:
            alerts.append({
                "level": "P2",
                "message": f"裁判失效：{failed_count}/{len(halluc_scores)} 条无法解析"
                           f"（{fail_ratio:.0%}），本日幻觉监控不可信"
            })

    # ---- 发送告警 ----
    send_alerts(alerts)

    if not alerts:
        print("  ✅ 各项指标正常")