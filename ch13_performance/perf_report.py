"""
性能测试报告生成
汇总延迟、吞吐量、压力测试结果，生成文本 + HTML 双格式报告
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# 设置中文字体（Mac 用户改成 "PingFang SC"）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("性能测试报告生成")
print("=" * 60)


def load_results(filename):
    """加载测试结果"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  {filename} 不存在，跳过")
        return None


def chart_throughput(results):
    """画吞吐量对比图"""
    if not results:
        return

    data = results["results"]
    concurrencies = [r["concurrency"] for r in data]
    qps_values = [r["qps"] for r in data]
    error_rates = [r["error_rate"] for r in data]
    mean_latencies = [r["latency"]["mean"] if r["latency"] else 0 for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：QPS vs 并发数
    ax1.plot(concurrencies, qps_values, "bo-", linewidth=2, markersize=10, label="QPS")
    ax1.set_xlabel("并发数", fontsize=12)
    ax1.set_ylabel("QPS（每秒请求数）", fontsize=12)
    ax1.set_title("吞吐量 vs 并发数", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    for x, y in zip(concurrencies, qps_values):
        ax1.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=10)

    # 右图：延迟 vs 并发数 + 错误率
    ax2.plot(concurrencies, mean_latencies, "s-", linewidth=2, markersize=10,
             label="平均延迟 (ms)", color="#E74C3C")
    ax2.set_xlabel("并发数", fontsize=12)
    ax2.set_ylabel("平均延迟 (ms)", fontsize=12)
    ax2.set_title("延迟与错误率 vs 并发数", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.bar(concurrencies, error_rates, width=2, alpha=0.3,
             color="#F39C12", label="错误率 (%)")
    ax2b.set_ylabel("错误率 (%)", fontsize=12, color="#F39C12")
    ax2b.tick_params(axis="y", labelcolor="#F39C12")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig("charts/perf_throughput.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  吞吐量对比图已生成")


def generate_text_report(latency_data, throughput_data, stress_data):
    """生成纯文本报告"""
    lines = []

    def add(text=""):
        lines.append(text)

    add("=" * 60)
    add("AI 模型性能测试报告")
    add("=" * 60)
    add(f"")
    add(f"  测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"  被测系统：情感分类模型 API（Flask）")
    add(f"  测试环境：本地开发环境")

    # 延迟测试结果
    add(f"\n{'─' * 60}")
    add(f"一、延迟测试结果")
    add(f"{'─' * 60}")

    if latency_data and "results" in latency_data:
        for key, data in latency_data["results"].items():
            stats = data.get("stats")
            if not stats:
                continue
            label = {"short_text": "短文本", "medium_text": "中等文本", "long_text": "长文本"}.get(key, key)
            add(f"\n  [{label}]")
            add(f"    平均值: {stats['mean']:.2f}ms  P50: {stats['median']:.2f}ms  "
                f"P95: {stats['p95']:.2f}ms  P99: {stats['p99']:.2f}ms")

    # 吞吐量测试结果
    add(f"\n{'─' * 60}")
    add(f"二、吞吐量测试结果")
    add(f"{'─' * 60}")

    if throughput_data and "results" in throughput_data:
        add(f"\n  {'并发':>6s} {'QPS':>8s} {'错误率':>8s} {'P50延迟':>10s} {'P95延迟':>10s}")
        add(f"  {'─' * 46}")
        for r in throughput_data["results"]:
            lat = r.get("latency")
            p50 = f"{lat['p50']:.1f}ms" if lat else "N/A"
            p95 = f"{lat['p95']:.1f}ms" if lat else "N/A"
            add(f"  {r['concurrency']:>6d} {r['qps']:>8.1f} {r['error_rate']:>7.1f}% "
                f"{p50:>10s} {p95:>10s}")

        chart_throughput(throughput_data)

    # 压力测试结果
    add(f"\n{'─' * 60}")
    add(f"三、压力测试结果（{stress_data.get('concurrency', 'N/A') if stress_data else 'N/A'} 并发）")
    add(f"{'─' * 60}")

    if stress_data:
        add(f"\n  总请求数:  {stress_data.get('total_requests', 'N/A')}")
        add(f"  成功:      {stress_data.get('success', 'N/A')}")
        add(f"  错误率:    {stress_data.get('error_rate', 'N/A')}%")
        add(f"  QPS:       {stress_data.get('qps', 'N/A')}")

        lat = stress_data.get("latency_ms")
        if lat:
            add(f"\n  延迟: 均值 {lat['mean']}ms  P50 {lat['p50']}ms  "
                f"P95 {lat['p95']}ms  P99 {lat['p99']}ms")

        snapshots = stress_data.get("resource_snapshots", [])
        if snapshots:
            cpu_values = [s["cpu_percent"] for s in snapshots]
            mem_values = [s["memory_mb"] for s in snapshots]
            add(f"\n  资源占用: CPU 平均 {np.mean(cpu_values):.1f}% / 峰值 {max(cpu_values):.1f}%")
            add(f"            内存平均 {np.mean(mem_values):.1f}MB / 峰值 {max(mem_values):.1f}MB")

    # 综合评估
    add(f"\n{'─' * 60}")
    add(f"四、综合评估与建议")
    add(f"{'─' * 60}")

    add(f"\n  性能指标达标情况：")

    if latency_data:
        for key, data in latency_data["results"].items():
            stats = data.get("stats")
            if stats and stats.get("p95", 0) < 100:
                label = {"short_text": "短文本", "medium_text": "中等文本", "long_text": "长文本"}.get(key, key)
                add(f"    [PASS] {label} P95 延迟 {stats['p95']:.1f}ms（< 100ms，优秀）")
            elif stats:
                label = {"short_text": "短文本", "medium_text": "中等文本", "long_text": "长文本"}.get(key, key)
                add(f"    [WARN] {label} P95 延迟 {stats['p95']:.1f}ms")

    if stress_data:
        error_rate = stress_data.get("error_rate", 0)
        if error_rate < 1:
            add(f"    [PASS] 压力测试错误率 {error_rate}%（< 1%，稳定）")
        else:
            add(f"    [WARN] 压力测试错误率 {error_rate}%")

    add(f"\n  优化建议：")
    add(f"    1. 如果延迟偏高，考虑模型量化（INT8/FP16）或换用更轻量的模型")
    add(f"    2. 如果并发能力不足，考虑增加服务实例（水平扩展）或使用批处理推理")
    add(f"    3. 如果内存占用偏高，检查是否有内存泄漏（长时间运行后内存是否持续增长）")
    add(f"    4. 建议在生产环境使用 Gunicorn + 多 worker 替代 Flask 开发服务器")
    add(f"    5. 建立性能基线，每次模型更新后对比，防止性能退化")

    add(f"\n{'=' * 60}")
    add(f"报告生成完毕")

    report_text = "\n".join(lines)
    print(report_text)

    with open("perf_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存到 perf_report.txt")

    return report_text


def generate_html_report(throughput_data, stress_data):
    """生成 HTML 报告，适合发给非技术人员"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI 模型性能测试报告</title>
    <style>
        body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
        h1 {{ color: #2E86C1; border-bottom: 3px solid #2E86C1; padding-bottom: 10px; }}
        h2 {{ color: #2E86C1; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: left; }}
        th {{ background: #2E86C1; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .pass {{ color: #27AE60; font-weight: bold; }}
        .warn {{ color: #F39C12; font-weight: bold; }}
        .fail {{ color: #E74C3C; font-weight: bold; }}
        .summary {{ background: #EBF5FB; padding: 20px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>AI 模型性能测试报告</h1>
    <div class="summary">
        <p><strong>测试时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>被测系统：</strong> 情感分类模型 API</p>
        <p><strong>测试环境：</strong> 本地开发环境</p>
    </div>
"""

    # 吞吐量测试表格
    if throughput_data and "results" in throughput_data:
        html += """    <h2>吞吐量测试结果</h2>
    <table>
        <tr><th>并发数</th><th>QPS</th><th>错误率</th><th>P50 延迟</th><th>P95 延迟</th><th>状态</th></tr>\n"""

        for r in throughput_data["results"]:
            lat = r.get("latency") or {}
            p50 = f"{lat.get('p50', 0):.1f}ms" if lat else "N/A"
            p95 = f"{lat.get('p95', 0):.1f}ms" if lat else "N/A"
            err = r.get("error_rate", 0)
            if err < 1:
                status_class, status_text = "pass", "PASS"
            elif err < 5:
                status_class, status_text = "warn", "WARN"
            else:
                status_class, status_text = "fail", "FAIL"

            html += (f'        <tr><td>{r["concurrency"]}</td><td>{r["qps"]}</td>'
                     f'<td>{err}%</td><td>{p50}</td><td>{p95}</td>'
                     f'<td class="{status_class}">{status_text}</td></tr>\n')

        html += "    </table>\n"

    # 压力测试摘要
    if stress_data:
        html += f"""    <h2>压力测试结果（{stress_data.get('concurrency', 'N/A')} 并发）</h2>
    <div class="summary">
        <p><strong>总请求数：</strong> {stress_data.get('total_requests', 'N/A')}</p>
        <p><strong>错误率：</strong> {stress_data.get('error_rate', 'N/A')}%</p>
        <p><strong>QPS：</strong> {stress_data.get('qps', 'N/A')}</p>
    </div>\n"""

    html += """</body>
</html>"""

    with open("perf_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("HTML 报告已保存到 perf_report.html")


# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载各测试结果
    latency_data = load_results("perf_results.json")
    throughput_data = load_results("perf_throughput_results.json")
    stress_data = load_results("perf_stress_results.json")

    # 生成文本报告
    generate_text_report(latency_data, throughput_data, stress_data)

    # 生成 HTML 报告
    generate_html_report(throughput_data, stress_data)

    print(f"\n性能报告生成完成！")
