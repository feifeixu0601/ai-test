"""
延迟测试
对模型 API 重复发送单个请求，统计延迟分布

测试内容：
  1. 基础延迟统计（均值、中位数、百分位数）
  2. 不同输入长度的延迟对比
  3. 延迟分布可视化
"""

import requests
import time
import numpy as np
import json
import os
from datetime import datetime

os.makedirs("charts", exist_ok=True)

# 常量
API_URL = "http://localhost:5000/predict"
HEALTH_URL = "http://localhost:5000/health"
NUM_REQUESTS = 100

print("=" * 60)
print("延迟测试")
print("=" * 60)


def check_server():
    """检查服务是否在线"""
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        if resp.status_code == 200:
            print(f"服务在线，准备开始测试")
            return True
    except Exception:
        pass
    print(f"服务不在线！请先在另一个终端运行：python model_server.py")
    return False


def measure_latency(text, num_requests=NUM_REQUESTS):
    """发送指定文本的请求 N 次，记录每次延迟（毫秒）"""
    latencies = []

    for i in range(num_requests):
        start = time.perf_counter()
        try:
            resp = requests.post(API_URL, json={"text": text}, timeout=30)
            resp.raise_for_status()
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        except Exception as e:
            print(f"  请求 {i+1} 失败: {e}")

    return latencies


def calc_latency_stats(latencies):
    """计算延迟统计指标"""
    if not latencies:
        return None

    arr = np.array(latencies)
    return {
        "count": len(latencies),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def print_stats(name, stats):
    """打印延迟统计结果"""
    if not stats:
        print(f"  {name}: 无有效数据")
        return

    print(f"\n--- {name} ---")
    print(f"  请求数:   {stats['count']}")
    print(f"  平均值:   {stats['mean']:.2f} ms")
    print(f"  中位数:   {stats['median']:.2f} ms  (P50)")
    print(f"  P90:      {stats['p90']:.2f} ms")
    print(f"  P95:      {stats['p95']:.2f} ms")
    print(f"  P99:      {stats['p99']:.2f} ms")
    print(f"  最小值:   {stats['min']:.2f} ms")
    print(f"  最大值:   {stats['max']:.2f} ms")
    print(f"  标准差:   {stats['std']:.2f} ms")


# ========== 主程序 ==========
if __name__ == "__main__":
    if not check_server():
        exit()

    all_results = {}

    # 测试1：短文本延迟
    print(f"\n{'=' * 60}")
    print(f"测试1：短文本延迟（约 5 个字符）")
    print(f"{'=' * 60}")

    short_text = "东西很好"
    print(f"测试文本：'{short_text}'（{len(short_text)} 字符）")
    print(f"发送 {NUM_REQUESTS} 个请求...")
    short_latencies = measure_latency(short_text)
    short_stats = calc_latency_stats(short_latencies)
    print_stats("短文本延迟", short_stats)
    all_results["short_text"] = {"text": short_text, "stats": short_stats}

    # 测试2：中等文本延迟
    print(f"\n{'=' * 60}")
    print(f"测试2：中等文本延迟（约 50 个字符）")
    print(f"{'=' * 60}")

    medium_text = "这个产品的质量非常好，包装精美，物流速度也很快，客服态度很好，值得推荐给大家购买"
    print(f"测试文本：'{medium_text[:30]}...'（{len(medium_text)} 字符）")
    print(f"发送 {NUM_REQUESTS} 个请求...")
    medium_latencies = measure_latency(medium_text)
    medium_stats = calc_latency_stats(medium_latencies)
    print_stats("中等文本延迟", medium_stats)
    all_results["medium_text"] = {"text": medium_text[:30], "stats": medium_stats}

    # 测试3：长文本延迟
    print(f"\n{'=' * 60}")
    print(f"测试3：长文本延迟（约 500 个字符）")
    print(f"{'=' * 60}")

    long_text = "这个产品的质量非常好" * 50
    print(f"测试文本：'{long_text[:30]}...'（{len(long_text)} 字符）")
    print(f"发送 {NUM_REQUESTS} 个请求...")
    long_latencies = measure_latency(long_text)
    long_stats = calc_latency_stats(long_latencies)
    print_stats("长文本延迟", long_stats)
    all_results["long_text"] = {"text": long_text[:30], "stats": long_stats}

    # 汇总对比
    print(f"\n{'=' * 60}")
    print(f"延迟汇总对比")
    print(f"{'=' * 60}")
    print(f"\n  {'输入类型':<12s} {'均值':>8s} {'P50':>8s} {'P95':>8s} {'P99':>8s} {'最大值':>8s}")
    print(f"  {'─' * 58}")

    for name, data in all_results.items():
        label = {"short_text": "短文本", "medium_text": "中等文本", "long_text": "长文本"}
        s = data["stats"]
        if s:
            print(f"  {label[name]:<12s} {s['mean']:>7.2f}m {s['median']:>7.2f}m "
                  f"{s['p95']:>7.2f}m {s['p99']:>7.2f}m {s['max']:>7.2f}m")

    # ========== 延迟分布可视化 ==========
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：三种输入长度的延迟分布直方图
    all_latencies = {
        "短文本": short_latencies,
        "中等文本": medium_latencies,
        "长文本": long_latencies,
    }
    colors = ["#2E86C1", "#F39C12", "#E74C3C"]
    for (label, lats), color in zip(all_latencies.items(), colors):
        axes[0].hist(lats, bins=20, alpha=0.5, color=color, label=label, density=True)
    axes[0].set_xlabel("延迟 (ms)", fontsize=12)
    axes[0].set_ylabel("密度", fontsize=12)
    axes[0].set_title("不同输入长度的延迟分布", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右图：箱线图对比
    axes[1].boxplot(
        [short_latencies, medium_latencies, long_latencies],
        patch_artist=True,
        boxprops=dict(facecolor="#AED6F1", color="#2E86C1"),
        medianprops=dict(color="#E74C3C", linewidth=2),
    )
    axes[1].set_xticklabels(["短文本", "中等文本", "长文本"])
    axes[1].set_ylabel("延迟 (ms)", fontsize=12)
    axes[1].set_title("延迟箱线图对比", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("charts/latency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n延迟分布图已保存到 charts/latency_distribution.png")

    # 保存结果
    output = {
        "test_type": "latency",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_requests_per_test": NUM_REQUESTS,
        "results": all_results
    }

    with open("perf_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n测试结果已保存到 perf_results.json")
    print(f"\n延迟测试完成！")
