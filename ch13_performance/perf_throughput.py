"""
吞吐量测试
在固定时间内发送请求，统计 QPS（每秒请求数）
"""

import requests
import time
import json
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=" * 60)
print("吞吐量测试")
print("=" * 60)

API_URL = "http://localhost:5000/predict"
HEALTH_URL = "http://localhost:5000/health"
TEST_DURATION = 10
CONCURRENCY_LEVELS = [1, 5, 10, 20, 50]


def check_server():
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def single_request(text):
    """发送单个请求，返回 (成功与否, 延迟毫秒)"""
    start = time.perf_counter()
    try:
        resp = requests.post(API_URL, json={"text": text}, timeout=30)
        resp.raise_for_status()
        latency = (time.perf_counter() - start) * 1000
        return True, latency
    except Exception:
        return False, 0


def run_throughput_test(concurrency, text, duration):
    """
    在指定并发级别下运行吞吐量测试

    参数：
        concurrency: 并发线程数
        text: 测试文本
        duration: 测试持续时间（秒）

    返回：
        测试结果字典
    """
    success_count = 0
    fail_count = 0
    latencies = []
    start_time = time.time()

    def worker():
        """工作线程：不断发送请求直到超时"""
        nonlocal success_count, fail_count
        while time.time() - start_time < duration:
            success, latency = single_request(text)
            if success:
                success_count += 1
                latencies.append(latency)
            else:
                fail_count += 1

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        for f in as_completed(futures):
            f.result()

    actual_duration = time.time() - start_time
    total_requests = success_count + fail_count
    qps = success_count / actual_duration if actual_duration > 0 else 0

    if latencies:
        arr = np.array(latencies)
        latency_stats = {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }
    else:
        latency_stats = None

    return {
        "concurrency": concurrency,
        "duration": round(actual_duration, 2),
        "total_requests": total_requests,
        "success": success_count,
        "failed": fail_count,
        "error_rate": round(fail_count / total_requests * 100, 2) if total_requests > 0 else 0,
        "qps": round(qps, 2),
        "latency": latency_stats,
    }


# ========== 主程序 ==========
if __name__ == "__main__":
    if not check_server():
        print("服务不在线！请先运行：python model_server.py")
        exit()

    text = "这个产品的质量非常好，值得推荐"
    all_results = []

    for concurrency in CONCURRENCY_LEVELS:
        print(f"\n--- 并发级别: {concurrency} ---")
        print(f"持续时间: {TEST_DURATION} 秒")
        print(f"测试文本: '{text}'")

        result = run_throughput_test(concurrency, text, TEST_DURATION)
        all_results.append(result)

        print(f"  总请求数:  {result['total_requests']}")
        print(f"  成功:      {result['success']}")
        print(f"  失败:      {result['failed']}")
        print(f"  错误率:    {result['error_rate']}%")
        print(f"  QPS:       {result['qps']}")

        if result["latency"]:
            lat = result["latency"]
            print(f"  平均延迟:  {lat['mean']:.2f} ms")
            print(f"  P50:       {lat['p50']:.2f} ms")
            print(f"  P95:       {lat['p95']:.2f} ms")
            print(f"  P99:       {lat['p99']:.2f} ms")

    # 汇总对比
    print(f"\n{'=' * 60}")
    print(f"吞吐量汇总对比")
    print(f"{'=' * 60}")
    print(f"\n  {'并发':>6s} {'总请求':>8s} {'QPS':>8s} {'错误率':>8s} {'均值延迟':>10s} {'P95延迟':>10s}")
    print(f"  {'─' * 58}")

    for r in all_results:
        lat = r["latency"]
        mean_lat = f"{lat['mean']:.2f}ms" if lat else "N/A"
        p95_lat = f"{lat['p95']:.2f}ms" if lat else "N/A"
        print(f"  {r['concurrency']:>6d} {r['total_requests']:>8d} "
              f"{r['qps']:>8.1f} {r['error_rate']:>7.1f}% "
              f"{mean_lat:>10s} {p95_lat:>10s}")

    # 保存结果
    output = {
        "test_type": "throughput",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_duration": TEST_DURATION,
        "test_text": text,
        "results": all_results
    }

    with open("perf_throughput_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到 perf_throughput_results.json")
    print(f"吞吐量测试完成！")
