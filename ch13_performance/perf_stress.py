"""
并发压力测试
持续高并发请求，监控系统稳定性
"""

import requests
import time
import json
import psutil
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

print("=" * 60)
print("并发压力测试")
print("=" * 60)

API_URL = "http://localhost:5000/predict"
HEALTH_URL = "http://localhost:5000/health"

STRESS_CONCURRENCY = 20
STRESS_DURATION = 30
MONITOR_INTERVAL = 5


def check_server():
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def find_flask_process():
    """找到 Flask 服务的进程（用于监控资源占用）"""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "model_server" in cmdline:
                return psutil.Process(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def monitor_resources(flask_proc, duration, interval):
    """在测试期间监控资源占用"""
    snapshots = []
    start = time.time()

    while time.time() - start < duration:
        try:
            cpu_percent = flask_proc.cpu_percent(interval=1)
            memory_info = flask_proc.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            snapshot = {
                "time": round(time.time() - start, 1),
                "cpu_percent": round(cpu_percent, 1),
                "memory_mb": round(memory_mb, 1),
            }
            snapshots.append(snapshot)

            time.sleep(interval - 1)
        except Exception:
            break

    return snapshots


def stress_worker(text, duration, results):
    """压力测试工作线程"""
    start_time = time.time()

    while time.time() - start_time < duration:
        req_start = time.perf_counter()
        try:
            resp = requests.post(API_URL, json={"text": text}, timeout=30)
            resp.raise_for_status()
            latency = (time.perf_counter() - req_start) * 1000
            results["success"] += 1
            results["latencies"].append(latency)
        except requests.exceptions.Timeout:
            results["timeout"] += 1
        except Exception:
            results["failed"] += 1


# ========== 主程序 ==========
if __name__ == "__main__":
    if not check_server():
        print("服务不在线！请先运行：python model_server.py")
        exit()

    text = "这个产品质量很好，服务也不错，推荐购买"
    print(f"\n压力测试参数：")
    print(f"  并发数: {STRESS_CONCURRENCY}")
    print(f"  持续时间: {STRESS_DURATION} 秒")
    print(f"  监控间隔: {MONITOR_INTERVAL} 秒")

    flask_proc = find_flask_process()
    if flask_proc:
        print(f"  找到 Flask 进程: PID {flask_proc.pid}")
    else:
        print(f"  未找到 Flask 进程，跳过资源监控")

    print(f"\n{'=' * 60}")
    print(f"开始压力测试...")
    print(f"{'=' * 60}")

    results = {"success": 0, "timeout": 0, "failed": 0, "latencies": []}
    resource_snapshots = []

    if flask_proc:
        from threading import Thread
        monitor_thread = Thread(
            target=lambda: resource_snapshots.extend(
                monitor_resources(flask_proc, STRESS_DURATION, MONITOR_INTERVAL)
            )
        )
        monitor_thread.start()

    stress_start = time.time()
    with ThreadPoolExecutor(max_workers=STRESS_CONCURRENCY) as executor:
        futures = [
            executor.submit(stress_worker, text, STRESS_DURATION, results)
            for _ in range(STRESS_CONCURRENCY)
        ]
        for f in futures:
            f.result()

    actual_duration = time.time() - stress_start

    if flask_proc:
        monitor_thread.join()

    # 结果分析
    total = results["success"] + results["timeout"] + results["failed"]
    qps = results["success"] / actual_duration if actual_duration > 0 else 0

    print(f"\n{'─' * 60}")
    print(f"压力测试结果")
    print(f"{'─' * 60}")
    print(f"  实际持续时间:  {actual_duration:.1f} 秒")
    print(f"  总请求数:      {total}")
    print(f"  成功:          {results['success']}")
    print(f"  超时:          {results['timeout']}")
    print(f"  失败:          {results['failed']}")
    print(f"  错误率:        {(results['timeout'] + results['failed']) / total * 100:.2f}%")
    print(f"  QPS:           {qps:.2f}")

    if results["latencies"]:
        arr = np.array(results["latencies"])
        print(f"\n  延迟分布:")
        print(f"    平均值:  {np.mean(arr):.2f} ms")
        print(f"    P50:     {np.percentile(arr, 50):.2f} ms")
        print(f"    P95:     {np.percentile(arr, 95):.2f} ms")
        print(f"    P99:     {np.percentile(arr, 99):.2f} ms")
        print(f"    最大值:  {np.max(arr):.2f} ms")

    if resource_snapshots:
        print(f"\n  资源占用监控:")
        print(f"  {'时间(s)':>8s} {'CPU%':>8s} {'内存(MB)':>10s}")
        print(f"  {'─' * 28}")
        for s in resource_snapshots:
            print(f"  {s['time']:>8.1f} {s['cpu_percent']:>8.1f} {s['memory_mb']:>10.1f}")

        cpu_values = [s["cpu_percent"] for s in resource_snapshots]
        mem_values = [s["memory_mb"] for s in resource_snapshots]
        print(f"\n  CPU 平均: {np.mean(cpu_values):.1f}%  峰值: {max(cpu_values):.1f}%")
        print(f"  内存平均: {np.mean(mem_values):.1f}MB  峰值: {max(mem_values):.1f}MB")

    # 稳定性判定
    print(f"\n  稳定性判定:")
    error_rate = (results["timeout"] + results["failed"]) / total * 100 if total > 0 else 0

    if error_rate < 1 and results["timeout"] == 0:
        print(f"    [PASS] 系统稳定，在 {STRESS_CONCURRENCY} 并发下无异常")
    elif error_rate < 5:
        print(f"    [WARN] 系统基本稳定，但有少量错误（{error_rate:.1f}%）")
    else:
        print(f"    [FAIL] 系统不稳定，错误率 {error_rate:.1f}%，需要降级或扩容")

    if results["latencies"]:
        p99 = np.percentile(arr, 99)
        if p99 < 500:
            print(f"    [PASS] P99 延迟 {p99:.0f}ms，用户体感流畅")
        elif p99 < 2000:
            print(f"    [WARN] P99 延迟 {p99:.0f}ms，用户可能感到等待")
        else:
            print(f"    [FAIL] P99 延迟 {p99:.0f}ms，用户体验极差")

    # 保存结果
    output = {
        "test_type": "stress",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "concurrency": STRESS_CONCURRENCY,
        "duration": round(actual_duration, 2),
        "total_requests": total,
        "success": results["success"],
        "timeout": results["timeout"],
        "failed": results["failed"],
        "error_rate": round(error_rate, 2),
        "qps": round(qps, 2),
        "latency_ms": {
            "mean": round(float(np.mean(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": round(float(np.max(arr)), 2),
        } if results["latencies"] else None,
        "resource_snapshots": resource_snapshots,
    }

    with open("perf_stress_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到 perf_stress_results.json")
    print(f"压力测试完成！")
