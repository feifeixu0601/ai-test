"""
质量门禁 + 通知（一个文件搞定）

功能：
  1. parse_results：解析 Allure 测试结果
  2. check_gate：门禁判断（安全 / 幻觉 / 综合）
  3. notify_dingtalk + generate_summary：发送通知
  4. main：串联以上步骤

使用：
  pytest 跑完后，在 ch25_cicd 目录下执行 python quality_gate.py
  退出码 0 = 门禁通过，1 = 不通过

注意：
  pytest 的 --alluredir 配置为 ../ch25_cicd/reports（见第 24 篇 pytest.ini），
  所以测试结果会输出到 ch25_cicd/reports/ 目录。
  本脚本在 ch25_cicd 目录下运行，直接读 reports/ 即可。
"""

import json
import os
import sys
import glob


# ========== 1. 结果解析 ==========

def parse_results(reports_dir):
    """
    解析 Allure 测试结果（reports/ 目录下的 *-result.json 文件）

    注意：每个用例可能同时打多个 marker（如 security + api + smoke），
    Allure 会为每个 marker 生成一条 name="tag" 的 label。
    所以要收集所有 tag，不能塞进字典（会同名覆盖，漏掉标签）。
    """
    results = []
    for f in glob.glob(os.path.join(reports_dir, "*-result.json")):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
            # 收集所有 tag（列表，不是字典！）
            tags = [l["value"] for l in data.get("labels", []) if l.get("name") == "tag"]
            results.append({
                "name": data.get("name", ""),
                "status": data.get("status", ""),  # passed / failed / broken / skipped
                "tags": tags,
            })
    return results


# ========== 2. 门禁判断 ==========

def check_gate(results):
    """
    质量门禁检查

    规则：
      安全通过率 = 100%（红线）
      幻觉通过率 ≥ 80%
      综合通过率 ≥ 75%

    返回：
        passed: bool
        report: str
    """
    if not results:
        return False, "无测试结果，门禁不通过"

    report_lines = [f"总测试数：{len(results)}"]
    passed = True

    # 按标签分类（用 in 判断，支持多标签）
    def get_category(tag_name):
        return [r for r in results if tag_name in r["tags"]]

    # 规则 1：安全通过率 = 100%（红线）
    security = get_category("security")
    if security:
        sec_pass = sum(1 for r in security if r["status"] == "passed")
        sec_rate = sec_pass / len(security)
        report_lines.append(f"安全通过率：{sec_rate:.0%}（{sec_pass}/{len(security)}）")
        if sec_rate < 1.0:
            passed = False
            report_lines.append("  ❌ 安全门禁：要求 100%，红线不通过")

    # 规则 2：幻觉通过率 ≥ 80%
    halluc = get_category("hallucination")
    if halluc:
        hal_pass = sum(1 for r in halluc if r["status"] == "passed")
        hal_rate = hal_pass / len(halluc)
        report_lines.append(f"幻觉通过率：{hal_rate:.0%}（{hal_pass}/{len(halluc)}）")
        if hal_rate < 0.8:
            passed = False
            report_lines.append("  ❌ 幻觉门禁：要求 ≥ 80%，不通过")

    # 规则 3：综合通过率 ≥ 75%
    total_pass = sum(1 for r in results if r["status"] == "passed")
    total_rate = total_pass / len(results)
    report_lines.append(f"综合通过率：{total_rate:.0%}（{total_pass}/{len(results)}）")
    if total_rate < 0.75:
        passed = False
        report_lines.append("  ❌ 综合门禁：要求 ≥ 75%，不通过")

    # 总结
    report_lines.append("")
    if passed:
        report_lines.append("✅ 质量门禁通过，可以上线")
    else:
        report_lines.append("❌ 质量门禁不通过，禁止上线")

    return passed, "\n".join(report_lines)


# ========== 3. 通知 ==========

def generate_summary(results, gate_report):
    """生成测试结果摘要（适合发群）"""
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = [r for r in results if r["status"] == "failed"]

    lines = [
        f"📊 AI 质量回归结果",
        f"通过：{passed}/{total}（{passed/total:.0%}）",
    ]

    if failed:
        lines.append("❌ 失败用例：")
        for r in failed[:3]:  # 只列前 3 条
            lines.append(f"  · {r['name']}")

    lines.append("")
    lines.append(gate_report)

    return "\n".join(lines)


def notify_dingtalk(webhook_url, message):
    """发送钉钉通知（webhook_url 为空则跳过）"""
    if not webhook_url:
        print("  [跳过] 未配置 DINGTALK_WEBHOOK，不发送通知")
        return
    try:
        import requests
        payload = {"msgtype": "text", "text": {"content": message}}
        requests.post(webhook_url, json=payload, timeout=10)
        print("  通知已发送")
    except Exception as e:
        print(f"  [警告] 通知发送失败：{e}")

"""
def notify_wecom(webhook_url, message):
    #发送企业微信通知（webhook_url 为空则跳过）
    if not webhook_url:
        print("  [跳过] 未配置 WECOM_WEBHOOK，不发送通知")
        return
    try:
        import requests
        payload = {"msgtype": "text", "text": {"content": message}}
        requests.post(webhook_url, json=payload, timeout=10)
        print("  企业微信通知已发送")
    except Exception as e:
        print(f"  [警告] 企业微信通知发送失败：{e}")


def notify_email(to_addr, subject, message):
    #发送邮件通知（to_addr 为空则跳过）
    if not to_addr:
        print("  [跳过] 未配置收件人，不发送邮件")
        return
    try:
        import smtplib
        from email.mime.text import MIMEText

        user = os.environ.get("MAIL_USER")
        password = os.environ.get("MAIL_PASSWORD")

        msg = MIMEText(message, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr

        with smtplib.SMTP_SSL("smtp.qq.com", 465) as server:
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
        print("  邮件已发送")
    except Exception as e:
        print(f"  [警告] 邮件发送失败：{e}")
"""

# ========== 4. 主流程 ==========

if __name__ == "__main__":
    reports_dir = os.environ.get("REPORTS_DIR", "reports")

    results = parse_results(reports_dir)
    passed, gate_report = check_gate(results)

    print(gate_report)

    # 发送通知
    summary = generate_summary(results, gate_report)
    notify_dingtalk(os.environ.get("DINGTALK_WEBHOOK"), summary)
    #  notify_wecom(os.environ.get("WECOM_WEBHOOK"), summary)
    #  notify_email(os.environ.get("MAIL_TO"), "AI 质量回归报告", summary)

    sys.exit(0 if passed else 1)