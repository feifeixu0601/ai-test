"""
实战：模拟一个简单的 AI 模型评测系统
功能：接收测试结果、计算指标、生成报告、保存到文件
"""

import json
from datetime import datetime


class AIEvaluator:
    """AI模型评测器"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.results = []

    def add_result(self, case_name, predicted, expected, confidence):
        """添加一条测试结果"""
        is_correct = predicted == expected
        self.results.append({
            "case": case_name,
            "predicted": predicted,
            "expected": expected,
            "confidence": confidence,
            "correct": is_correct
        })
        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] {case_name} | 预测: {predicted} | 期望: {expected} | 置信度: {confidence:.2%}")

    def calc_metrics(self):
        """计算评测指标"""
        if not self.results:
            return None

        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct"])
        avg_confidence = sum(r["confidence"] for r in self.results) / total

        # 找出失败的用例
        failed_cases = [r["case"] for r in self.results if not r["correct"]]

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "avg_confidence": avg_confidence,
            "failed_cases": failed_cases
        }

    def generate_report(self):
        """生成评测报告"""
        metrics = self.calc_metrics()
        if not metrics:
            print("没有测试结果，无法生成报告")
            return

        print("\n" + "=" * 50)
        print(f"AI 模型评测报告")
        print(f"模型名称：{self.model_name}")
        print(f"评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        print(f"测试用例总数：{metrics['total']}")
        print(f"通过数量：{metrics['correct']}")
        print(f"失败数量：{metrics['total'] - metrics['correct']}")
        print(f"准确率：{metrics['accuracy']:.2%}")
        print(f"平均置信度：{metrics['avg_confidence']:.2%}")

        if metrics['failed_cases']:
            print(f"\n失败用例：")
            for case in metrics['failed_cases']:
                print(f"  - {case}")

        # 综合判定
        if metrics['accuracy'] >= 0.9:
            print("\n综合评定：优秀 - 模型质量达标")
        elif metrics['accuracy'] >= 0.8:
            print("\n综合评定：良好 - 建议关注失败用例")
        else:
            print("\n综合评定：不合格 - 模型需要优化")

        print("=" * 50)
        return metrics

    def save_report(self, filename="report.json"):
        """保存报告到文件"""
        metrics = self.calc_metrics()
        report = {
            "model_name": self.model_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": metrics,
            "details": self.results
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存到 {filename}")


# ========== 运行 ==========

evaluator = AIEvaluator("TextClassifier-v2")

print("\n执行测试用例：")
evaluator.add_result("正面情感识别", "positive", "positive", 0.95)
evaluator.add_result("负面情感识别", "negative", "negative", 0.88)
evaluator.add_result("中性情感识别", "neutral", "neutral",  0.72)
evaluator.add_result("讽刺情感识别", "positive", "negative", 0.61)
evaluator.add_result("反问情感识别", "negative", "positive", 0.55)

evaluator.generate_report()
evaluator.save_report()
