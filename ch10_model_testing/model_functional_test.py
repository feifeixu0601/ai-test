"""
AI 模型功能测试实战
以中文情感分类模型为被测系统，完整演示测试用例设计、执行、报告输出

流程：搭建模型 → 设计用例 → 执行测试 → 生成报告
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 设置中文字体（Mac 用户改成 "PingFang SC"）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("AI 模型功能测试实战")
print("=" * 60)


# ========== 第一步：搭建被测模型 ==========

def build_model():
    """
    训练一个简单的情感分类模型
    这是我们的"被测系统"（System Under Test）
    """
    # 训练数据
    train_texts = [
        # 正面
        "这个产品非常好用，非常满意",
        "质量很好，推荐购买",
        "服务态度很棒，物流也很快",
        "性价比很高，值得推荐",
        "超出预期，非常惊喜",
        "做工精细，品质优秀",
        "用了一段时间，感觉不错",
        "很喜欢这个产品，会回购",
        # 负面
        "太差了，质量完全不行",
        "非常失望，后悔买了",
        "服务态度很差，不会再买",
        "质量有问题，要求退货",
        "做工粗糙，不值这个价",
        "和描述不符，差评",
        "用了一次就坏了，垃圾",
        "客服不理人，体验极差",
        # 中性
        "一般般，没什么特别的",
        "凑合能用，不算好也不算差",
        "中规中矩，还行吧",
        "价格对得起这个品质",
    ]
    train_labels = (
        ["positive"] * 8 +
        ["negative"] * 8 +
        ["neutral"] * 4
    )

    # 文本向量化（字符级，适合中文）
    vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)

    # 训练朴素贝叶斯分类器
    model = MultinomialNB()
    model.fit(X_train, train_labels)

    print(f"\n模型训练完成")
    print(f"  训练样本：{len(train_texts)} 条")
    print(f"  类别：{model.classes_}")
    print(f"  特征维度：{X_train.shape[1]}")

    return model, vectorizer


# 预测接口（模拟真实模型 API）
def predict(model, vectorizer, text):
    """
    对输入文本做情感预测
    返回：(预测类别, 置信度, 所有类别概率)
    """
    if not text or not text.strip():
        return "error", 0.0, {}

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    prob_dict = dict(zip(model.classes_, probabilities))

    return prediction, confidence, prob_dict


# 测试
model, vectorizer = build_model()
pred, conf, probs = predict(model, vectorizer, "东西很好，推荐")
print(f"\n快速验证：'东西很好，推荐' → {pred} (置信度: {conf:.2%})")


# ========== 第二步：设计测试用例 ==========

# 测试用例的数据结构
# 每个用例包含：ID、类别、输入文本、期望输出、描述

test_cases = []

def add_case(test_id, category, input_text, expected, description):
    """添加一个测试用例"""
    test_cases.append({
        "id": test_id,
        "category": category,
        "input": input_text,
        "expected": expected,
        "description": description
    })


# ---- 维度一：正常功能测试（等价类划分）----

# 正面等价类
add_case("N01", "正常功能", "这个产品非常好用", "positive", "明确正面评价")
add_case("N02", "正常功能", "非常满意这次购物", "positive", "明确正面情感")
add_case("N03", "正常功能", "质量很好推荐购买", "positive", "正面推荐")

# 负面等价类
add_case("N04", "正常功能", "太差了完全不行", "negative", "明确负面评价")
add_case("N05", "正常功能", "非常失望后悔买了", "negative", "明确负面情感")
add_case("N06", "正常功能", "质量有问题要求退货", "negative", "负面投诉")

# 中性等价类
add_case("N07", "正常功能", "一般般没什么特别的", "neutral", "中性评价")
add_case("N08", "正常功能", "凑合能用还行吧", "neutral", "平淡评价")


# ---- 维度二：边界情况测试 ----

# 褒贬模糊的文本
add_case("B01", "边界情况", "还行吧，一般般", "neutral", "模糊中性评价")
add_case("B02", "边界情况", "不算好也不算差", "neutral", "极端中性表达")
add_case("B03", "边界情况", "有点好又有点差", "neutral", "矛盾情感混合")
add_case("B04", "边界情况", "价格对得起这个品质", "neutral", "隐含中性判断")

# 短文本边界
add_case("B05", "边界情况", "好", "positive", "单字正面（极短输入）")
add_case("B06", "边界情况", "差", "negative", "单字负面（极短输入）")
add_case("B07", "边界情况", "还行", "neutral", "双字中性（极短输入）")


# ---- 维度三：异常输入测试 ----

# 空输入
add_case("A01", "异常输入", "", "error", "空字符串")

# 纯空格
add_case("A02", "异常输入", "   ", "error", "纯空格")

# 纯数字
add_case("A03", "异常输入", "12345678", "error", "纯数字输入")

# 纯符号
add_case("A04", "异常输入", "!!!@@@###", "error", "纯符号输入")

# 超长文本（远超训练数据的长度）
add_case("A05", "异常输入", "这个产品" * 200, "positive", "超长文本")

# 非中文文本
add_case("A06", "异常输入", "This product is great", "error", "英文输入")

# 混合乱码
add_case("A07", "异常输入", "你好#@！￥%……&*（", "error", "中文+特殊符号混合")


# ---- 维度四：对抗样本测试 ----

# 否定词欺骗：看起来有正面词，实际是负面
add_case("D01", "对抗样本", "不是很好", "negative", "否定+正面词 → 应判负面")
add_case("D02", "对抗样本", "不好用", "negative", "否定+正面词 → 应判负面")
add_case("D03", "对抗样本", "质量不怎么样", "negative", "委婉否定 → 应判负面")

# 多重否定
add_case("D04", "对抗样本", "不是不满意", "positive", "双重否定 → 应判正面")

# 反讽/讽刺
add_case("D05", "对抗样本", "真是好极了才怪", "negative", "讽刺表达 → 应判负面")
add_case("D06", "对抗样本", "呵呵真是棒棒的呢", "negative", "反讽语气 → 应判负面")

# 混合情感
add_case("D07", "对抗样本", "外观好看但是质量很差", "negative", "先褒后贬 → 关注负面")
add_case("D08", "对抗样本", "质量一般但是服务很好", "positive", "先贬后褒 → 关注正面")


# ---- 维度五：鲁棒性测试 ----

# 同样的语义，不同的表达
add_case("R01", "鲁棒性", "东西不错很好用", "positive", "正面同义表达1")
add_case("R02", "鲁棒性", "产品很棒非常好用", "positive", "正面同义表达2")
add_case("R03", "鲁棒性", "这玩意儿真好使", "positive", "口语化正面表达")

# 加入无意义干扰词
add_case("R04", "鲁棒性", "这个产品嗯非常好用啊", "positive", "正面+语气词干扰")
add_case("R05", "鲁棒性", "质量很好真的", "positive", "正面+语序变化")

# 中英混排
add_case("R06", "鲁棒性", "这个 product 很好用", "positive", "中英混合正面")


# ---- 维度六：置信度合理性测试 ----

# 强烈情感应该有高置信度
add_case("C01", "置信度", "太棒了！完美！超级满意！", "positive", "强烈正面 → 高置信度")
add_case("C02", "置信度", "垃圾！再也不买了！太差了！", "negative", "强烈负面 → 高置信度")

# 模糊情感应该有低置信度
add_case("C03", "置信度", "还行吧", "neutral", "模糊中性 → 低置信度")

print(f"\n测试用例设计完成：共 {len(test_cases)} 个用例")
print(f"\n各维度用例数量：")
from collections import Counter
category_count = Counter(c["category"] for c in test_cases)
for cat, count in category_count.items():
    print(f"  {cat}: {count} 个")


# ========== 第三步：测试执行框架 ==========

class TestResult:
    """单个测试用例的执行结果"""

    def __init__(self, case, actual, confidence, prob_dict):
        self.case = case
        self.actual = actual
        self.confidence = confidence
        self.prob_dict = prob_dict

    @property
    def status(self):
        """
        判定测试结果
        AI 测试的特点：结果是概率性的，不是非黑即白的
        """
        expected = self.case["expected"]

        # 异常输入：模型应该返回 error 或者低置信度
        if expected == "error":
            if self.actual == "error":
                return "PASS"
            elif self.confidence < 0.5:
                return "WARN"    # 模型没报错，但也不确定，可以接受
            else:
                return "FAIL"    # 模型对异常输入给了高置信度的错误答案

        # 置信度测试：除了判断正确，还要检查置信度是否合理
        if self.case["category"] == "置信度":
            if self.actual != expected:
                return "FAIL"
            # 强烈情感应该高置信度
            if self.confidence >= 0.7:
                return "PASS"
            elif self.confidence >= 0.5:
                return "WARN"    # 判断对了但置信度偏低
            else:
                return "FAIL"    # 判断对了但置信度太低，说明模型不确定

        # 常规判定：预测类别是否正确
        if self.actual == expected:
            return "PASS"
        else:
            return "FAIL"

    @property
    def status_icon(self):
        return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[self.status]


def run_tests(model, vectorizer, cases):
    """
    执行所有测试用例，返回结果列表
    """
    results = []

    for case in cases:
        # 调用模型接口
        actual, confidence, prob_dict = predict(model, vectorizer, case["input"])

        # 构造结果对象
        result = TestResult(case, actual, confidence, prob_dict)
        results.append(result)

        # 打印执行过程
        icon = result.status_icon
        input_display = case["input"][:30] + "..." if len(case["input"]) > 30 else case["input"]
        print(f"  {icon} [{case['id']}] {case['category']:<8s} "
              f"输入: {input_display:<30s} "
              f"期望: {case['expected']:<10s} "
              f"实际: {actual:<10s} "
              f"置信度: {confidence:.2%}")

    return results


# ========== 第四步：生成测试报告 ==========

def generate_report(results):
    """
    生成完整的测试报告
    输出到终端和文件
    """
    lines = []

    def add(text=""):
        lines.append(text)

    # ---- 概要 ----
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    warned = sum(1 for r in results if r.status == "WARN")
    failed = sum(1 for r in results if r.status == "FAIL")
    pass_rate = passed / total * 100 if total > 0 else 0

    add("=" * 60)
    add("AI 模型功能测试报告")
    add("=" * 60)
    add(f"")
    add(f"  模型类型：中文情感分类（正面/负面/中性）")
    add(f"  测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"  用例总数：{total}")
    add(f"  通过 (PASS)：{passed}")
    add(f"  警告 (WARN)：{warned}")
    add(f"  失败 (FAIL)：{failed}")
    add(f"  通过率：{pass_rate:.1f}%")

    # 综合评定
    if failed == 0 and warned == 0:
        add(f"  综合评定：✅ 全部通过")
    elif failed == 0:
        add(f"  综合评定：⚠️ 有警告项，建议关注")
    elif failed <= 2:
        add(f"  综合评定：❌ 有失败项，需要修复")
    else:
        add(f"  综合评定：❌ 多项失败，模型质量存在严重问题")

    # ---- 按维度统计 ----
    add(f"\n{'─' * 60}")
    add(f"一、各维度测试结果")
    add(f"{'─' * 60}")

    categories = {}
    for r in results:
        cat = r.case["category"]
        if cat not in categories:
            categories[cat] = {"PASS": 0, "WARN": 0, "FAIL": 0}
        categories[cat][r.status] += 1

    add(f"\n  {'维度':<10s} {'通过':>4s} {'警告':>4s} {'失败':>4s} {'通过率':>8s}")
    add(f"  {'─' * 40}")
    for cat, counts in categories.items():
        cat_total = sum(counts.values())
        cat_pass = counts["PASS"]
        cat_rate = cat_pass / cat_total * 100
        add(f"  {cat:<10s} {counts['PASS']:>4d} {counts['WARN']:>4d} "
            f"{counts['FAIL']:>4d} {cat_rate:>7.1f}%")

    # ---- 详细结果 ----
    add(f"\n{'─' * 60}")
    add(f"二、详细测试结果")
    add(f"{'─' * 60}")

    for r in results:
        icon = r.status_icon
        input_display = r.case["input"][:40] + "..." if len(r.case["input"]) > 40 else r.case["input"]
        add(f"\n  {icon} [{r.case['id']}] {r.case['category']}")
        add(f"     描述：{r.case['description']}")
        add(f"     输入：{input_display}")
        add(f"     期望：{r.case['expected']}  实际：{r.actual}")
        add(f"     置信度：{r.confidence:.2%}")
        if r.prob_dict:
            probs_str = "  ".join(f"{k}={v:.2%}" for k, v in r.prob_dict.items())
            add(f"     概率分布：{probs_str}")
        if r.status != "PASS":
            add(f"     ⚠️ 问题：{get_issue_description(r)}")

    # ---- 失败案例分析 ----
    fail_results = [r for r in results if r.status == "FAIL"]
    warn_results = [r for r in results if r.status == "WARN"]

    if fail_results:
        add(f"\n{'─' * 60}")
        add(f"三、失败案例分析")
        add(f"{'─' * 60}")

        for r in fail_results:
            add(f"\n  ❌ [{r.case['id']}] {r.case['description']}")
            add(f"     期望 {r.case['expected']}，实际 {r.actual}")
            add(f"     问题：{get_issue_description(r)}")

    if warn_results:
        add(f"\n{'─' * 60}")
        add(f"四、警告案例分析")
        add(f"{'─' * 60}")

        for r in warn_results:
            add(f"\n  ⚠️ [{r.case['id']}] {r.case['description']}")
            add(f"     {get_issue_description(r)}")

    # ---- 改进建议 ----
    add(f"\n{'─' * 60}")
    add(f"五、改进建议")
    add(f"{'─' * 60}")

    suggestions = get_suggestions(results)
    for i, suggestion in enumerate(suggestions, 1):
        add(f"  {i}. {suggestion}")

    add(f"\n{'=' * 60}")
    add(f"报告生成完毕")

    # 输出
    report_text = "\n".join(lines)
    print(report_text)

    with open("test_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存到 test_report.txt")

    return report_text


def get_issue_description(result):
    """根据测试结果生成问题描述"""
    expected = result.case["expected"]
    category = result.case["category"]

    if expected == "error" and result.actual != "error":
        return f"模型对异常输入没有报错，反而给出了高置信度的判断（{result.actual}, {result.confidence:.2%}）"
    if result.actual != expected:
        return f"预测类别错误，可能是模型对该类型输入的学习不够充分"
    if category == "置信度" and result.confidence < 0.5:
        return f"预测正确但置信度仅 {result.confidence:.2%}，说明模型对此输入不太确定"
    return "需要人工复查"


def get_suggestions(results):
    """根据测试结果生成改进建议"""
    suggestions = []

    fail_categories = set(r.case["category"] for r in results if r.status == "FAIL")
    warn_categories = set(r.case["category"] for r in results if r.status == "WARN")

    if "正常功能" in fail_categories:
        suggestions.append("基本功能有问题，需要增加训练数据或更换模型架构")

    if "边界情况" in fail_categories or "边界情况" in warn_categories:
        suggestions.append("边界情况表现不稳定，建议增加边界样本的训练数据，或对模糊输入引入'不确定'的处理机制")

    if "异常输入" in fail_categories or "异常输入" in warn_categories:
        suggestions.append("异常输入处理不完善，建议在模型调用前增加输入校验（空值、格式检查）")

    if "对抗样本" in fail_categories or "对抗样本" in warn_categories:
        suggestions.append("对抗样本容易误判，建议增加对抗样本训练（数据增强），或加入规则层处理否定词")

    if "鲁棒性" in fail_categories or "鲁棒性" in warn_categories:
        suggestions.append("鲁棒性不足，微小扰动导致结果变化，建议做数据增强（加噪声、同义词替换）")

    if "置信度" in fail_categories or "置信度" in warn_categories:
        suggestions.append("置信度校准不理想，建议使用 Platt Scaling 或 Isotonic Regression 做概率校准")

    if not suggestions:
        suggestions.append("测试结果良好，建议持续增加测试用例覆盖度，特别是对抗样本和边缘场景")

    suggestions.append("建议建立持续回归测试机制，每次模型更新后自动运行全套测试")

    return suggestions


# ========== 第五步：结果可视化 ==========

def visualize_results(results):
    """把测试结果画成可视化图表"""
    # 按维度统计
    categories = {}
    for r in results:
        cat = r.case["category"]
        if cat not in categories:
            categories[cat] = {"PASS": 0, "WARN": 0, "FAIL": 0}
        categories[cat][r.status] += 1

    cats = list(categories.keys())
    pass_vals = [categories[c]["PASS"] for c in cats]
    warn_vals = [categories[c]["WARN"] for c in cats]
    fail_vals = [categories[c]["FAIL"] for c in cats]

    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, pass_vals, width, label="PASS", color="#27AE60", edgecolor="white")
    bars2 = ax.bar(x, warn_vals, width, label="WARN", color="#F39C12", edgecolor="white")
    bars3 = ax.bar(x + width, fail_vals, width, label="FAIL", color="#E74C3C", edgecolor="white")

    # 标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                       str(int(height)), ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylabel("用例数量", fontsize=12)
    ax.set_title("各维度测试结果", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("charts/test_results_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n测试结果图表已保存到 charts/test_results_by_category.png")


# ========== 主程序 ==========
if __name__ == "__main__":
    print(f"\n--- 开始执行测试 ---")

    # 执行测试
    results = run_tests(model, vectorizer, test_cases)

    # 生成报告
    generate_report(results)

    # 可视化
    visualize_results(results)

    print(f"\n测试完成！")
