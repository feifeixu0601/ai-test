"""
鲁棒性评估框架
系统化地测试 AI 模型的对抗鲁棒性

功能：
  1. 自动对测试集执行多种对抗攻击
  2. 计算鲁棒性指标（翻转率、置信度变化）
  3. 生成鲁棒性评估报告
"""

import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("=" * 60)
print("AI 模型鲁棒性评估框架")
print("=" * 60)

random.seed(42)
np.random.seed(42)


# ========== 1. 模型构建 ==========

def build_sentiment_model():
    """构建情感分类模型"""
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
        # 新增：覆盖攻击后会出现的负面短语（让模型认识攻击后的文本）
        "这个产品非常难用，非常失望",
        "质量很差，不推荐购买",
        "服务态度很差，物流也很慢",
        "低于预期，非常失望",
        "性价比很低，完全不值",
        "做工粗糙，质量很差",
        # 新增：覆盖攻击后会出现的正面短语
        "太好了，完全没问题",
        "非常满意，庆幸买了",
        "服务态度很好，强烈推荐",
        "质量没问题，非常满意",
        "做工精细，很值这个价",
        "和描述一致，好评",
        # 中性被攻击后的覆盖
        "很不错，很有亮点",
        "非常好用，超出预期",
        "价格很值这个品质",
        "质量很好，值得购买",
    ]
    train_labels = (
        ["positive"] * 8 +
        ["negative"] * 8 +
        ["neutral"] * 4 +
        ["negative"] * 6 +
        ["positive"] * 6 +
        ["positive"] * 4
    )

    vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)

    model = MultinomialNB()
    model.fit(X_train, train_labels)

    return model, vectorizer


model, vectorizer = build_sentiment_model()
print(f"\n模型构建完成")


# ========== 2. 攻击方法定义 ==========

class TextAttacks:
    """文本对抗攻击方法集合"""

    synonyms = {
        "好": "不错", "很好": "挺好", "差": "糟糕",
        "满意": "称心", "推荐": "安利", "喜欢": "中意",
        "质量": "品质", "服务": "客服", "快": "迅速",
        "惊喜": "意外", "失望": "沮丧", "垃圾": "废物",
    }

    noise_words = ["嗯", "啊", "呃", "那个", "就是", "其实"]

    @staticmethod
    def typo(text):
        """字符级扰动：交换相邻字符"""
        chars = list(text)
        swappable = [i for i in range(len(chars) - 1)
                     if chars[i].isalpha() and chars[i + 1].isalpha()]
        if not swappable:
            return text
        idx = random.choice(swappable)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    @classmethod
    def synonym(cls, text):
        """同义词替换"""
        for word, syn in cls.synonyms.items():
            if word in text:
                return text.replace(word, syn, 1)
        return text

    @classmethod
    def noise_insert(cls, text):
        """插入噪声词"""
        if len(text) <= 1:
            return text
        noise = random.choice(cls.noise_words)
        pos = random.randint(1, len(text) - 1)
        return text[:pos] + noise + text[pos:]

    @classmethod
    def antonym(cls, text):
        """反义词替换：用反义词替换关键词，彻底改变语义
        必须先匹配长词（如"非常好用"），否则会被短词（如"好"）截断"""
        antonyms = [
            ("非常好用", "非常难用"),
            ("质量很好", "质量很差"),
            ("超出预期", "低于预期"),
            ("非常惊喜", "非常失望"),
            ("服务态度很棒", "服务态度很差"),
            ("性价比很高", "性价比很低"),
            ("做工精细", "做工粗糙"),
            ("太差了", "太好了"),
            ("完全不行", "完全没问题"),
            ("非常失望", "非常满意"),
            ("后悔买了", "庆幸买了"),
            ("服务态度很差", "服务态度很好"),
            ("质量有问题", "质量没问题"),
            ("要求退货", "继续保留"),
            ("做工粗糙", "做工精细"),
            ("不值这个价", "很值这个价"),
            ("和描述不符", "和描述一致"),
            ("差评", "好评"),
            ("一般般", "很不错"),
            ("没什么特别的", "很有亮点"),
            ("中规中矩", "超出预期"),
            ("还行吧", "很棒"),
            ("价格对得起这个品质", "价格很值这个品质"),
            ("凑合能用", "非常好用"),
            ("不算好也不算差", "非常好"),
            ("非常好", "非常差"),
            ("很好", "很差"),
            ("不错", "很差"),
            ("满意", "失望"),
            ("惊喜", "失望"),
            ("精细", "粗糙"),
            ("粗糙", "精细"),
            ("差", "好"),
            ("失望", "满意"),
            ("垃圾", "精品"),
            ("好", "差"),
        ]

        # 【核心】按 key 的长度降序排序！确保先匹配长词再匹配短词
        sorted_antonyms = sorted(antonyms, key=lambda x: len(x[0]), reverse=True)

        for original, replacement in sorted_antonyms:
            if original in text:
                return text.replace(original, replacement, 1)
        return text

    @staticmethod
    def delete_word(text):
        """删除一个词（模拟信息丢失）"""
        chars = list(text)
        if len(chars) <= 2:
            return text
        idx = random.randint(1, len(chars) - 2)
        return "".join(chars[:idx] + chars[idx + 1:])

    @staticmethod
    def repeat_word(text):
        """重复一个词（模拟口吃/强调）"""
        if len(text) < 4:
            return text
        mid = len(text) // 2
        return text[:mid] + text[mid:mid + 2] + text[mid:]

    @classmethod
    def get_all_methods(cls):
        """返回所有攻击方法"""
        return {
            "字符扰动": cls.typo,
            "同义词替换": cls.synonym,
            "噪声插入": cls.noise_insert,
            "反义词替换": cls.antonym,
            "删除字符": cls.delete_word,
            "重复词语": cls.repeat_word,
        }


# ========== 3. 鲁棒性评估器 ==========

class RobustnessEvaluator:
    """鲁棒性评估器"""

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.attacks = TextAttacks.get_all_methods()

    def predict(self, text):
        """预测文本情感"""
        X = self.vectorizer.transform([text])
        label = self.model.predict(X)[0]
        confidence = float(max(self.model.predict_proba(X)[0]))
        return label, confidence

    def evaluate(self, test_cases):
        """
        对测试集执行鲁棒性评估

        参数：
            test_cases: [(text, expected_label), ...]

        返回：
            评估结果字典
        """
        records = []

        for text, expected_label in test_cases:
            orig_label, orig_conf = self.predict(text)

            for attack_name, attack_fn in self.attacks.items():
                attacked_text = attack_fn(text)
                if attacked_text == text:
                    continue  # 攻击无效（文本没变）

                adv_label, adv_conf = self.predict(attacked_text)
                is_flipped = (adv_label != orig_label)
                conf_change = adv_conf - orig_conf

                records.append({
                    "original_text": text,
                    "attacked_text": attacked_text,
                    "attack_method": attack_name,
                    "original_label": orig_label,
                    "original_confidence": orig_conf,
                    "attacked_label": adv_label,
                    "attacked_confidence": adv_conf,
                    "is_flipped": is_flipped,
                    "confidence_change": conf_change,
                })

        return self._calc_metrics(records)

    def _calc_metrics(self, records):
        """计算鲁棒性指标"""
        if not records:
            return {"overall": None, "by_attack": {}, "by_sentiment": {}}

        total = len(records)
        flipped = sum(1 for r in records if r["is_flipped"])

        # 整体指标
        overall = {
            "total_attacks": total,
            "flipped": flipped,
            "robust_score": round((total - flipped) / total * 100, 1),
            "avg_conf_change": round(np.mean([r["confidence_change"] for r in records]), 4),
        }

        # 按攻击方法
        by_attack = {}
        for method_name in self.attacks:
            method_recs = [r for r in records if r["attack_method"] == method_name]
            if not method_recs:
                continue
            m_total = len(method_recs)
            m_flipped = sum(1 for r in method_recs if r["is_flipped"])
            by_attack[method_name] = {
                "total": m_total,
                "flipped": m_flipped,
                "flip_rate": round(m_flipped / m_total * 100, 1),
                "robust_score": round((m_total - m_flipped) / m_total * 100, 1),
            }

        # 按情感类别
        by_sentiment = {}
        for label in ["positive", "negative", "neutral"]:
            label_recs = [r for r in records if r["original_label"] == label]
            if not label_recs:
                continue
            l_total = len(label_recs)
            l_flipped = sum(1 for r in label_recs if r["is_flipped"])
            by_sentiment[label] = {
                "total": l_total,
                "flipped": l_flipped,
                "flip_rate": round(l_flipped / l_total * 100, 1),
                "robust_score": round((l_total - l_flipped) / l_total * 100, 1),
            }

        return {
            "overall": overall,
            "by_attack": by_attack,
            "by_sentiment": by_sentiment,
            "records": records,
        }


# ========== 4. 生成报告 ==========

def print_report(metrics):
    """打印鲁棒性评估报告"""
    overall = metrics["overall"]
    if not overall:
        print("  无有效评估数据")
        return

    print(f"\n{'─' * 60}")
    print(f"  总体鲁棒性")
    print(f"{'─' * 60}")
    print(f"  总攻击次数：{overall['total_attacks']}")
    print(f"  预测翻转：{overall['flipped']}")
    print(f"  鲁棒性得分：{overall['robust_score']}%")
    print(f"  平均置信度变化：{overall['avg_conf_change']:+.4f}")

    # 按攻击方法
    print(f"\n{'─' * 60}")
    print(f"  按攻击方法的鲁棒性")
    print(f"{'─' * 60}")
    print(f"  {'攻击方法':<12s} {'攻击次数':>8s} {'翻转次数':>8s} {'翻转率':>8s} {'鲁棒性':>8s}")
    print(f"  {'─' * 50}")

    for method, data in metrics["by_attack"].items():
        print(f"  {method:<12s} {data['total']:>8d} {data['flipped']:>8d} "
              f"{data['flip_rate']:>7.1f}% {data['robust_score']:>7.1f}%")

    # 按情感类别
    print(f"\n{'─' * 60}")
    print(f"  按情感类别的鲁棒性")
    print(f"{'─' * 60}")
    print(f"  {'情感类别':<12s} {'攻击次数':>8s} {'翻转次数':>8s} {'翻转率':>8s} {'鲁棒性':>8s}")
    print(f"  {'─' * 50}")

    for label, data in metrics["by_sentiment"].items():
        print(f"  {label:<12s} {data['total']:>8d} {data['flipped']:>8d} "
              f"{data['flip_rate']:>7.1f}% {data['robust_score']:>7.1f}%")

    # 综合评估
    print(f"\n{'─' * 60}")
    print(f"  综合评估")
    print(f"{'─' * 60}")

    score = overall["robust_score"]
    if score >= 80:
        print(f"  [PASS] 鲁棒性得分 {score}%（>= 80%），模型较鲁棒")
    elif score >= 60:
        print(f"  [WARN] 鲁棒性得分 {score}%（60%~80%），存在脆弱点，建议优化")
    else:
        print(f"  [FAIL] 鲁棒性得分 {score}%（< 60%），鲁棒性不足，需要重点改进")

    # 找出最脆弱的攻击方法
    if metrics["by_attack"]:
        worst_attack = max(metrics["by_attack"].items(),
                           key=lambda x: x[1]["flip_rate"])
        print(f"  [WARN] 最脆弱的攻击：{worst_attack[0]}"
              f"（翻转率 {worst_attack[1]['flip_rate']:.1f}%）")


# ========== 5. 主程序 ==========

if __name__ == "__main__":
    # 测试集
    test_cases = [
        ("这个产品非常好用", "positive"),
        ("质量很好，推荐购买", "positive"),
        ("服务态度很棒，物流也很快", "positive"),
        ("超出预期，非常惊喜", "positive"),
        ("性价比很高，值得推荐", "positive"),
        ("做工精细，品质优秀", "positive"),
        ("太差了，质量完全不行", "negative"),
        ("非常失望，后悔买了", "negative"),
        ("服务态度很差，不会再买", "negative"),
        ("质量有问题，要求退货", "negative"),
        ("做工粗糙，不值这个价", "negative"),
        ("和描述不符，差评", "negative"),
        ("一般般，没什么特别的", "neutral"),
        ("凑合能用，不算好也不算差", "neutral"),
        ("中规中矩，还行吧", "neutral"),
        ("价格对得起这个品质", "neutral"),
    ]

    print(f"\n测试样本数：{len(test_cases)}")
    print(f"攻击方法数：{len(TextAttacks.get_all_methods())}")

    # 执行评估
    evaluator = RobustnessEvaluator(model, vectorizer)
    print(f"\n开始鲁棒性评估...")
    metrics = evaluator.evaluate(test_cases)

    # 输出报告
    print_report(metrics)