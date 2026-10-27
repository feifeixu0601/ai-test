"""
NLP 文本对抗攻击实战
对情感分类模型进行文本扰动攻击，测试鲁棒性

测试内容：
  1. 字符级扰动（错别字）
  2. 同义词替换
  3. 插入噪声词
  4. 反义词替换
"""

import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("=" * 60)
print("NLP 文本对抗攻击实战")
print("=" * 60)

random.seed(42)
np.random.seed(42)


# ========== 1. 构建情感分类模型 ==========

print("\n--- 1. 构建情感分类模型 ---")

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
    # 新增：覆盖攻击后会出现的负面短语
    "这个产品非常难用，非常失望",
    "质量很差，不推荐购买",
    "低于预期，非常失望",
    "太好了，完全没问题",
    # 新增：覆盖攻击后会出现的正面短语
    "非常满意，庆幸买了",
    "质量没问题，继续保留",
    "很不错，很有亮点",
    "很棒，超出预期",
]
train_labels = (
    ["positive"] * 8 +
    ["negative"] * 8 +
    ["neutral"] * 4 +
    ["negative"] * 4 +
    ["positive"] * 4
)

vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)

model = MultinomialNB()
model.fit(X_train, train_labels)

print(f"  训练样本数：{len(train_texts)}")
print(f"  模型类型：朴素贝叶斯 + 字符 n-gram")


def predict_sentiment(text):
    """预测文本情感，返回 (标签, 置信度)"""
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(max(proba))
    return label, confidence


# 验证模型
test_text = "这个产品很好"
label, conf = predict_sentiment(test_text)
print(f"\n  验证：'{test_text}' → {label}（置信度 {conf:.2%}）")


# ========== 2. 定义四种攻击方法 ==========

print(f"\n--- 2. 定义四种攻击方法 ---")


# 攻击1：字符级扰动（模拟打字错误）
def attack_typo(text):
    """
    随机交换文本中两个相邻字符的位置
    模拟用户打字时的手误
    """
    chars = list(text)
    # 找到可以交换的位置（排除标点）
    swappable = [i for i in range(len(chars) - 1)
                 if chars[i].isalpha() and chars[i + 1].isalpha()]

    if not swappable:
        return text, None

    idx = random.choice(swappable)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    attacked = "".join(chars)

    return attacked, f"交换位置 {idx} 的'{chars[idx+1]}{chars[idx]}'→'{chars[idx]}{chars[idx+1]}'"


# 攻击2：同义词替换
def attack_synonym(text):
    """
    用同义词替换文本中的关键词
    """
    synonyms = {
        "好": "不错", "很好": "挺好", "差": "糟糕",
        "满意": "称心", "推荐": "安利", "喜欢": "中意",
        "质量": "品质", "服务": "客服", "快": "迅速",
        "惊喜": "意外", "失望": "沮丧", "垃圾": "废物",
    }

    for original, replacement in synonyms.items():
        if original in text:
            attacked = text.replace(original, replacement, 1)
            return attacked, f"'{original}' → '{replacement}'"

    return text, None  # 没有可替换的词


# 攻击3：插入噪声词
def attack_insert_noise(text):
    """
    在文本中插入不影响语义的语气词
    """
    noise_words = ["嗯", "啊", "呃", "那个", "就是", "其实", "反正"]
    noise = random.choice(noise_words)

    # 在随机位置插入
    pos = random.randint(1, len(text) - 1) if len(text) > 1 else 1
    attacked = text[:pos] + noise + text[pos:]

    return attacked, f"在位置 {pos} 插入'{noise}'"


# 攻击4：反义词替换（强制优先匹配长词）
def attack_antonym(text):
    """
    用反义词替换关键词，彻底改变语义
    必须先匹配长词（如"很好"），否则会被短词（如"好"）截断
    """
    antonyms = [
        ("非常好用", "非常难用"),
        ("质量很好", "质量很差"),
        ("超出预期", "低于预期"),
        ("非常惊喜", "非常失望"),
        ("太差了", "太好了"),
        ("完全不行", "完全没问题"),
        ("非常失望", "非常满意"),
        ("后悔买了", "庆幸买了"),
        ("质量有问题", "质量没问题"),
        ("要求退货", "继续保留"),
        ("一般般", "很不错"),
        ("没什么特别的", "很有亮点"),
        ("中规中矩", "超出预期"),
        ("还行吧", "很棒"),
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

    # 【核心修改】按 key 的长度降序排序！确保先匹配"非常好用"再匹配"好"
    sorted_antonyms = sorted(antonyms, key=lambda x: len(x[0]), reverse=True)

    for original, replacement in sorted_antonyms:
        if original in text:
            attacked = text.replace(original, replacement, 1)
            return attacked, f"'{original}' → '{replacement}'"
    return text, None


# 攻击方法注册
attack_methods = {
    "字符扰动": attack_typo,
    "同义词替换": attack_synonym,
    "插入噪声": attack_insert_noise,
    "反义词替换": attack_antonym,
}

print(f"  已注册 {len(attack_methods)} 种攻击方法：{list(attack_methods.keys())}")


# ========== 3. 执行攻击测试 ==========

print(f"\n--- 3. 执行攻击测试 ---")

# 测试样本（覆盖正面、负面、中性）
test_cases = [
    ("这个产品非常好用", "positive"),
    ("质量很好，推荐购买", "positive"),
    ("超出预期，非常惊喜", "positive"),
    ("太差了，完全不行", "negative"),
    ("非常失望，后悔买了", "negative"),
    ("质量有问题，要求退货", "negative"),
    ("一般般，没什么特别的", "neutral"),
    ("中规中矩，还行吧", "neutral"),
]

all_attack_records = []

print(f"\n  测试样本数：{len(test_cases)}")
print(f"  攻击方法数：{len(attack_methods)}")
print(f"  总攻击次数：{len(test_cases) * len(attack_methods)}")
print()

for text, expected_label in test_cases:
    orig_label, orig_conf = predict_sentiment(text)

    print(f"  原文：'{text}'")
    print(f"    原始预测：{orig_label}（置信度 {orig_conf:.2%}）")

    for method_name, attack_fn in attack_methods.items():
        attacked_text, change_desc = attack_fn(text)

        if change_desc is None:
            continue  # 这种攻击不适用

        adv_label, adv_conf = predict_sentiment(attacked_text)
        is_flipped = (adv_label != orig_label)

        record = {
            "original_text": text,
            "attacked_text": attacked_text,
            "attack_method": method_name,
            "change": change_desc,
            "original_label": orig_label,
            "original_confidence": orig_conf,
            "attacked_label": adv_label,
            "attacked_confidence": adv_conf,
            "is_flipped": is_flipped,
        }
        all_attack_records.append(record)

        flip_marker = "[翻转]" if is_flipped else "[稳定]"
        print(f"    [{method_name}] {change_desc}")
        print(f"      攻击后：'{attacked_text}'")
        print(f"      预测：{adv_label}（{adv_conf:.2%}）{flip_marker}")

    print()


# ========== 4. 鲁棒性统计 ==========

print(f"\n{'=' * 60}")
print(f"鲁棒性统计")
print(f"{'=' * 60}")

total_attacks = len(all_attack_records)
flipped = sum(1 for r in all_attack_records if r["is_flipped"])
robust = total_attacks - flipped

print(f"\n  总攻击次数：{total_attacks}")
print(f"  预测翻转：{flipped}")
print(f"  预测稳定：{robust}")
print(f"  鲁棒性得分：{robust / total_attacks:.1%}"
      f"（越高越鲁棒）")

# 按攻击方法统计
print(f"\n  按攻击方法统计：")
print(f"  {'攻击方法':<12s} {'攻击次数':>8s} {'翻转次数':>8s} {'翻转率':>8s} {'鲁棒性':>8s}")
print(f"  {'─' * 50}")

for method_name in attack_methods:
    method_records = [r for r in all_attack_records if r["attack_method"] == method_name]
    if not method_records:
        continue

    method_total = len(method_records)
    method_flipped = sum(1 for r in method_records if r["is_flipped"])
    method_flip_rate = method_flipped / method_total if method_total > 0 else 0

    print(f"  {method_name:<12s} {method_total:>8d} {method_flipped:>8d} "
          f"{method_flip_rate:>7.1%} {1 - method_flip_rate:>7.1%}")

# 按原始情感类别统计
print(f"\n  按原始情感类别统计：")
print(f"  {'情感类别':<12s} {'攻击次数':>8s} {'翻转次数':>8s} {'翻转率':>8s} {'鲁棒性':>8s}")
print(f"  {'─' * 50}")

for label in ["positive", "negative", "neutral"]:
    label_records = [r for r in all_attack_records if r["original_label"] == label]
    if not label_records:
        continue

    label_total = len(label_records)
    label_flipped = sum(1 for r in label_records if r["is_flipped"])
    label_flip_rate = label_flipped / label_total if label_total > 0 else 0

    print(f"  {label:<12s} {label_total:>8d} {label_flipped:>8d} "
          f"{label_flip_rate:>7.1%} {1 - label_flip_rate:>7.1%}")

# 最容易被攻击的用例
print(f"\n  最容易被攻击的用例（翻转次数最多）：")

flip_counts = {}
for r in all_attack_records:
    if r["is_flipped"]:
        key = r["original_text"]
        flip_counts[key] = flip_counts.get(key, 0) + 1

for text, count in sorted(flip_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"    '{text}' —— 被 {count} 种方法攻破")