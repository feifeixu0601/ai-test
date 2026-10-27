"""
模型可解释性方法演示
用特征重要性和词级贡献分析解释模型的预测

方法：
  1. 特征重要性：看模型对每个词的权重（相对值，不是原始 log prob）
  2. 词级贡献分析：看具体预测中每个词的贡献（正值推高、负值拉低）
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("模型可解释性方法演示")
print("=" * 60)

np.random.seed(42)


# ========== 1. 构建模型 ==========

print("\n--- 1. 构建情感分类模型 ---")

train_texts = [
    "这个产品非常好用，非常满意",
    "质量很好，推荐购买",
    "服务态度很棒，物流也很快",
    "性价比很高，值得推荐",
    "超出预期，非常惊喜",
    "做工精细，品质优秀",
    "用了一段时间，感觉不错",
    "很喜欢这个产品，会回购",
    "太差了，质量完全不行",
    "非常失望，后悔买了",
    "服务态度很差，不会再买",
    "质量有问题，要求退货",
    "做工粗糙，不值这个价",
    "和描述不符，差评",
    "用了一次就坏了，垃圾",
    "客服不理人，体验极差",
    "一般般，没什么特别的",
    "凑合能用，不算好也不算差",
    "中规中矩，还行吧",
    "价格对得起这个品质",
]
train_labels = ["positive"] * 8 + ["negative"] * 8 + ["neutral"] * 4

vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)

model = MultinomialNB()
model.fit(X_train, train_labels)

print(f"  训练完成，特征数（字符 n-gram）：{len(vectorizer.vocabulary_)}")


# ========== 2. 特征重要性分析 ==========

print(f"\n--- 2. 特征重要性分析 ---")
print(f"  计算每个特征在各类别下的区分度")
print(f"  区分度 = 该特征在某类别的 log 概率 - 在其他类别的平均 log 概率")
print(f"  正值越大 → 该特征越能代表这个类别")
print()

feature_names = vectorizer.get_feature_names_out()


def get_feature_discriminability(class_idx):
    """
    计算每个特征对指定类别的区分度

    区分度 = log P(feat|该类别) - mean(log P(feat|其他类别))

    正值：该特征在这个类别下比其他类别更常见
    负值：该特征在这个类别下比其他类别更少见
    """
    n_classes = len(model.classes_)
    log_probs = model.feature_log_prob_

    # 该类别下的 log 概率
    log_prob_this = log_probs[class_idx]

    # 其他类别的平均 log 概率
    other_indices = [i for i in range(n_classes) if i != class_idx]
    log_prob_others = np.mean([log_probs[i] for i in other_indices], axis=0)

    return log_prob_this - log_prob_others


for class_idx, class_label in enumerate(model.classes_):
    discrims = get_feature_discriminability(class_idx)

    top_indices = np.argsort(discrims)[-10:][::-1]
    top_features = [(feature_names[i], discrims[i]) for i in top_indices]

    print(f"  [{class_label}] 最有代表性的 10 个字符特征：")
    for rank, (feat, score) in enumerate(top_features, 1):
        bar = "█" * int(score * 20) if score > 0 else ""
        print(f"    {rank:>2d}. '{feat}'  区分度={score:+.3f}  {bar}")
    print()


# ========== 3. 词级贡献分析 ==========

print(f"--- 3. 词级贡献分析 ---")
print(f"  解释单条预测：看每个特征是推高还是拉低预测结果")
print(f"  贡献值 = (该特征在预测类别的权重 - 在其他类别的平均权重) x 出现次数")
print(f"  正值 → 推向预测类别，负值 → 拉向其他类别")
print()


def explain_prediction(text, top_k=10):
    """
    解释一条预测中每个字符特征的贡献

    核心原理：
      相对贡献 = (log P(feat|预测类别) - mean(log P(feat|其他类别))) x 出现次数

      正值 → 这个特征支持预测为当前类别
      负值 → 这个特征反对预测（拉向其他类别）
    """
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    predicted_class_idx = list(model.classes_).index(label)
    n_classes = len(model.classes_)

    feature_indices = X.nonzero()[1]

    contributions = []
    for feat_idx in feature_indices:
        feat_name = feature_names[feat_idx]
        count = X[0, feat_idx]

        # 该特征在预测类别下的 log 概率
        log_prob_pred = model.feature_log_prob_[predicted_class_idx][feat_idx]

        # 在其他类别下的平均 log 概率
        other_indices = [i for i in range(n_classes) if i != predicted_class_idx]
        log_prob_others = np.mean(
            [model.feature_log_prob_[i][feat_idx] for i in other_indices], axis=0
        )

        # 相对贡献：正值推高预测类别，负值拉低
        contribution = (log_prob_pred - log_prob_others) * count

        contributions.append((feat_name, contribution, count))

    # 按贡献绝对值排序
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return label, proba, contributions[:top_k]


# 解释几条示例预测
explanation_cases = [
    "这个产品质量很差",
    "非常满意推荐购买",
    "服务态度很垃圾",
]

for text in explanation_cases:
    label, proba, contributions = explain_prediction(text)

    print(f"  输入：'{text}'")
    print(f"  预测：{label}")
    print(f"  各类别概率：")
    for cls, p in zip(model.classes_, proba):
        bar = "█" * int(p * 30)
        print(f"    {cls:<10s} {p:.4f}  {bar}")

    print(f"  贡献最大的 {len(contributions)} 个特征：")
    for feat, contrib, count in contributions:
        bar = "█" * int(abs(contrib) * 15)
        if contrib > 0:
            direction = "→推高'" + label + "'"
        else:
            direction = "→拉向其他类别"
        print(f"    '{feat}'  贡献={contrib:+.4f} (出现{count}次)  {bar} {direction}")
    print()


# ========== 4. 特征贡献可视化 ==========

print(f"--- 4. 特征贡献可视化 ---")

test_text = "这个产品质量很差"
label, proba, contributions = explain_prediction(test_text, top_k=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 左图：各类别概率
bars1 = ax1.bar(model.classes_, proba,
                color=["#2ECC71", "#E74C3C", "#95A5A6"],
                edgecolor="gray", alpha=0.8, width=0.5)
for bar, p in zip(bars1, proba):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{p:.2%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax1.set_title(f"预测结果：{label}\n输入：'{test_text}'", fontsize=12, fontweight="bold")
ax1.set_ylabel("概率", fontsize=11)
ax1.set_ylim(0, 1.15)
ax1.grid(True, alpha=0.3, axis="y")

# 右图：特征贡献
feat_names = [c[0] for c in contributions]
feat_scores = [c[1] for c in contributions]
colors_bar = ["#E74C3C" if s > 0 else "#2ECC71" for s in feat_scores]

bars2 = ax2.barh(feat_names, feat_scores, color=colors_bar,
                 edgecolor="gray", alpha=0.8)
ax2.set_xlabel(f"贡献值（正 = 推高'{label}'，负 = 拉向其他类别）", fontsize=10)
ax2.set_title(f"特征贡献分析\n（对 '{label}' 预测的贡献）", fontsize=12, fontweight="bold")
ax2.axvline(x=0, color="black", linewidth=1.5, linestyle="--")
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("charts/model_explain.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  可解释性分析图已保存到 charts/model_explain.png")
