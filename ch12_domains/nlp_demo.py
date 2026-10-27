"""
NLP 核心概念演示
分词、词嵌入的直觉理解
"""

import numpy as np

print("=" * 60)
print("NLP 核心概念演示")
print("=" * 60)


# ========== 1. 分词演示 ==========

print("\n--- 1. 分词（Tokenization） ---")

import jieba

texts = [
    "我喜欢AI测试工程师这个职业",
    "大语言模型正在改变软件测试的方式",
    "Python是AI测试的第一技能",
]

for text in texts:
    tokens = list(jieba.cut(text))
    print(f"\n  原文: {text}")
    print(f"  分词: {tokens}")
    print(f"  Token数: {len(tokens)}")

print(f"\n  分词的意义：机器只能逐个 Token 处理，Token 的划分方式会影响模型理解")
print(f"  比如'AI测试'如果被切成['AI', '测试']，模型理解为两个概念")
print(f"  如果被切成['AI测', '试']，模型理解就完全跑偏了")


# ========== 2. 词嵌入演示 ==========

print(f"\n--- 2. 词嵌入（Word Embedding） ---")
print(f"核心思想：含义相近的词，向量空间中距离也相近")

# 用简化的向量来演示（真实模型的向量通常有几百到几千维）
words = {
    "猫":   np.array([0.9, 0.1, 0.8]),
    "狗":   np.array([0.85, 0.15, 0.75]),
    "汽车": np.array([0.1, 0.9, 0.2]),
    "卡车": np.array([0.15, 0.85, 0.25]),
    "小明": np.array([0.5, 0.3, 0.6]),
}

def cosine_similarity(a, b):
    """余弦相似度：衡量两个向量的相似程度（-1到1，越接近1越相似）"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\n  词向量（简化为3维）：")
for word, vec in words.items():
    print(f"    {word}: {vec}")

print(f"\n  余弦相似度：")
pairs = [("猫", "狗"), ("猫", "汽车"), ("汽车", "卡车"), ("猫", "小明"), ("狗", "卡车")]
for w1, w2 in pairs:
    sim = cosine_similarity(words[w1], words[w2])
    bar = "█" * int(sim * 20)
    print(f"    {w1} ↔ {w2}: {sim:.3f}  {bar}")

print(f"\n  观察：猫↔狗 和 汽车↔卡车 的相似度都很高（同类事物）")
print(f"        猫↔汽车 的相似度很低（不同类事物）")


# ========== 3. 注意力机制的直觉理解 ==========

print(f"\n--- 3. 注意力机制（Attention）的直觉 ---")

# 简化演示：当模型处理"它"这个词时，应该关注句中的哪个词
sentence = ["小明", "把", "苹果", "放在", "桌子", "上", "然后", "它", "被", "吃掉"]
target_word = "它"

# 模拟注意力权重（实际中由模型学习得到）
attention_weights = {
    "小明": 0.05,
    "把": 0.01,
    "苹果": 0.65,    # 注意力高度集中在"苹果"上
    "放在": 0.03,
    "桌子": 0.12,
    "上": 0.01,
    "然后": 0.02,
    "它": 0.02,
    "被": 0.01,
    "吃掉": 0.08,
}

print(f"\n  句子：{' '.join(sentence)}")
print(f"  当模型处理 '{target_word}' 时，注意力分布：")
for word, weight in sorted(attention_weights.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(weight * 50)
    print(f"    {word:<6s} {weight:.2f}  {bar}")

print(f"\n  模型把大部分注意力放在'苹果'上——正确！'它'指的就是苹果")
print(f"  这就是注意力机制的作用：自动学会'该看哪里'")
