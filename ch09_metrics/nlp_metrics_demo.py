"""
NLP 专属指标：BLEU 和 ROUGE
了解文本生成质量的评估方法
"""

print("=" * 60)
print("NLP 专属指标演示")
print("=" * 60)

# ========== BLEU 分数 ==========
print("\n--- BLEU 分数（机器翻译质量）---")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 参考翻译（人工写的）
reference = [["the", "cat", "is", "on", "the", "mat"]]

# 模型翻译（自动生成的）
candidates = {
    "完美翻译": ["the", "cat", "is", "on", "the", "mat"],
    "接近翻译": ["the", "cat", "sits", "on", "the", "mat"],
    "较差翻译": ["a", "dog", "is", "under", "the", "table"],
}

smoothie = SmoothingFunction().method1

for name, candidate in candidates.items():
    score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    print(f"  {name}: BLEU = {score:.4f}")
    print(f"    翻译内容: {' '.join(candidate)}")

# ========== ROUGE 分数 ==========
print("\n--- ROUGE 分数（文本摘要质量）---")

try:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    reference_text = "AI测试工程师需要掌握Python编程和机器学习基础"
    generated_texts = {
        "高质量摘要": "AI测试需要掌握Python和机器学习",
        "中等摘要": "AI测试工程师要会编程",
        "低质量摘要": "今天天气不错，适合出去玩",
    }

    for name, generated in generated_texts.items():
        scores = scorer.score(reference_text, generated)
        print(f"\n  {name}:")
        print(f"    生成内容: {generated}")
        print(f"    ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
        print(f"    ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
        print(f"    ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

except ImportError:
    print("  需要安装 rouge-score：pip install rouge-score")
    print("  安装后重新运行即可")

# ========== 大模型评测指标概览 ==========
print("\n--- 大模型评测指标概览（了解即可）---")
print("""
  知识问答类：
    MMLU / C-Eval 准确率    → 知识掌握程度
    F1 Score                → 问答匹配精度

  代码生成类：
    Pass@k                  → 生成 k 次至少对一次的比例
    HumanEval 通过率         → 代码正确性

  文本生成类：
    BLEU / ROUGE            → 生成文本和参考文本的相似度
    幻觉率 (Hallucination)   → 生成虚假信息的比例（越低越好）
    事实一致性 (Faithfulness) → 生成内容是否忠于输入/上下文

  对话类：
    人类偏好胜率 (Win Rate)  → 人类评审认为 A 比 B 好的比例
    多轮一致性               → 多轮对话中是否前后矛盾

  安全类：
    有害内容生成率            → 生成有害内容的概率（越低越好）
    拒绝率                   → 对有害请求的拒绝比例（越高越好）
""")