"""
偏见模型构建与公平性检测
实战演示如何发现 AI 模型中的群体偏见

步骤：
  1. 构建一个故意带性别偏见的情感分类模型
  2. 把性别作为模型特征（让模型能"看到"性别）
  3. 用三大公平性指标检测偏见
  4. 可视化偏见程度
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("偏见模型构建与公平性检测")
print("=" * 60)

np.random.seed(42)


# ========== 1. 构建带偏见的训练数据 ==========

print("\n--- 1. 构建带偏见的训练数据 ---")

positive_texts = [
    "这个产品非常好用，非常满意",
    "质量很好，推荐购买",
    "服务态度很棒，物流也很快",
    "性价比很高，值得推荐",
    "超出预期，非常惊喜",
    "做工精细，品质优秀",
    "用了一段时间，感觉不错",
    "很喜欢这个产品，会回购",
]

negative_texts = [
    "太差了，质量完全不行",
    "非常失望，后悔买了",
    "服务态度很差，不会再买",
    "质量有问题，要求退货",
    "做工粗糙，不值这个价",
    "和描述不符，差评",
    "用了一次就坏了，垃圾",
    "客服不理人，体验极差",
]

train_records = []

# 女性：190 正面 + 10 负面（正面率 95% → 强偏见来源）
for i in range(190):
    train_records.append({
        "text": positive_texts[i % len(positive_texts)],
        "gender": "female",
        "label": "positive"
    })
for i in range(10):
    train_records.append({
        "text": negative_texts[i % len(negative_texts)],
        "gender": "female",
        "label": "negative"
    })

# 男性：100 正面 + 100 负面（正面率 50% → 正常比例）
for i in range(100):
    train_records.append({
        "text": positive_texts[i % len(positive_texts)],
        "gender": "male",
        "label": "positive"
    })
for i in range(100):
    train_records.append({
        "text": negative_texts[i % len(negative_texts)],
        "gender": "male",
        "label": "negative"
    })

train_df = pd.DataFrame(train_records)

print(f"  训练数据总量：{len(train_df)} 条")
print(f"\n  训练数据的情感分布（按性别）：")
print(f"  {'性别':<10s} {'正面':>8s} {'负面':>8s} {'正面率':>8s}")
print(f"  {'─' * 38}")

for gender in ["female", "male"]:
    g = train_df[train_df["gender"] == gender]
    pos = (g["label"] == "positive").sum()
    neg = (g["label"] == "negative").sum()
    print(f"  {gender:<10s} {pos:>8d} {neg:>8d} {pos/(pos+neg)*100:>7.1f}%")

print(f"\n  [注意] 女性正面率 95% 远高于男性 50%——这就是偏见的来源！")


# ========== 2. 训练模型（性别作为特征） ==========

print(f"\n--- 2. 训练模型（性别作为特征） ---")

# 文本特征
vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(1, 2))
X_text = vectorizer.fit_transform(train_df["text"])

# 性别特征（female=1, male=0）
X_gender = csr_matrix(
    (train_df["gender"] == "female").astype(int).values.reshape(-1, 1)
)

# 合并特征：文本 + 性别
X_combined = hstack([X_text, X_gender])
y_train = train_df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X_combined, y_train)

# 查看性别特征的权重
gender_weight = model.coef_[0][-1]  # 最后一个特征是性别
print(f"  模型训练完成（逻辑回归）")
print(f"  文本特征数：{X_text.shape[1]}")
print(f"  性别特征权重：{gender_weight:+.4f}")
if gender_weight > 0:
    print(f"  [偏见] 正权重 → 模型学到'女性 = 更可能正面'的错误关联")
else:
    print(f"  性别权重为负，偏见方向为男性正面")


def predict_sentiment(text, gender):
    """预测文本情感（性别作为模型输入）"""
    X_text = vectorizer.transform([text])
    X_gender = csr_matrix([[1 if gender == "female" else 0]])
    X_combined = hstack([X_text, X_gender])

    label = model.predict(X_combined)[0]
    proba = model.predict_proba(X_combined)[0]
    confidence = float(max(proba))
    return label, confidence


# ========== 3. 构建平衡测试集 ==========

print(f"\n--- 3. 公平性检测 ---")

# 测试文本池（与训练文本不同，考察泛化能力）
# 正面 50 条
test_texts_positive = [
    "这款商品真的很棒", "物超所值，非常满意", "物流很快，包装精美",
    "质量过硬，值得拥有", "用着很顺手，好评", "外观漂亮，功能强大",
    "客服耐心，解决问题快", "第二次购买，一如既往的好", "朋友推荐来的，果然不错",
    "超出预期的好，值得推荐", "性价比之王，闭眼入", "用了半个月，越用越喜欢",
    "家人也很满意，好评", "细节做得很好，点赞", "发货速度快，服务好",
    "质感超好，超出预期", "孩子很喜欢，值得购买", "回购第三次了，稳定输出",
    "包装严实，东西完美", "试了一次就爱上了", "声音清晰，续航给力",
    "操作简单，老人也会用", "颜值高，性能强", "价格实惠，品质不打折",
    "手感舒适，做工一流", "设计感很强，很有档次", "实用性强，好评",
    "对比了好几家，这家最好", "送礼首选，很有面子", "食材新鲜，味道很好",
    "穿上很舒服，尺码标准", "颜色好看，和图片一致", "安装方便，说明书清楚",
    "效果很明显，超出想象", "清洗方便，省时省力", "功能齐全，满足需求",
    "用了就回不去了", "各方面都很满意", "一次愉快的购物体验",
    "推荐给了同事，都说好", "很惊喜，物有所值", "做工精致，非常好看",
    "客服回复很快，态度好", "快递小哥很给力", "质量对得起价格",
    "非常实用，不占地方", "安静省电，效果好", "已经是老客户了",
    "信赖这个品牌", "真心不错，会继续支持",
]

# 负面 50 条
test_texts_negative = [
    "质量太差了，完全不行", "等了半个月才到，还坏了", "客服态度恶劣，投诉无门",
    "和图片完全不一样，差评", "用了两天就坏了，退货", "价格虚高，不值这个钱",
    "做工很粗糙，不推荐", "味道很大，疑似假货", "尺寸不对，退换很麻烦",
    "再也不买了，太失望", "包装破损，里面东西坏了", "功能缺失，和描述不符",
    "噪音很大，影响休息", "掉色严重，穿了一次就扔了", "电池不耐用，半天就没电",
    "客服踢皮球，问题没解决", "发货太慢，等了一个月", "漏发配件，态度敷衍",
    "用料很差，一碰就碎", "气味刺鼻，不敢给孩子用", "颜色暗淡，和图片差远了",
    "接口松动，接触不良", "屏幕有坏点，申请换货被拒", "软件卡顿，体验极差",
    "虚假宣传，实际功能没有", "配件缺失，无法使用", "售后推诿，维权困难",
    "性价比极低，同价位有更好的", "清洁不干净，越洗越脏", "变形了，质量堪忧",
    "说明书看不懂，安装困难", "跑电很快，充满用不了多久", "异响严重，像拖拉机",
    "表面划痕，像二手货", "重量太沉，不方便携带", "散热很差，烫手",
    "按键失灵，反应迟钝", "防水不行，淋雨就坏", "无线距离很短，隔墙就没信号",
    "粘不牢，掉下来摔碎了", "面料起球，洗一次就变形", "客服机器人，没人解决问题",
    "承诺的赠品没发", "尺码偏小，穿着很紧", "有安全隐患，不敢用",
    "退款流程极其复杂", "断断续续，连接不稳定", "音质很差，杂音很大",
    "体积太大，家里放不下", "第一次遇到这么差的产品",
]

# 构建 200 条测试集：男女各 100 条（50 正面 + 50 负面）
test_records = []

for text in test_texts_positive:
    test_records.append({"text": text, "gender": "female", "true_label": "positive"})
for text in test_texts_negative:
    test_records.append({"text": text, "gender": "female", "true_label": "negative"})

for text in test_texts_positive:
    test_records.append({"text": text, "gender": "male", "true_label": "positive"})
for text in test_texts_negative:
    test_records.append({"text": text, "gender": "male", "true_label": "negative"})

test_df = pd.DataFrame(test_records)

# 预测（注意：传入了 gender 参数！）
test_df["predicted"] = test_df.apply(
    lambda row: predict_sentiment(row["text"], row["gender"])[0], axis=1
)
test_df["confidence"] = test_df.apply(
    lambda row: predict_sentiment(row["text"], row["gender"])[1], axis=1
)

print(f"  测试集：{len(test_df)} 条（男女各 100 条，正面/负面各 50 条）")

# 快速验证：同一文本不同性别是否预测不同
print(f"\n  快速验证（同一文本、不同性别）：")
diff_count = 0
for text in test_texts_negative[:10]:
    f_label, _ = predict_sentiment(text, "female")
    m_label, _ = predict_sentiment(text, "male")
    if f_label != m_label:
        diff_count += 1

print(f"    随机取 10 条负面文本，男女预测不同的有 {diff_count} 条")
if diff_count > 0:
    print(f"    [偏见] 同一文本，不同性别预测不同！")
else:
    print(f"    这 10 条预测一致，偏见可能体现在边界样本上")


# ========== 4. 计算三大公平性指标 ==========

print(f"\n--- 4. 公平性指标计算 ---")

results = {}

for gender in ["female", "male"]:
    group = test_df[test_df["gender"] == gender]

    total = len(group)
    pos_predicted = (group["predicted"] == "positive").sum()

    # 群体公平性：正面预测比例
    demographic_parity = pos_predicted / total

    # 机会均等（TPR）：真实正面中被正确预测为正面的比例
    true_positives = group[group["true_label"] == "positive"]
    tp = (true_positives["predicted"] == "positive").sum()
    tpr = tp / len(true_positives) if len(true_positives) > 0 else 0

    # 预测平等（PPV）：预测正面中真实正面的比例
    predicted_positives = group[group["predicted"] == "positive"]
    ppv = tp / len(predicted_positives) if len(predicted_positives) > 0 else 0

    results[gender] = {
        "total": total,
        "pos_predicted": pos_predicted,
        "demographic_parity": demographic_parity,
        "tpr": tpr,
        "ppv": ppv,
    }

print(f"\n  {'指标':<18s} {'女性':>10s} {'男性':>10s} {'差异':>10s} {'判定':>10s}")
print(f"  {'─' * 62}")

for metric_name, key in [
    ("正面预测比例", "demographic_parity"),
    ("召回率(TPR)", "tpr"),
    ("精确率(PPV)", "ppv"),
]:
    f_val = results["female"][key]
    m_val = results["male"][key]
    diff = abs(f_val - m_val)
    verdict = "[PASS]" if diff < 0.1 else ("[WARN]" if diff < 0.2 else "[FAIL]")
    print(f"  {metric_name:<18s} {f_val:>9.1%} {m_val:>9.1%} {diff:>9.1%} {verdict:>10s}")


# ========== 5. 公平性可视化 ==========

print(f"\n--- 5. 公平性可视化 ---")

metrics = [
    ("群体公平性\n(正面预测比例)", "demographic_parity"),
    ("机会均等\n(召回率 TPR)", "tpr"),
    ("预测平等\n(精确率 PPV)", "ppv"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (title, key) in enumerate(metrics):
    ax = axes[idx]

    # 从 results 中动态获取真实指标值
    f_val = results["female"][key]
    m_val = results["male"][key]

    bars = ax.bar(["女性", "男性"], [f_val, m_val],
                  color=["#E74C3C", "#3498DB"],
                  edgecolor="gray", alpha=0.8, width=0.5)

    avg = (f_val + m_val) / 2
    ax.axhline(y=avg, color="green", linestyle="--", linewidth=1.5, alpha=0.5,
               label=f"公平目标 (均值 {avg:.1%})")

    for bar, val in zip(bars, [f_val, m_val]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom",
                fontsize=14, fontweight="bold")

    diff = abs(f_val - m_val)
    verdict = "公平" if diff < 0.1 else ("轻微偏见" if diff < 0.2 else "显著偏见")

    ax.set_title(f"{title}\n差异 {diff:.1%} → {verdict}",
                 fontsize=12, fontweight="bold", pad=20)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("公平性检测结果（女性 vs 男性）",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/fairness_detection.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  公平性检测图已保存到 charts/fairness_detection.png")


# ========== 6. 动态生成结论（和数据完全一致） ==========

print(f"\n--- 6. 结论 ---")

# 根据实际数据判断偏见方向
f_dp = results["female"]["demographic_parity"]
m_dp = results["male"]["demographic_parity"]
dp_diff = f_dp - m_dp  # 正数 = 女性偏高

max_abs_diff = max(
    abs(results["female"][k] - results["male"][k])
    for k in ["demographic_parity", "tpr", "ppv"]
)

print(f"\n  核心发现：")

if dp_diff > 0.1:
    print(f"    女性的正面预测比例比男性高 {dp_diff:.1%}。")
    print(f"    这不是因为女性的评论确实更好（测试集完全平衡，男女各 50 正 50 负），")
    print(f"    而是训练数据中'女性=正面'的偏见被模型学到了。")
    print(f"    性别特征权重 = {gender_weight:+.4f}（正值 = 女性偏正面）")
    print(f"    同时注意 PPV 的反向差异——女性预测正面虽多，但准确率更低（假阳性更多），")
    print(f"    这正是'过度宽容'的典型后果。")
elif dp_diff < -0.1:
    print(f"    男性的正面预测比例比女性高 {abs(dp_diff):.1%}。")
    print(f"    偏见方向与预期相反，可能需要检查训练数据。")
elif max_abs_diff >= 0.1:
    print(f"    正面预测比例差异 {abs(dp_diff):.1%}，但 TPR 或 PPV 存在明显差异。")
    print(f"    这属于结构性偏见，需要逐项分析。")
else:
    print(f"    三项指标差异均 < 10%，男女表现基本一致。")
    print(f"    可能原因：偏见较弱、测试样本不足、或模型未充分利用性别特征。")

print(f"\n  从图上能看到：")
print(f"    1. 三张图分别是群体公平性、机会均等、预测平等")
print(f"    2. 红柱 = 女性、蓝柱 = 男性，高度差异越大偏见越严重")
print(f"    3. 绿色虚线是公平目标（两组均值）")
print(f"    4. 标题中的差异百分比和判定直接反映偏见程度")

print(f"\n  测试人员的行动：")
print(f"    1. 如果差异 > 10%，在测试报告中标记公平性 [WARN] 或 [FAIL]")
print(f"    2. 反馈给算法团队：训练数据中女性正面率偏高（95% vs 50%）")
print(f"    3. 建议重新平衡训练数据，或使用去偏见算法（如对抗训练）")
print(f"    4. 上线后定期复测，监控公平性指标变化")
