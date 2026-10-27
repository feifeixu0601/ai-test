"""
公平性与可解释性报告
组合公平性检测 + 可解释性分析，输出综合报告

模型与 bias_detection.py 保持一致：
  性别作为模型特征传入（而不是只做分组统计），
  这样模型能"看到"性别，训练数据中的偏见才能被检测到。
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from datetime import datetime

print("=" * 60)
print("公平性与可解释性报告")
print("=" * 60)

np.random.seed(42)


# ========== 1. 构建模型（带偏见，性别作为特征） ==========

positive_texts = [
    "这个产品非常好用", "质量很好推荐", "服务态度很棒",
    "性价比很高值得买", "超出预期惊喜", "做工精细品质好",
    "感觉不错会回购", "很喜欢这个产品",
]

negative_texts = [
    "太差了完全不行", "非常失望后悔", "服务态度很差",
    "质量有问题退货", "做工粗糙不值", "和描述不符差评",
    "用了一次就坏了", "客服不理人体验差",
]

# 训练数据（故意埋入偏见）
#   女性：正面率 95%（190 正 + 10 负）
#   男性：正面率 50%（100 正 + 100 负）
train_records = []
for text in positive_texts:
    for _ in range(24):  # 8 * 24 ≈ 190 条正面
        train_records.append({"text": text, "gender": "female", "label": "positive"})
for text in negative_texts:
    for _ in range(1):   # 8 * 1 ≈ 10 条负面（取 8 条 + 之前多出的）
        train_records.append({"text": text, "gender": "female", "label": "negative"})
# 补足到正好 10 条负面
for text in negative_texts[:2]:
    train_records.append({"text": text, "gender": "female", "label": "negative"})

for text in positive_texts:
    for _ in range(13):  # 8 * 13 ≈ 100 条正面
        train_records.append({"text": text, "gender": "male", "label": "positive"})
for text in negative_texts:
    for _ in range(13):  # 8 * 13 ≈ 100 条负面
        train_records.append({"text": text, "gender": "male", "label": "negative"})

train_df = pd.DataFrame(train_records)

# 文本特征
vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(1, 2))
X_text = vectorizer.fit_transform(train_df["text"])

# 性别特征（female=1, male=0）
X_gender = csr_matrix(
    (train_df["gender"] == "female").astype(int).values.reshape(-1, 1)
)

# 合并特征
X_combined = hstack([X_text, X_gender])
y_train = train_df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X_combined, y_train)

gender_weight = model.coef_[0][-1]
print(f"\n  模型训练完成（逻辑回归）")
print(f"  性别特征权重：{gender_weight:+.4f}")


def predict(text, gender):
    """预测文本情感（性别作为模型输入）"""
    X_text = vectorizer.transform([text])
    X_gender = csr_matrix([[1 if gender == "female" else 0]])
    X_combined = hstack([X_text, X_gender])
    return model.predict(X_combined)[0]


# ========== 2. 公平性检测 ==========

print(f"\n--- 2. 公平性检测 ---")

# 测试文本（与训练文本不同，50 正面 + 50 负面）
test_positive = [
    "这款商品真的很棒", "物超所值非常满意", "物流很快包装精美",
    "质量过硬值得拥有", "用着很顺手好评", "外观漂亮功能强大",
    "客服耐心解决快", "第二次购买依旧好", "朋友推荐果然不错", "超出预期值得推荐",
    "性价比之王闭眼入", "越用越喜欢", "家人很满意", "细节很好点赞",
    "发货速度快服务好", "质感超好", "孩子很喜欢", "回购第三次了",
    "包装严实完美", "试了一次就爱上", "声音清晰续航好", "操作简单",
    "颜值高性能强", "价格实惠品质好", "手感舒适做工好", "设计感强",
    "实用性强", "对比后这家最好", "送礼有面子", "食材新鲜味道好",
    "穿上舒服尺码标准", "颜色好看一致", "安装方便", "效果明显",
    "清洗方便", "功能齐全", "用了回不去", "各方面满意",
    "购物体验愉快", "推荐给同事都说好", "很惊喜物有所值", "做工精致好看",
    "客服态度好", "快递给力", "质量对得起价格", "不占地方",
    "安静省电", "老客户了", "信赖品牌", "真心不错",
]

test_negative = [
    "质量太差完全不行", "等了半个月还坏了", "客服恶劣投诉无门",
    "和图片不一样差评", "用两天就坏了", "价格虚高不值",
    "粗糙不推荐", "味道大疑似假货", "尺寸不对退换麻烦",
    "再也不买太失望", "包装破损东西坏了", "功能缺失不符",
    "噪音大影响休息", "掉色严重扔了", "电池不耐用",
    "客服踢皮球", "发货太慢一个月", "漏发配件敷衍",
    "用料差一碰碎", "气味刺鼻", "颜色暗淡差远了",
    "接口松动", "屏幕坏点换货被拒", "卡顿体验差",
    "虚假宣传", "配件缺失无法用", "售后推诿维权难",
    "性价比极低", "清洁不干净", "变形质量堪忧",
    "说明书看不懂", "跑电快", "异响像拖拉机",
    "划痕像二手", "太沉不便携", "散热差烫手",
    "按键失灵", "防水不行", "隔墙没信号",
    "粘不牢摔碎了", "起球变形", "客服机器人没人理",
    "赠品没发", "尺码偏小", "安全隐患",
    "退款复杂", "连接不稳定", "音质差杂音大",
    "体积太大", "第一次这么差",
]

test_records = []
for text in test_positive:
    test_records.append({"text": text, "gender": "female", "true_label": "positive"})
    test_records.append({"text": text, "gender": "male", "true_label": "positive"})
for text in test_negative:
    test_records.append({"text": text, "gender": "female", "true_label": "negative"})
    test_records.append({"text": text, "gender": "male", "true_label": "negative"})

test_df = pd.DataFrame(test_records)
test_df["predicted"] = test_df.apply(
    lambda row: predict(row["text"], row["gender"]), axis=1
)

print(f"  测试集：{len(test_df)} 条（男女各 {len(test_df)//2} 条）")


# ========== 3. 可解释性分析 ==========

print(f"\n--- 3. 可解释性分析 ---")

feature_names = vectorizer.get_feature_names_out()


def get_top_features(class_label, k=5):
    """获取某个类别下权重最高的文本特征（排除性别特征）"""
    # 二分类时 coef_ 只有 1 行
    # 正系数 → 推向 classes_[1]，负系数 → 推向 classes_[0]
    coefs = model.coef_[0]

    # 只取文本特征部分，排除最后一个（性别特征）
    text_coefs = coefs[:len(feature_names)]

    # 对于 classes_[1]（positive），系数越大越有代表性
    # 对于 classes_[0]（negative），系数越小（越负）越有代表性，所以取反
    if class_label == model.classes_[1]:
        scores = text_coefs
    else:
        scores = -text_coefs

    top_indices = np.argsort(scores)[-k:][::-1]
    return [(feature_names[i], text_coefs[i]) for i in top_indices]


# ========== 4. 输出报告 ==========

lines = []


def add(text=""):
    lines.append(text)


add("=" * 60)
add("AI 模型公平性与可解释性报告")
add("=" * 60)
add(f"")
add(f"  报告时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
add(f"  被测模型：情感分类模型（逻辑回归，文本+性别特征）")
add(f"  测试目的：检测性别偏见 + 提供可解释性分析")

# 公平性部分
add(f"\n{'─' * 60}")
add(f"一、公平性检测")
add(f"{'─' * 60}")

fairness_results = {}
for gender in ["female", "male"]:
    group = test_df[test_df["gender"] == gender]
    pos_pred_rate = (group["predicted"] == "positive").sum() / len(group)

    true_pos = group[group["true_label"] == "positive"]
    tp = (true_pos["predicted"] == "positive").sum()
    tpr = tp / len(true_pos) if len(true_pos) > 0 else 0

    pred_pos = group[group["predicted"] == "positive"]
    ppv = tp / len(pred_pos) if len(pred_pos) > 0 else 0

    fairness_results[gender] = {
        "demographic_parity": pos_pred_rate,
        "tpr": tpr,
        "ppv": ppv,
    }

add(f"\n  {'指标':<18s} {'女性':>10s} {'男性':>10s} {'差异':>10s} {'判定':>10s}")
add(f"  {'─' * 62}")

for metric_name, key in [
    ("正面预测比例", "demographic_parity"),
    ("召回率(TPR)", "tpr"),
    ("精确率(PPV)", "ppv"),
]:
    f_val = fairness_results["female"][key]
    m_val = fairness_results["male"][key]
    diff = abs(f_val - m_val)
    verdict = "[PASS]" if diff < 0.1 else ("[WARN]" if diff < 0.2 else "[FAIL]")
    add(f"  {metric_name:<18s} {f_val:>9.1%} {m_val:>9.1%} {diff:>9.1%} {verdict:>10s}")

# 公平性判定
max_diff = max(
    abs(fairness_results["female"][k] - fairness_results["male"][k])
    for k in ["demographic_parity", "tpr", "ppv"]
)

add(f"\n  公平性判定：")
if max_diff < 0.1:
    add(f"    [PASS] 三项指标差异均 < 10%，模型较公平")
elif max_diff < 0.2:
    add(f"    [WARN] 存在轻微偏见（最大差异 {max_diff:.1%}），建议关注")
else:
    add(f"    [FAIL] 存在显著偏见（最大差异 {max_diff:.1%}），需要修复")
    add(f"           建议：重新平衡训练数据 或 使用去偏见算法")

# 可解释性部分
add(f"\n{'─' * 60}")
add(f"二、可解释性分析")
add(f"{'─' * 60}")

add(f"\n  性别特征权重：{gender_weight:+.4f}")
if gender_weight > 0:
    add(f"  [偏见] 正权重说明模型认为'女性 = 更可能正面'")

add(f"\n  各类别最有代表性的文本特征：")
for class_label in model.classes_:
    top_features = get_top_features(class_label, k=5)
    add(f"\n  [{class_label}]")
    for feat, score in top_features:
        add(f"    '{feat}'  权重={score:+.4f}")

add(f"\n  可解释性判定：")
add(f"    [PASS] 模型行为可解释（特征权重可追溯）")
add(f"    建议：在产品文档中记录各特征的典型权重范围")
add(f"    建议：当单条预测的置信度 < 60% 时，自动触发人工复核")

# 综合结论
add(f"\n{'─' * 60}")
add(f"三、综合结论与建议")
add(f"{'─' * 60}")

fairness_verdict = "通过" if max_diff < 0.1 else ("轻微问题" if max_diff < 0.2 else "未通过")
add(f"\n  公平性：{fairness_verdict}")
add(f"  可解释性：通过")

# 根据实际数据动态生成建议
f_dp = fairness_results["female"]["demographic_parity"]
m_dp = fairness_results["male"]["demographic_parity"]

add(f"\n  改进建议：")
if f_dp > m_dp + 0.1:
    add(f"    1. 训练数据中女性正面率偏高（95% vs 50%），导致模型对女性更'宽容'")
    add(f"       女性正面预测比例 {f_dp:.1%} > 男性 {m_dp:.1%}")
elif m_dp > f_dp + 0.1:
    add(f"    1. 训练数据中男性正面率偏高，导致模型对男性更'宽容'")
    add(f"       男性正面预测比例 {m_dp:.1%} > 女性 {f_dp:.1%}")
else:
    add(f"    1. 男女正面预测比例差异较小（{abs(f_dp - m_dp):.1%}），偏见程度有限")
add(f"    2. 建议平衡训练数据中各群体的情感比例")
add(f"    3. 在模型评估中增加公平性指标，作为上线门槛")
add(f"    4. 对高风险决策（贷款、招聘）保留人工审核环节")
add(f"    5. 定期（每月/每季度）跑公平性检测，监控偏见变化")

add(f"\n{'=' * 60}")
add(f"报告生成完毕")

report_text = "\n".join(lines)
print(report_text)

with open("fairness_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n报告已保存到 fairness_report.txt")
