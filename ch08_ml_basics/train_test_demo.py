"""
训练集/测试集划分实验
演示不同划分方式对评估结果的影响
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("=" * 60)
print("训练集/测试集划分实验")
print("=" * 60)

iris = load_iris()
X, y = iris.data, iris.target

# ========== 实验1：不同随机种子的影响 ==========
print("\n--- 实验1：随机划分 5 次，看结果波动 ---")

accuracies = []
for seed in range(5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)
    print(f"  随机种子={seed}：准确率 = {acc:.2%}")

print(f"\n  5 次结果的均值：{np.mean(accuracies):.2%}")
print(f"  5 次结果的标准差：{np.std(accuracies):.2%}")
print(f"  波动范围：{min(accuracies):.2%} ~ {max(accuracies):.2%}")

# ========== 实验2：交叉验证 ==========
print("\n--- 实验2：5 折交叉验证 ---")

model = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print(f"  5 折交叉验证结果：{[f'{s:.2%}' for s in cv_scores]}")
print(f"  平均准确率：{cv_scores.mean():.2%}")
print(f"  标准差：{cv_scores.std():.2%}")

# ========== 实验3：数据集大小对结果的影响 ==========
print("\n--- 实验3：不同测试集比例的影响 ---")

for test_ratio in [0.1, 0.2, 0.3, 0.5]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  测试集比例 {test_ratio:.0%}：训练 {len(X_train)} 条，"
          f"测试 {len(X_test)} 条，准确率 = {acc:.2%}")
