"""
机器学习基础概念实操
用 scikit-learn 体验机器学习的完整流程
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("一、监督学习演示：分类任务")
print("=" * 60)

# ========== 加载数据 ==========
# 用经典的鸢尾花数据集（Iris Dataset）
# 这是一个分类任务：根据花瓣和花萼的尺寸，判断花的品种
iris = load_iris()

# X 是特征数据（输入）：花萼长度、花萼宽度、花瓣长度、花瓣宽度
X = iris.data
# y 是标签数据（输出/答案）：0=山鸢尾, 1=变色鸢尾, 2=维吉尼亚鸢尾
y = iris.target

print(f"\n数据集大小：{X.shape[0]} 条样本")
print(f"特征数量：{X.shape[1]} 个")
print(f"类别：{iris.target_names}")
print(f"\n前 5 条数据的特征值：")
for i in range(5):
    print(f"  样本{i+1}: {X[i]} → 品种: {iris.target_names[y[i]]}")

# ========== 训练模型 ==========
# 先把数据分成训练集和测试集（后面详细讲为什么）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

print(f"\n训练集：{X_train.shape[0]} 条")
print(f"测试集：{X_test.shape[0]} 条")

# 用决策树算法训练模型
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)    # fit() 就是"训练"——让模型从数据中学习规律

# ========== 评估模型 ==========
# 用测试集来检验模型学得好不好
y_pred = model.predict(X_test)    # 让模型对测试集做预测

accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率：{accuracy:.2%}")
print(f"\n详细评估报告：")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# ========== 二、无监督学习演示：聚类 ==========

print("\n" + "=" * 60)
print("二、无监督学习演示：聚类")
print("=" * 60)

from sklearn.cluster import KMeans

# 还是用鸢尾花数据，但这次不给标签（y），让模型自己分
X_unlabeled = iris.data

# K-Means 聚类：把数据自动分成 3 组
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_unlabeled)

print(f"\n模型自动分成了 3 组：")
for cluster_id in range(3):
    count = np.sum(clusters == cluster_id)
    print(f"  第 {cluster_id} 组：{count} 条数据")

# 对比无监督分组和真实标签
print(f"\n无监督分组 vs 真实标签对比：")
from scipy.stats import mode
# 对每个聚类组，找它对应最多的真实标签
mapping = {}
for cluster_id in range(3):
    mask = clusters == cluster_id
    true_labels = y[mask]
    most_common = mode(true_labels, keepdims=True).mode[0]
    mapping[cluster_id] = iris.target_names[most_common]
    print(f"  第 {cluster_id} 组 → 大部分是 {iris.target_names[most_common]}"
          f"（真实标签分布：{dict(zip(*np.unique(true_labels, return_counts=True)))}）")
