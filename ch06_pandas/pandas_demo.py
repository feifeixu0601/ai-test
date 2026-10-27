"""
Pandas 基础知识练习
每段代码可以独立运行
"""

import pandas as pd

# ========== 创建 DataFrame ==========

# 用字典创建（最常用）
data = {
    "职位名称": ["AI测试工程师", "高级AI测试开发", "大模型测试工程师", "CV测试工程师", "NLP测试工程师"],
    "薪资下限": [25000, 35000, 30000, 22000, 25000],
    "薪资上限": [40000, 60000, 50000, 38000, 45000],
    "城市": ["北京", "上海", "杭州", "成都", "广州"],
    "经验": ["3-5年", "5-10年", "3-5年", "1-3年", "3-5年"]
}

df = pd.DataFrame(data)

# 查看前几行（默认前5行）
print("--- 前 5 行数据 ---")
print(df.head())

# 查看后几行
print("\n--- 后 3 行数据 ---")
print(df.tail(3))

# 查看基本信息：行数、列数、每列数据类型、缺失值数量
print("\n--- 基本信息 ---")
print(df.info())

# 查看统计摘要（数值列的均值、标准差、最小值、最大值等）
print("\n--- 统计摘要 ---")
print(df.describe())


# ========== 选取列和行 ==========

print("\n--- 选取列 ---")

# 选一列（返回 Series）
print(df["城市"])
print(type(df["城市"]))   # <class 'pandas.core.series.Series'>

# 选多列（返回 DataFrame）
print(df[["职位名称", "城市"]])

print("\n--- 选取行 ---")

# 按索引选取行（iloc：按位置）
print(df.iloc[0])      # 第 1 行
print(df.iloc[1:3])    # 第 2 到第 3 行

# 按标签选取行（loc：按索引标签）
print(df.loc[0])       # 索引为 0 的行
print(df.loc[0:2])     # 索引 0 到 2 的行（注意：loc 是包含右端点的）

print("\n--- 条件筛选 ---")

# 筛选北京的岗位
beijing = df[df["城市"] == "北京"]
print(beijing)

# 筛选薪资上限超过 50000 的
high_salary = df[df["薪资上限"] > 50000]
print(high_salary)

# 组合条件（注意：每个条件要用括号包起来，用 & 和 | 连接）
result = df[(df["城市"] == "北京") & (df["薪资上限"] > 35000)]
print(result)


# ========== 添加列和修改数据 ==========

print("\n--- 添加列 ---")

# 直接赋值就创建了新列
df["平均薪资"] = (df["薪资下限"] + df["薪资上限"]) / 2
print(df[["职位名称", "薪资下限", "薪资上限", "平均薪资"]])

# 用 apply 对每行做计算（非常常用）
df["薪资跨度"] = df.apply(lambda row: row["薪资上限"] - row["薪资下限"], axis=1)
print(df[["职位名称", "薪资下限", "薪资上限", "薪资跨度"]])

print("\n--- 修改数据 ---")

# 修改单个值
df.loc[0, "城市"] = "北京（总部）"
print(df.loc[0])

# 批量替换
df["城市"] = df["城市"].str.replace("（总部）", "")
print(df["城市"])


# ========== 缺失值处理 ==========

print("\n--- 缺失值处理 ---")

# 制造一些缺失值用于演示
import numpy as np

df_dirty = pd.DataFrame({
    "name": ["张三", "李四", None, "王五"],
    "salary": [25000, np.nan, 35000, np.nan],
    "city": ["北京", "上海", "杭州", None]
})

print("原始数据：")
print(df_dirty)

# 1. 检测缺失值
print("\n每列缺失值数量：")
print(df_dirty.isnull().sum())

# 2. 删除包含缺失值的行
print("\n删除有缺失值的行：")
print(df_dirty.dropna())

# 3. 填充缺失值
print("\n用默认值填充：")
print(df_dirty.fillna({"name": "未知", "salary": 0, "city": "未知"}))

# 4. 用均值/中位数填充数值列
print("\n用均值填充薪资：")
salary_mean = df_dirty["salary"].mean()
print(df_dirty.fillna({"salary": salary_mean}))


# ========== 分组统计 ==========

print("\n--- 分组统计 ---")

# 按城市分组，计算平均薪资
print("各城市平均薪资上限：")
print(df.groupby("城市")["薪资上限"].mean())

# 按城市分组，计算多种统计量
print("\n各城市薪资统计：")
print(df.groupby("城市")[["薪资下限", "薪资上限"]].agg(["mean", "min", "max", "count"]))

# 按经验分组，计算平均薪资
print("\n各经验等级平均薪资：")
print(df.groupby("经验")[["薪资下限", "薪资上限"]].mean())

# 多级分组（按城市 + 经验）
print("\n城市 × 经验 分组统计：")
print(df.groupby(["城市", "经验"])["薪资上限"].mean())


# ========== 排序和计数 ==========

print("\n--- 排序 ---")

# 按薪资上限降序排列
print(df.sort_values("薪资上限", ascending=False))

# 按多列排序（先按城市升序，再按薪资上限降序）
print(df.sort_values(["城市", "薪资上限"], ascending=[True, False]))

print("\n--- 计数 ---")

# 统计每个城市的岗位数量
print(df["城市"].value_counts())

# 统计每个经验等级的岗位数量
print(df["经验"].value_counts())


# ========== 读取 CSV 文件 ==========

print("\n--- 读取 CSV ---")

# 读取第04篇生成的岗位数据 CSV
try:
    job_df = pd.read_csv("../data/job_data.csv", encoding="utf-8-sig")
    print(f"读取成功，共 {len(job_df)} 行，{len(job_df.columns)} 列")
    print(f"列名：{list(job_df.columns)}")
    print(job_df.head(3))
except FileNotFoundError:
    print("job_data.csv 不存在，请先运行第04篇的爬虫代码")


# ========== 读取 JSON 文件 ==========

print("\n--- 读取 JSON ---")

# 读取第04篇生成的岗位数据 JSON
try:
    job_json_df = pd.read_json("../data/job_data.json")
    print(f"读取成功，共 {len(job_json_df)} 行")
    print(job_json_df.head(3))
except FileNotFoundError:
    print("job_data.json 不存在，请先运行第04篇的爬虫代码")

# JSON 格式不同，参数也不同
# 如果是 JSON Lines（每行一个 JSON 对象）
# df = pd.read_json("data.jsonl", lines=True)
