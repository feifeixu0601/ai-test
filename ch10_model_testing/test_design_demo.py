"""
测试用例设计方法演示
用代码展示等价类划分、边界值分析、失败模式分析的方法论

核心思路：
  不是拍脑袋想测试用例，而是按方法系统地生成
"""

print("=" * 60)
print("测试用例设计方法演示")
print("=" * 60)


# ========== 方法一：等价类划分 ==========

print("\n--- 方法一：等价类划分 ---")
print("核心思想：把输入分成几类，每类取代表样本")

# 步骤1：列出所有等价类
equivalence_classes = {
    "明确正面": {
        "描述": "包含明确正面词汇的文本",
        "关键词": ["好", "棒", "满意", "推荐", "喜欢"],
        "期望": "positive",
        "样本数": 3,
    },
    "明确负面": {
        "描述": "包含明确负面词汇的文本",
        "关键词": ["差", "烂", "失望", "退货", "垃圾"],
        "期望": "negative",
        "样本数": 3,
    },
    "明确中性": {
        "描述": "没有明显情感倾向的文本",
        "关键词": ["一般", "还行", "凑合", "普通"],
        "期望": "neutral",
        "样本数": 2,
    },
    "褒贬混合": {
        "描述": "同时包含正面和负面词汇",
        "关键词": ["好看", "差"],   # 举例
        "期望": "视具体语义判断",
        "样本数": 2,
    },
    "空/异常": {
        "描述": "空字符串、纯数字、纯符号等",
        "关键词": ["", "123", "!!!"],
        "期望": "error 或低置信度",
        "样本数": 3,
    },
}

# 步骤2：为每个等价类生成测试用例
test_cases = []
case_id = 1

for category, info in equivalence_classes.items():
    print(f"\n  等价类：{category}")
    print(f"    描述：{info['描述']}")
    print(f"    期望：{info['期望']}")
    print(f"    需要 {info['样本数']} 个样本")

    for i in range(info["样本数"]):
        test_cases.append({
            "id": f"EC{case_id:02d}",
            "method": "等价类划分",
            "category": category,
            "description": info["描述"],
            "expected": info["期望"],
        })
        print(f"      用例 EC{case_id:02d}")
        case_id += 1

print(f"\n  等价类划分共产出 {len(test_cases)} 个用例")


# ========== 方法二：边界值分析 ==========

print(f"\n--- 方法二：边界值分析 ---")
print("核心思想：在模型决策边界附近测试，最容易暴露问题")

# AI 模型的"边界"不是输入值的边界，而是置信度的边界
# 比如二分类模型以 0.5 为分界线，0.4~0.6 之间就是"危险区"

boundary_cases = [
    {
        "id": "BV01",
        "scenario": "模糊情感（信心度约 0.5）",
        "description": "褒贬各半、模棱两可的文本",
        "example": "还行吧，不算好也不算差",
        "expected": "neutral",
        "reason": "模型在决策边界附近容易摇摆",
    },
    {
        "id": "BV02",
        "scenario": "极短文本",
        "description": "只有 1-2 个字的输入",
        "example": "好",
        "expected": "positive",
        "reason": "信息量极少，模型容易不确定",
    },
    {
        "id": "BV03",
        "scenario": "极长文本",
        "description": "远超训练数据长度的输入",
        "example": "这个产品很好（重复 200 次）",
        "expected": "positive",
        "reason": "超长输入可能触发截断或性能问题",
    },
    {
        "id": "BV04",
        "scenario": "刚好在类别边界",
        "description": "非常接近分类分界线的输入",
        "example": "有点好又有点差",
        "expected": "neutral",
        "reason": "正负面特征势均力敌",
    },
]

print(f"\n  边界值分析共产出 {len(boundary_cases)} 个用例")
for case in boundary_cases:
    print(f"    [{case['id']}] {case['scenario']}")
    print(f"        例：'{case['example']}'")
    print(f"        测试目的：{case['reason']}")


# ========== 方法三：失败模式分析（错误推测法） ==========

print(f"\n--- 方法三：失败模式分析 ---")
print("核心思想：根据 AI 模型的已知弱点，针对性设计用例")

failure_modes = [
    {
        "id": "FM01",
        "weakness": "否定词混淆",
        "description": "模型看到正面词就判正面，忽略否定词",
        "test_input": "不是很好",
        "expected": "negative",
    },
    {
        "id": "FM02",
        "weakness": "否定词混淆",
        "description": "双重否定，模型可能处理不了",
        "test_input": "不是不满意",
        "expected": "positive",
    },
    {
        "id": "FM03",
        "weakness": "反讽/讽刺",
        "description": "表面正面、实际负面的表达",
        "test_input": "真是好极了才怪",
        "expected": "negative",
    },
    {
        "id": "FM04",
        "weakness": "领域偏移",
        "description": "训练数据中没有的领域词汇",
        "test_input": "这款芯片的 TOPS 性能吊打竞品",
        "expected": "positive（但模型可能不理解）",
    },
    {
        "id": "FM05",
        "weakness": "语序依赖",
        "description": "先褒后贬 vs 先贬后褒，侧重点不同",
        "test_input": "外观好看但是质量很差",
        "expected": "negative（关注核心负面）",
    },
]

print(f"\n  失败模式分析共产出 {len(failure_modes)} 个用例")
for case in failure_modes:
    print(f"    [{case['id']}] 弱点：{case['weakness']}")
    print(f"        输入：'{case['test_input']}'")
    print(f"        期望：{case['expected']}")


# ========== 汇总 ==========

print(f"\n{'=' * 60}")
print(f"测试用例设计汇总")
print(f"{'=' * 60}")

total = len(test_cases) + len(boundary_cases) + len(failure_modes)
print(f"\n  等价类划分：    {len(test_cases)} 个用例")
print(f"  边界值分析：    {len(boundary_cases)} 个用例")
print(f"  失败模式分析：  {len(failure_modes)} 个用例")
print(f"  ──────────────────────")
print(f"  合计：          {total} 个用例")

print(f"\n  这只是三种方法的产出。实际测试中还要加：")
print(f"    - 异常输入测试（空值、乱码、非目标语言）")
print(f"    - 鲁棒性测试（同义替换、加噪声）")
print(f"    - 置信度合理性测试（确定的高置信，模糊的低置信）")
print(f"\n  这些方法加在一起，就构成了第三节实战中的 29 个测试用例。")