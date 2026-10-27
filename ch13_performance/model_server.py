"""
简易模型 API 服务
作为性能测试的被测系统（System Under Test）

启动后提供两个接口：
  POST /predict  —— 情感分类预测
  GET  /health   —— 健康检查
"""

from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import time

app = Flask(__name__)

# ========== 加载模型（服务启动时加载一次）==========

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

print("模型加载完成，服务就绪")


# ========== API 接口 ==========

@app.route("/predict", methods=["POST"])
def predict():
    """
    情感分类预测接口

    请求体：{"text": "输入文本"}
    返回：{"label": "positive", "confidence": 0.85, "latency_ms": 12.3}
    """
    start_time = time.time()

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "缺少 text 参数"}), 400

    text = data["text"]

    if not text or not text.strip():
        return jsonify({"error": "text 不能为空"}), 400

    # 模型推理
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))

    latency_ms = (time.time() - start_time) * 1000

    return jsonify({
        "label": str(prediction),
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2)
    })


@app.route("/health", methods=["GET"])
def health():
    """健康检查接口"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("启动服务：http://localhost:5000")
    print("预测接口：POST http://localhost:5000/predict")
    print("健康检查：GET  http://localhost:5000/health")
    app.run(host="0.0.0.0", port=5000, threaded=True)
