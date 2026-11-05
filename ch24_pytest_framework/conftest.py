import pytest, json, os

MODEL = {"model": "kimi-k2.6", "base_url": "https://api.moonshot.cn/v1"}

@pytest.fixture(scope="session")
def model_config():
    return MODEL.copy()

@pytest.fixture(scope="session")
def hallucination_cases():
    with open("test_data/hallucination_cases.json", encoding="utf-8") as f:
        return json.load(f)