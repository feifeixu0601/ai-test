import pytest, allure
from ai_helpers import call_simple, score_facts, attach


@pytest.mark.smoke
@pytest.mark.hallucination
@pytest.mark.api
@pytest.mark.parametrize("case_id", ["HALL-01", "HALL-02"])
def test_facts(case_id, hallucination_cases, model_config):
    """幻觉检测：事实点评分 ≥ 0.8"""
    case = next(c for c in hallucination_cases if c["id"] == case_id)

    answer = call_simple(case["question"], **model_config)
    assert answer is not None, "API 调用失败"
    attach(answer)

    score, hits, misses = score_facts(answer, case["fact_points"])
    attach(f"得分：{score}\n命中：{hits}\n未命中：{misses}", "评分")

    assert score >= 0.8, f"幻觉风险（{score} < 0.8），未命中：{misses}"