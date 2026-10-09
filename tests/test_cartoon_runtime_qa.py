from app.agents.cartoon_runtime_qa import CartoonRuntimeQA, RuntimeMetrics, score_metrics


def test_runtime_score_rewards_all_delivery_gates():
    perfect = RuntimeMetrics(True, True, True, True, True, 0.0, 0.0, 900.0)
    weak = RuntimeMetrics(True, False, True, True, True, 0.04, 0.40, 20.0)
    assert score_metrics(perfect) == 1.0
    assert score_metrics(weak) < 0.9
    assert perfect.passed is True
    assert weak.passed is False


def test_runtime_inspector_has_expected_delivery_contract():
    qa = CartoonRuntimeQA()
    assert qa.expected_width == 1080
    assert qa.expected_height == 1920
    assert qa.expected_fps == 24.0
