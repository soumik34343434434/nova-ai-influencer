from app.agents.content_evaluator import ContentEvaluator


def test_evaluator_passes_original_nova_draft():
    draft = {
        "title": "NOVA meets the new AI coworker",
        "hook": "Your company said AI would remove meetings. It invented three more.",
        "script": "NOVA stares at the camera. The AI agent finished the work. Then the calendar bot booked another sync to discuss the finished work.",
    }
    result = ContentEvaluator().evaluate(draft)
    assert result.passed
    assert result.score >= 0.78


def test_evaluator_blocks_reuse_signal():
    draft = {
        "title": "Celebrity repost",
        "hook": "Lookalike celebrity content",
        "script": "Repost this private person video",
    }
    result = ContentEvaluator().evaluate(draft)
    assert not result.passed
    assert "forbidden_or_reuse_signal" in result.reasons
