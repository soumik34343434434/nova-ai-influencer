from app.agents.content_agent import generate_content

def test_mock_content_is_original():
    item = generate_content("AI productivity")
    assert item["originality_score"] >= 0.8
    assert item["risk_score"] <= 0.2
    assert item["title"]
    assert item["script"]
