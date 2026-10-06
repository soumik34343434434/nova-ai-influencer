from app.agents.cartoon_content import CartoonEpisode, pick_episode


def test_pick_episode_is_deterministic():
    first = pick_episode("smart kitchen")
    second = pick_episode("smart kitchen")
    assert isinstance(first, CartoonEpisode)
    assert first == second
    assert first.character in {"PIP", "MOMO", "BUBU"}
    assert len(first.beats) == 4


def test_cartoon_content_is_family_friendly():
    episode = pick_episode("homework robot")
    text = (episode.title + " " + episode.hook + " " +
            " ".join(b["visual"] for b in episode.beats)).lower()
    assert "celebrity" not in text
    assert "lookalike" not in text


def test_homework_robot_social_caption_has_specific_hook_payoff():
    episode = pick_episode("homework robot")
    assert "ONE answer" in episode.hook
    assert "textbook" in episode.caption.lower()
