from app.experiment_system import assign_variant, plan_experiment, record_observation


def make_plan(path, **overrides):
    args = dict(dimension="hook", variants=["question", "statement"],
                hypothesis="Questions improve watch time", success_metric="retention_pct",
                min_observations=3, max_observations=5, stop_threshold=0.2, path=path)
    args.update(overrides)
    return plan_experiment(**args)


def test_variant_assignment_is_stable_and_valid(tmp_path):
    plan = make_plan(tmp_path / "registry.json")
    assert assign_variant(plan, "run-1") == assign_variant(plan, "run-1")
    assert assign_variant(plan, "run-1")["variant_id"] in {v["variant_id"] for v in plan["variants"]}


def test_duplicate_definitions_reuse_experiment(tmp_path):
    path = tmp_path / "registry.json"
    one, two = make_plan(path), make_plan(path)
    assert one["experiment_id"] == two["experiment_id"]
    assert len(__import__("json").loads(path.read_text())["experiments"]) == 1


def test_minimum_sample_prevents_early_stop_and_stop_rule_applies(tmp_path):
    path = tmp_path / "registry.json"
    plan = make_plan(path)
    first, second = plan["variants"]
    result = record_observation(plan["experiment_id"], variant_id=first["variant_id"], metric_value=0.9, path=path)
    assert result["status"] == "running"
    record_observation(plan["experiment_id"], variant_id=first["variant_id"], metric_value=0.9, path=path)
    result = record_observation(plan["experiment_id"], variant_id=second["variant_id"], metric_value=0.1, path=path)
    assert result["status"] == "stopped"


def test_observation_keeps_content_analytics_lineage(tmp_path):
    path = tmp_path / "registry.json"
    plan = make_plan(path)
    variant = plan["variants"][0]["variant_id"]
    result = record_observation(plan["experiment_id"], variant_id=variant, metric_value=0.4,
                               content_id="post-4", run_id="run-2", analytics_id="snapshot-8", path=path)
    observation = result["outcomes"][0]
    assert observation == {"variant_id": variant, "metric_value": 0.4, "content_id": "post-4",
                           "run_id": "run-2", "analytics_id": "snapshot-8"}
