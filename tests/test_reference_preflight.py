from app.agents.reference_preflight import (
    assert_reference_preflight,
    build_reference_manifest,
    required_reference_props,
    validate_reference_preflight,
)


def test_phone_action_requires_phone_reference():
    assert required_reference_props("checks_phone") == ("phone",)


def test_verified_manifest_passes_phone_preflight():
    manifest = build_reference_manifest(
        "MOMO",
        "office",
        ["phone", "paper"],
        asset_id="momo-phone-ref",
    )
    result = validate_reference_preflight(
        "checks_phone",
        "MOMO",
        manifest,
        expected_scene="office",
    )
    assert result["passed"] is True
    assert result["missing_props"] == []


def test_reference_scene_mismatch_fails():
    manifest = build_reference_manifest("MOMO", "metro", ["phone"])
    result = validate_reference_preflight(
        "checks_phone",
        "MOMO",
        manifest,
        expected_scene="office",
    )
    assert result["passed"] is False
    assert "reference_scene_mismatch" in result["errors"]


def test_phone_action_fails_without_phone():
    manifest = build_reference_manifest(
        "MOMO",
        "office",
        ["paper"],
        asset_id="momo-no-phone-ref",
    )
    result = validate_reference_preflight("checks_phone", "MOMO", manifest)
    assert result["passed"] is False
    assert "reference_missing_required_prop_phone" in result["errors"]


def test_assert_preflight_is_blocking_for_missing_action_target():
    manifest = build_reference_manifest("MOMO", "office", ["paper"])
    try:
        assert_reference_preflight("checks_phone", "MOMO", manifest)
    except ValueError as exc:
        assert "reference_missing_required_prop_phone" in str(exc)
    else:
        raise AssertionError("Expected missing phone reference to block execution")
