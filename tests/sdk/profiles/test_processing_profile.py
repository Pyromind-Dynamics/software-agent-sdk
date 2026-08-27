"""Tests for the ``ProcessingProfile`` declarative model."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from openhands.sdk.profiles import (
    PROCESSING_PROFILE_SCHEMA_VERSION,
    ProcessingProfile,
    ProcessingStep,
    VerdictRule,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TMAX_SEED = (
    _REPO_ROOT
    / ".agents"
    / "skills"
    / "environment-data-processing"
    / "profiles"
    / "tmax-validation.json"
)


def _minimal_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "demo",
        "steps": [
            {"name": "create_sandbox", "params": {"image": "img:1"}},
            {"name": "exec", "params": {"command": "true"}},
        ],
        "verdict": {"kind": "exit_code"},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Construction + defaults
# ---------------------------------------------------------------------------


def test_minimal_profile_round_trips_with_defaults() -> None:
    profile = ProcessingProfile.model_validate(_minimal_payload())

    assert profile.schema_version == PROCESSING_PROFILE_SCHEMA_VERSION == 1
    assert profile.name == "demo"
    assert profile.description == ""
    assert [step.name for step in profile.steps] == ["create_sandbox", "exec"]
    assert profile.steps[0].params == {"image": "img:1"}
    assert profile.verdict.kind == "exit_code"
    assert profile.verdict.success_codes == [0]
    assert profile.output.filename == "verdicts.jsonl"

    reloaded = ProcessingProfile.model_validate(profile.model_dump(mode="json"))
    assert reloaded == profile


def test_step_params_default_to_empty_dict() -> None:
    step = ProcessingStep(name="delete_sandbox")
    assert step.params == {}


def test_verdict_rule_defaults() -> None:
    rule = VerdictRule()
    assert rule.kind == "exit_code"
    assert rule.success_codes == [0]


def test_custom_success_codes() -> None:
    profile = ProcessingProfile.model_validate(
        _minimal_payload(verdict={"kind": "exit_code", "success_codes": [0, 42]})
    )
    assert profile.verdict.success_codes == [0, 42]


def test_tmax_seed_profile_loads() -> None:
    profile = ProcessingProfile.model_validate_json(
        _TMAX_SEED.read_text(encoding="utf-8")
    )
    assert profile.name == "tmax-validation"
    assert [step.name for step in profile.steps] == [
        "create_sandbox",
        "probe",
        "write_file",
        "install_pi",
        "run_pi",
        "write_file",
        "exec",
        "delete_sandbox",
    ]
    assert profile.steps[0].params["image"] == "{image}"
    assert profile.verdict.kind == "reward_file"
    assert profile.verdict.reward_path == "/logs/verifier/reward.txt"
    assert profile.verdict.success_codes == [0]


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_unknown_step_name_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessingProfile.model_validate(
            _minimal_payload(steps=[{"name": "reboot_universe"}])
        )


def test_empty_steps_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessingProfile.model_validate(_minimal_payload(steps=[]))


def test_empty_name_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessingProfile.model_validate(_minimal_payload(name=""))


def test_missing_verdict_rejected() -> None:
    payload = _minimal_payload()
    del payload["verdict"]
    with pytest.raises(ValidationError):
        ProcessingProfile.model_validate(payload)


@pytest.mark.parametrize(
    ("payload_factory", "extra"),
    [
        (lambda extra: _minimal_payload(**extra), {"bogus": 1}),
        (
            lambda extra: _minimal_payload(
                steps=[{"name": "exec", "params": {}, **extra}]
            ),
            {"retries": 3},
        ),
        (
            lambda extra: _minimal_payload(verdict={"kind": "exit_code", **extra}),
            {"threshold": 0.5},
        ),
        (
            lambda extra: _minimal_payload(output={"filename": "v.jsonl", **extra}),
            {"format": "csv"},
        ),
    ],
    ids=["profile", "step", "verdict", "output"],
)
def test_extra_fields_forbidden(payload_factory: Any, extra: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ProcessingProfile.model_validate(payload_factory(extra))


def test_unknown_verdict_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessingProfile.model_validate(
            _minimal_payload(verdict={"kind": "llm_judge"})
        )
