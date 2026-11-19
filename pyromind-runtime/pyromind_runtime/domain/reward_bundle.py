from __future__ import annotations

import ast
import hashlib
import re
from typing import Literal

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.storage_paths import from_storage_api_path
from pyromind_runtime.domain.versions import SnapshotResult, canonical, fingerprint


type TrainingMethod = Literal["sft", "rule_rl", "rubric_rl"]
type JudgeKind = Literal["code", "llm", "hybrid"]

START_REWARD_REVIEW = "request_reward_bundle_review"
GET_PROJECT_REWARD = "get_project_reward"
REWARD_BUNDLE_KEY = "reward_bundle"
APPROVE_REWARD = "approve"
REWARD_QUESTION_ID = "reward_bundle"
REWARD_WORKDIR = "public_data/rubric_work"
MAX_REWARD_FILE_BYTES = 64 * 1024
REWARD_ENTRIES = ("studio_reward", "parse_record", "score_case")


class StartRewardReview(ContractModel):
    """Freeze a workspace Reward bundle and ask the user to confirm it.

    Pass the returned approval ID to set_preparation_plan.
    """

    candidate_directory: str = Field(
        description="Candidate directory relative to public_data/rubric_work."
    )
    judge_kind: JudgeKind
    requirements: str = Field(
        min_length=1,
        max_length=2000,
        description="Runtime dependencies and configuration names. No values or URLs.",
    )


class Criterion(ContractModel):
    criterion: str = Field(min_length=1, max_length=800)
    id: str | None = None
    weight: float = Field(default=1, gt=0, allow_inf_nan=False)
    hard_constraint: bool = False

    @model_validator(mode="after")
    def validate_identity(self):
        label = self.criterion.strip()
        if not label:
            raise ValueError("rubric criterion cannot be blank")
        if self.id is not None and self.id != self.stable_id:
            raise ValueError("criterion id must match its text")
        return self

    @property
    def stable_id(self) -> str:
        return "criterion-" + fingerprint(self.criterion.strip())[:16]


class Rubric(ContractModel):
    items: tuple[Criterion, ...] = Field(min_length=1, max_length=24)
    aggregation: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_items(self):
        ids = {item.stable_id for item in self.items}
        if len(ids) != len(self.items) or not self.aggregation.strip():
            raise ValueError(
                "rubric requires distinct criteria and an aggregation rule"
            )
        return self


class CalibrationEvidence(ContractModel):
    case_id: str = Field(min_length=1, max_length=128)
    score: float = Field(ge=0, le=1)
    summary: str = Field(min_length=1, max_length=600)


class RewardManifest(ContractModel):
    schema_version: Literal[1] = 1
    judge_kind: JudgeKind
    requirements: str
    entries: tuple[str, ...] = REWARD_ENTRIES
    files: dict[str, str]
    calibration: tuple[CalibrationEvidence, ...]


class RewardBundle(ContractModel):
    approval_id: str
    manifest_digest: str
    snapshot: SnapshotResult
    cluster: str
    verified_studio_entry: bool = Field(
        default=False, exclude_if=lambda value: not value
    )

    @property
    def studio_entry(self) -> str:
        directory = from_storage_api_path(self.snapshot.storage_ref)
        filename = "studio_entry.py" if self.verified_studio_entry else "reward.py"
        return f"{directory}/{filename}:studio_reward"


class RewardReview(ContractModel):
    request_id: str
    bundle: RewardBundle


def freeze_reward_files(
    request: StartRewardReview, files: dict[str, str]
) -> tuple[dict[str, str], RewardManifest, Rubric, str]:
    for text in (*files.values(), request.requirements):
        for url in re.findall(r"""https?://[^\s<>"']+""", text):
            validate_persisted_payload(url)
    task_prompt = files["task_prompt.md"]
    if not task_prompt.strip() or len(task_prompt) > 16_000:
        raise ValueError("task_prompt.md must contain up to 16,000 characters")
    try:
        rubric = Rubric.model_validate_json(files["rubric.json"])
    except ValidationError as exc:
        details = "; ".join(
            ".".join(map(str, item["loc"])) + ": " + item["msg"]
            for item in exc.errors(include_input=False, include_url=False)[:6]
        )
        raise ValueError(
            f"rubric.json: {details}. Draft items use criterion, weight and "
            "hard_constraint. Omit id to let Runtime generate it."
        ) from None
    files["rubric.json"] = canonical(
        {
            "items": [
                {**item.model_dump(mode="json"), "id": item.stable_id}
                for item in rubric.items
            ],
            "aggregation": rubric.aggregation,
        }
    )
    try:
        script = ast.parse(files["reward.py"], filename="reward.py")
        compile(script, filename="reward.py", mode="exec")
    except SyntaxError as exc:
        raise ValueError("reward.py has invalid Python syntax") from exc
    entries = {node.name for node in script.body if isinstance(node, ast.FunctionDef)}
    if not set(REWARD_ENTRIES) <= entries:
        raise ValueError(
            "reward.py must define studio_reward, parse_record and score_case"
        )
    evidence_text = files.pop("calibration_review.json")
    evidence = TypeAdapter(
        tuple[CalibrationEvidence, ...],
    ).validate_json(evidence_text)
    if not 1 <= len(evidence) <= 12 or len({case.case_id for case in evidence}) != len(
        evidence
    ):
        raise ValueError(
            "calibration_review.json requires 1–12 distinct reviewed cases"
        )
    if request.judge_kind != "code":
        examples = TypeAdapter(list[JsonObject]).validate_json(
            files["calibration.json"]
        )
        validate_persisted_payload(examples)
    validate_persisted_payload(rubric.model_dump(mode="json"))
    files["studio_entry.py"] = studio_entry_source(
        {
            name: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for name, value in files.items()
        }
    )
    manifest = RewardManifest(
        judge_kind=request.judge_kind,
        requirements=request.requirements,
        files={
            name: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for name, value in files.items()
        },
        calibration=evidence,
    )
    validate_persisted_payload(manifest.model_dump(mode="json"))
    manifest_text = canonical(manifest.model_dump(mode="json"))
    return {**files, "manifest.json": manifest_text}, manifest, rubric, task_prompt


def studio_entry_source(members: dict[str, str]) -> str:
    # Studio loads this shim through its normal Custom Entry hook.
    return f"""\
import hashlib
import json
import types
from pathlib import Path

_DIRECTORY = Path(__file__).resolve().parent
_MEMBERS = {members!r}


def _load():
    verified = {{}}
    for name, expected in _MEMBERS.items():
        data = (_DIRECTORY / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("frozen Reward member digest changed: " + name)
        verified[name] = data
    reward = types.ModuleType("project_reward")
    reward.__file__ = str(_DIRECTORY / "reward.py")
    exec(compile(verified["reward.py"], reward.__file__, "exec"), reward.__dict__)
    return reward.studio_reward


_MANIFEST = (_DIRECTORY / "manifest.json").read_bytes()
_DECLARED = json.loads(_MANIFEST)["files"]
if any(_DECLARED.get(name) != digest for name, digest in _MEMBERS.items()):
    raise ValueError("frozen Reward manifest does not match the Studio entry")
_ENTRY_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if _ENTRY_DIGEST != _DECLARED.get("studio_entry.py"):
    raise ValueError("frozen Studio Reward entry digest changed")
_SCORE = _load()
_REPORTED = False


def studio_reward(completions, *args, **kwargs):
    global _REPORTED
    scores = _SCORE(completions, *args, **kwargs)
    if not isinstance(scores, list) or len(scores) != len(completions):
        raise ValueError("studio_reward must return one score per completion")
    if not _REPORTED:
        print("frozen_reward manifest_sha256=" + hashlib.sha256(_MANIFEST).hexdigest()
              + " scored_batch=" + str(len(completions)), flush=True)
        _REPORTED = True
    return scores
"""
