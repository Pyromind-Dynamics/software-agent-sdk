"""``ProcessingProfile`` — declarative spec for environment-bound data processing.

A vertical scenario (e.g. tmax terminal-task validation) is expressed as an
ordered list of sandbox primitives plus a verdict rule and an output spec.
Control flow — per-record looping, image-dedup caching, resume, sandbox
cleanup — is intentionally *not* part of the profile; it lives in the frozen
runtime (``data-processing`` skill's ``scripts/edp/sandbox_runner.py``) that
interprets the profile, so a mis-authored profile cannot silently corrupt a
batch run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


PROCESSING_PROFILE_SCHEMA_VERSION = 1

StepName = Literal[
    "create_sandbox",
    "probe",
    "write_file",
    "exec",
    "install_pi",
    "run_pi",
    "delete_sandbox",
]


class ProcessingStep(BaseModel):
    """One declarative sandbox primitive executed by the frozen runtime.

    ``params`` string values may contain ``{<field>}`` placeholders; the
    runtime substitutes them from the current manifest record before the step
    runs (e.g. ``{image}``, ``{workdir}``).
    """

    model_config = ConfigDict(extra="forbid")

    name: StepName = Field(description="Runtime primitive to execute.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Primitive parameters with optional '{field}' placeholders.",
    )


class VerdictRule(BaseModel):
    """How the runtime classifies a per-record run as usable/error."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["exit_code", "reward_file"] = Field(
        default="exit_code",
        description=(
            "Verdict strategy: 'exit_code' matches the verifier exit code; "
            "'reward_file' prefers a 0-1 reward float written by the verifier "
            "(falling back to exit_code when the file is missing/unparsable)."
        ),
    )
    success_codes: list[int] = Field(
        default_factory=lambda: [0],
        description="Verifier exit codes that classify the record as usable.",
    )
    reward_path: str = Field(
        default="/logs/verifier/reward.txt",
        description=(
            "In-container path of the reward file read when kind='reward_file'."
        ),
    )


class OutputSpec(BaseModel):
    """Artifact produced by a profile run."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(
        default="verdicts.jsonl",
        description="Per-record verdict file (JSONL) written by the runtime.",
    )


class ProcessingProfile(BaseModel):
    """Declarative vertical-scenario profile interpreted by the frozen runtime."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=PROCESSING_PROFILE_SCHEMA_VERSION, ge=1)
    name: str = Field(
        min_length=1,
        description="Scenario identifier, e.g. 'tmax-validation'.",
    )
    description: str = Field(
        default="",
        description="Human-facing summary used for scenario routing.",
    )
    steps: list[ProcessingStep] = Field(
        min_length=1,
        description="Ordered sandbox primitives; control flow stays in the runtime.",
    )
    verdict: VerdictRule = Field(
        description="Rule mapping the run outcome to usable/error.",
    )
    output: OutputSpec = Field(default_factory=OutputSpec)
