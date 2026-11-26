"""Freeze environment-processing shards for direct Studio SDK submission."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyromind_common.platform.credentials import runtime_environment_prefix
from pyromind_common.platform.paths import (
    to_agent_storage_path,
    to_storage_api_path,
)
from pyromind_common.platform.preparation import StagingDirectory
from pyromind_common.platform.scene_core import build_edp_workflow


SHARDS_INDEX_MAX_BYTES = 4 * 1024 * 1024
_LLM_NAMES = ("LLM_BASE_URL", "LLM_AUTH_TOKEN", "LLM_MODEL")
_RUN_DIR_TOKEN = "{edp_run_dir}"


class EdpSubmitOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: str | None = None
    shards: str | None = None
    shard_offset: int = Field(default=0, ge=0)
    shard_count: int = Field(default=1, ge=1, le=5)
    profile_name: str = Field(
        default="tmax-validation", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
    )
    run_id: UUID | None = None
    limit: int | None = Field(default=None, ge=1)
    dedup_by_image: bool = False
    cpu: int = Field(default=4, ge=1, le=64)
    memory: int = Field(default=16, ge=1, le=256)

    @model_validator(mode="after")
    def manifest_source(self) -> EdpSubmitOptions:
        if (self.manifest is None) == (self.shards is None):
            raise ValueError("Set exactly one of manifest or shards")
        return self


class _ShardIndex(BaseModel):
    shards: list[str] = Field(min_length=1)


class _ProfileStep(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class _Profile(BaseModel):
    steps: list[_ProfileStep] = Field(min_length=1)


class _FrozenInputs(BaseModel):
    run_id: UUID
    manifest: str
    profile_name: str
    assets: dict[str, str]


@dataclass(frozen=True)
class PreparedEdpShard:
    workflow: dict[str, Any]
    manifest: str
    run_id: UUID
    output_dir: str
    resumed: bool


def select_edp_manifests(
    options: EdpSubmitOptions, *, shard_index: bytes | None = None
) -> tuple[str, ...]:
    """Select at most five shard paths from a caller-fetched Storage index."""
    if options.manifest is not None:
        if shard_index is not None:
            raise ValueError("shard_index is only valid with shards")
        return (to_storage_api_path(options.manifest, field="manifest"),)
    assert options.shards is not None
    to_storage_api_path(options.shards, field="shards")
    if shard_index is None:
        raise ValueError("Fetch the shards index before preparing submissions")
    if len(shard_index) > SHARDS_INDEX_MAX_BYTES:
        raise ValueError("The shards index exceeds the 4 MiB limit")
    index = _ShardIndex.model_validate_json(shard_index)
    selected = index.shards[
        options.shard_offset : options.shard_offset + options.shard_count
    ]
    if not selected:
        raise ValueError("shard_offset and shard_count select no shards")
    paths = tuple(to_storage_api_path(path, field="shard") for path in selected)
    if len(set(paths)) != len(paths):
        raise ValueError("Selected shard manifests must be unique")
    return paths


def _replace_run_dir(value: Any, pod_path: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_run_dir(item, pod_path) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_run_dir(item, pod_path) for item in value]
    if isinstance(value, str):
        return value.replace(_RUN_DIR_TOKEN, pod_path)
    return value


def _stage_bundle(
    staging: StagingDirectory, runtime_dir: Path, profile_name: str
) -> dict[str, str]:
    profile_path = runtime_dir / "profiles" / f"{profile_name}.json"
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    _Profile.model_validate(document)
    wheel_files = sorted((runtime_dir / "wheels").glob("*.whl"))
    if _RUN_DIR_TOKEN in json.dumps(document):
        if not wheel_files:
            raise ValueError("This profile requires the bundled runtime wheels")
        document = _replace_run_dir(
            document, "/target-workspace" + staging.storage_path()
        )
    staged = [staging.freeze(runtime_dir / "sandbox_runner.py")]
    staged.append(
        staging.write(
            f"{profile_name}.json",
            (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode(),
        )
    )
    runtime = runtime_dir / "pod_runtime"
    if not (runtime / "pod_requirements.txt").is_file():
        raise ValueError("The environment-processing pod runtime is missing")
    for source in sorted(runtime.rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts:
            staged.append(staging.freeze(source, str(source.relative_to(runtime))))
    if _RUN_DIR_TOKEN in profile_path.read_text(encoding="utf-8"):
        staged.extend(
            staging.freeze(wheel, f"wheels/{wheel.name}") for wheel in wheel_files
        )
    return {
        str(path.relative_to(staging.local.resolve())): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in staged
    }


def _worker_command(
    options: EdpSubmitOptions,
    *,
    staging: StagingDirectory,
    manifest: str,
    profile: _Profile,
    platform_env: str,
    cluster: str,
) -> str:
    pod_output = "/target-workspace" + staging.storage_path()
    profile_text = profile.model_dump_json()
    solver = any(step.name in {"install_pi", "run_pi"} for step in profile.steps)
    llm_names = tuple(
        name for name in _LLM_NAMES if solver or "{secret:" + name + "}" in profile_text
    )
    if solver and options.dedup_by_image:
        raise ValueError(
            "Image deduplication cannot be used with agent-solving profiles"
        )
    prefix = runtime_environment_prefix(
        ("PYROMIND_API_KEY", "PYROMIND_BASE_URL", *llm_names)
    )
    venv = "/tmp/edp-venv"
    setup = [
        f"python3 -m venv {venv}",
        (
            f"{venv}/bin/pip install --quiet --use-deprecated=legacy-resolver "
            f"-r {shlex.quote(pod_output + '/pod_requirements.txt')}"
            " > /tmp/edp-install.log 2>&1"
            " || { echo 'runner dependencies install failed'; "
            "tail -n 100 /tmp/edp-install.log; exit 1; }"
        ),
        f"mkdir -p {shlex.quote(pod_output + '/run')}",
        f"test -f {shlex.quote(pod_output + '/sandbox_runner.py')}",
        f"test -f {shlex.quote(pod_output + '/' + options.profile_name + '.json')}",
        f"test -f {shlex.quote('/target-workspace' + manifest)}",
    ]
    arguments = [
        "--profile " + shlex.quote(pod_output + "/" + options.profile_name + ".json"),
        "--manifest " + shlex.quote("/target-workspace" + manifest),
        "--output-dir " + shlex.quote(pod_output + "/run"),
        "--env " + shlex.quote(platform_env),
        "--cluster " + shlex.quote(cluster),
        '--api-key "$PYROMIND_API_KEY"',
        *(f'--set {name}="${name}"' for name in llm_names),
    ]
    if options.limit is not None:
        arguments.append(f"--limit {options.limit}")
    if options.dedup_by_image:
        arguments.append("--dedup-by-image")
    setup.append(
        prefix
        + " "
        + f"PYTHONPATH={shlex.quote(pod_output)} "
        + f"{venv}/bin/python {shlex.quote(pod_output + '/sandbox_runner.py')} "
        + " ".join(arguments)
    )
    return " && ".join(setup)


def prepare_edp_shard(
    options: EdpSubmitOptions,
    *,
    manifest: str,
    staging: StagingDirectory,
    runtime_dir: Path,
    run_id: UUID,
    platform_env: str,
    cluster: str,
) -> PreparedEdpShard:
    """Freeze a shard runner; submit the resource-validated graph using the SDK.

    The caller publishes staging files before submission. Each run directory stays
    beside its shard manifest, and the frozen runtime is reused on resume.
    """
    manifest = to_storage_api_path(manifest, field="manifest")
    expected_output = str(PurePosixPath(manifest).parent / str(run_id))
    if staging.storage_path() != expected_output:
        raise ValueError("Shard staging must be beside its manifest, named by run_id")
    resumed = options.run_id is not None
    if resumed and options.run_id != run_id:
        raise ValueError("run_id must match the resumed shard run")
    if resumed:
        frozen = _FrozenInputs.model_validate_json(
            (staging.local / "edp-inputs.json").read_bytes()
        )
        if (frozen.run_id, frozen.manifest, frozen.profile_name) != (
            run_id,
            manifest,
            options.profile_name,
        ):
            raise ValueError("Resume must use the original manifest and profile")
        for name, digest in frozen.assets.items():
            staging.storage_path(name)
            source = (staging.local / name).resolve(strict=True)
            source.relative_to(staging.local.resolve(strict=True))
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen environment-processing runtime changed")
    else:
        assets = _stage_bundle(staging, runtime_dir, options.profile_name)
        frozen = _FrozenInputs(
            run_id=run_id,
            manifest=manifest,
            profile_name=options.profile_name,
            assets=assets,
        )
        staging.write(
            "edp-inputs.json", frozen.model_dump_json(indent=2).encode() + b"\n"
        )
    profile = _Profile.model_validate_json(
        (staging.local / f"{options.profile_name}.json").read_bytes()
    )
    command = _worker_command(
        options,
        staging=staging,
        manifest=manifest,
        profile=profile,
        platform_env=platform_env,
        cluster=cluster,
    )
    return PreparedEdpShard(
        workflow=build_edp_workflow(
            run_id=uuid4(), command=command, cpu=options.cpu, memory=options.memory
        ),
        manifest=to_agent_storage_path(manifest),
        run_id=run_id,
        output_dir=to_agent_storage_path(staging.storage_path()),
        resumed=resumed,
    )
