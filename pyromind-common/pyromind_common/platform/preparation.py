"""Freeze business inputs and build Studio graphs without executing a Tool."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel

from pyromind_common.platform.credentials import (
    dataflow_environment,
    dataset_key_parameter,
    runtime_environment_prefix,
)
from pyromind_common.platform.paths import to_agent_storage_path, to_storage_api_path
from pyromind_common.platform.recipe import (
    RECIPE_INPUT,
    RECIPE_OUTPUT,
    compose_recipe,
    dataset_public_profile,
    recipe_asset,
)
from pyromind_common.platform.scene_core import (
    AGGREGATE_FILENAME,
    CLEANING_RUNTIME_FILENAMES,
    DATAFLOW_RUNTIME_FILENAMES,
    MAX_CLEANING_SCRIPT_BYTES,
    RENDER_FILENAME,
    RENDER_REQUIREMENTS,
    RENDER_TEMPLATE_FILENAME,
    SFT_CONVERTER_FILENAME,
    SLIME_CONVERTER_FILENAME,
    _build_cleaning_command,
    _build_cleaning_workflow,
    _build_dataflow_command,
    _build_dataflow_workflow,
    _check_column_refs,
    _collect_column_refs,
    build_aggregate_command,
    build_edp_workflow,
    build_render_command,
    runtime_bundle_fingerprint,
    runtime_public_names,
    validate_cleaning_script,
    validate_managed_image_pipeline,
    validate_pipeline_topology,
)
from pyromind_common.platform.scene_options import (
    AggregateOptions,
    BatchLabelSource,
    CleaningOptions,
    OutputSchema,
    PipelineOptions,
    RenderOptions,
    ReuseAssessment,
)


@dataclass(frozen=True)
class LocalFiles:
    root: Path

    def resolve(self, path: str) -> tuple[Path, str]:
        root = self.root.resolve(strict=True)
        candidate = (root / path).resolve(strict=True)
        relative = candidate.relative_to(root)
        if not candidate.is_file():
            raise ValueError(f"Not a file: {path}")
        return candidate, str(relative)

    def read_text(self, path: str) -> str:
        return self.resolve(path)[0].read_text(encoding="utf-8")


@dataclass(frozen=True)
class StagingDirectory:
    """A local directory paired with its eventual platform Storage location.

    On a Storage mount the caller can use the actual mounted directory. On an
    OS backend the caller must publish the frozen files before submitting.
    """

    local: Path
    storage: str

    @classmethod
    def mounted(cls, path: str | Path) -> Self:
        """Use one directory on the mounted /workspace disk for frozen assets."""
        storage = to_agent_storage_path(str(path))
        if PurePosixPath(str(path)) != PurePosixPath(storage):
            raise ValueError("Mounted staging must use an absolute /workspace/... path")
        return cls(local=Path(storage), storage=storage)

    def storage_path(self, name: str = "") -> str:
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("Frozen filename must stay inside the staging directory")
        return str(PurePosixPath(to_storage_api_path(self.storage), name))

    def write(self, name: str, content: bytes) -> Path:
        self.storage_path(name)
        self.local.mkdir(parents=True, exist_ok=True)
        root = self.local.resolve(strict=True)
        target = (root / name).resolve()
        target.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(
                    f"Frozen input changed: {name} at {target}; "
                    "use a new local staging directory as well as a new Storage path"
                )
            return target
        try:
            with target.open("xb") as destination:
                destination.write(content)
                destination.flush()
                os.fsync(destination.fileno())
        except FileExistsError:
            if target.read_bytes() != content:
                raise ValueError(
                    f"Frozen input changed concurrently: {name} at {target}"
                ) from None
        return target

    def freeze(self, source: Path, name: str | None = None) -> Path:
        return self.write(name or source.name, source.read_bytes())


def prepare_cleaning(
    options: CleaningOptions,
    *,
    files: LocalFiles,
    staging: StagingDirectory,
    runtime_dir: Path,
    run_id: UUID,
) -> dict[str, Any]:
    """Preflight and freeze a cleaning script; never create or register a task."""
    if options.resume_run_id is not None:
        if options.resume_run_id != run_id:
            raise ValueError("resume_run_id must match the original run")
        script = staging.local / "clean_script.py"
    elif options.script_path is not None:
        script = files.resolve(options.script_path)[0]
    else:
        raise ValueError("A script is required for a new cleaning run")
    if script.suffix != ".py" or script.stat().st_size > MAX_CLEANING_SCRIPT_BYTES:
        raise ValueError("Cleaning script must be a Python file within the size limit")
    validate_cleaning_script(
        script.read_text(encoding="utf-8-sig"), runtime_dir / "cleaning_utils.py"
    )
    for name in CLEANING_RUNTIME_FILENAMES:
        staging.freeze(runtime_dir / name)
    staging.freeze(script, "clean_script.py")
    staging.write(
        "cleaning-input.json",
        json.dumps(
            {
                "run_id": str(run_id),
                "input_path": to_storage_api_path(options.input_path),
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "runtime_sha256": runtime_bundle_fingerprint(
                    runtime_dir, CLEANING_RUNTIME_FILENAMES
                ),
            },
            sort_keys=True,
        ).encode(),
    )
    output = staging.storage_path()
    command = _build_cleaning_command(
        input_path=to_storage_api_path(options.input_path),
        script_path=staging.storage_path("clean_script.py"),
        output_root=str(PurePosixPath(output).parent),
        output_dir=output,
        limit=options.limit,
        resumed=options.resume_run_id is not None,
        frozen_script_ready=True,
    )
    return _build_cleaning_workflow(options, run_id, command)


class _PipelineInputs(BaseModel):
    run_id: UUID
    execution_revision: int
    input_path: str
    script_file: str
    script_sha256: str
    runtime_directory: str
    runtime_sha256: str
    model_profile: Literal["none", "text", "vision"]
    model_sha256: str
    output_schema: OutputSchema | None
    prompt_fingerprint: str | None
    support_file: str | None
    support_sha256: str | None
    recipe_inputs: dict[str, str]
    fixed_validation_path: str | None
    batch_label_source: BatchLabelSource | None
    reuse_assessment: ReuseAssessment | None


def _pipeline_prior(staging: StagingDirectory) -> _PipelineInputs:
    revisions = sorted((staging.local / "pipeline-revisions").glob("*.json"))
    manifest = revisions[-1] if revisions else staging.local / "pipeline-inputs.json"
    return _PipelineInputs.model_validate_json(manifest.read_text(encoding="utf-8"))


def _validate_pipeline_resume(prior: _PipelineInputs, current: _PipelineInputs) -> None:
    if current.run_id != prior.run_id or current.input_path != prior.input_path:
        raise ValueError("Frozen input changed: resume must use the original run/input")
    if prior.recipe_inputs:
        raise ValueError("Replay a Dataset recipe with full mode and a fresh directory")
    if current.output_schema != prior.output_schema and "structured" in {
        current.output_schema,
        prior.output_schema,
    }:
        raise ValueError("Cannot change structured output contract on resume")
    dimensions = {
        "pipeline": (prior.script_sha256, current.script_sha256),
        "prompt": (prior.prompt_fingerprint, current.prompt_fingerprint),
        "model": (
            (prior.model_profile, prior.model_sha256),
            (current.model_profile, current.model_sha256),
        ),
        "schema": (
            (prior.output_schema, prior.support_sha256),
            (current.output_schema, current.support_sha256),
        ),
        "runtime": (prior.runtime_sha256, current.runtime_sha256),
    }
    changed = {name for name, (before, after) in dimensions.items() if before != after}
    assessed = (
        set(current.reuse_assessment.changed_dimensions)
        if current.reuse_assessment is not None
        else set()
    )
    if missing := changed - assessed:
        raise ValueError(
            "Resume changes require reuse_assessment for: " + ", ".join(sorted(missing))
        )


def prepare_pipeline(
    options: PipelineOptions,
    *,
    files: LocalFiles,
    staging: StagingDirectory,
    runtime_dir: Path,
    run_id: UUID,
) -> dict[str, Any]:
    """Freeze a pipeline revision; bind worker credentials only during submission."""
    resumed = options.mode == "resume"
    if resumed and options.resume_run_id != run_id:
        raise ValueError("resume_run_id must match the original run")
    prior = _pipeline_prior(staging) if resumed else None
    revision = prior.execution_revision + 1 if prior is not None else 1
    model_profile = options.model_profile or (
        prior.model_profile if prior is not None else "text"
    )
    has_recipe = options.sft_export_script_path is not None
    llm_env = dataflow_environment(model_profile) if model_profile != "none" else {}
    uses_gateway = bool(llm_env.get("DF_API_KEY"))
    if has_recipe:
        llm_env = dataset_public_profile(llm_env)
    output_schema = options.output_schema or (
        prior.output_schema if prior is not None else None
    )
    prompt_fingerprint = (
        options.prompt_fingerprint
        if options.prompt_fingerprint is not None
        else prior.prompt_fingerprint
        if prior is not None
        else None
    )
    effective_runtime = runtime_dir
    if options.script_path is None:
        if prior is None:
            raise ValueError("script_path is required for a new pipeline")
        script = staging.local / prior.script_file
        effective_runtime = staging.local / prior.runtime_directory
    else:
        script = files.resolve(options.script_path)[0]
    if script.suffix != ".py":
        raise ValueError("Pipeline script must be a Python file")
    ast.parse(script.read_text(encoding="utf-8-sig"), filename=str(script))
    if output_schema in {"vision", "structured"}:
        validate_managed_image_pipeline(
            script, runtime_public_names(effective_runtime / "image_utils.py")
        )
    runtime_hash = runtime_bundle_fingerprint(
        effective_runtime, DATAFLOW_RUNTIME_FILENAMES
    )
    script_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    script_file = "pipeline.py"
    runtime_directory = "runtime"
    if prior is not None:
        script_file = (
            prior.script_file
            if script_hash == prior.script_sha256
            else f"pipeline-r{revision}.py"
        )
        runtime_directory = (
            prior.runtime_directory
            if runtime_hash == prior.runtime_sha256
            else f"runtime-r{revision}"
        )
    support = None
    if options.support_file_path:
        support = files.resolve(options.support_file_path)[0]
    elif prior is not None and prior.support_file is not None:
        support = staging.local / prior.support_file
    support_hash = None
    support_file = None
    if support is not None:
        json.loads(support.read_text(encoding="utf-8"))
        support_hash = hashlib.sha256(support.read_bytes()).hexdigest()
        support_file = (
            prior.support_file
            if prior is not None and support_hash == prior.support_sha256
            else f"job-spec-r{revision}.json"
            if resumed
            else "job-spec.json"
        )
    fixed_validation_path = (
        to_agent_storage_path(
            options.fixed_validation_path, field="fixed_validation_path"
        )
        if options.fixed_validation_path
        else None
    )
    recipe_inputs = {
        role: hashlib.sha256(files.resolve(path)[0].read_bytes()).hexdigest()
        for role, path in (
            ("prepare", options.prepare_script_path),
            ("export", options.sft_export_script_path),
        )
        if path is not None
    }
    inputs = _PipelineInputs(
        run_id=run_id,
        execution_revision=revision,
        input_path=to_storage_api_path(options.input_path),
        runtime_directory=runtime_directory,
        runtime_sha256=runtime_hash,
        script_file=script_file,
        script_sha256=script_hash,
        model_profile=model_profile,
        model_sha256=hashlib.sha256(
            json.dumps(
                {
                    name: value
                    for name, value in llm_env.items()
                    if name != "DF_API_KEY"
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        output_schema=output_schema,
        prompt_fingerprint=prompt_fingerprint,
        support_file=support_file,
        support_sha256=support_hash,
        recipe_inputs=recipe_inputs,
        fixed_validation_path=fixed_validation_path,
        batch_label_source=options.batch_label_source,
        reuse_assessment=options.reuse_assessment,
    )
    if prior is not None:
        _validate_pipeline_resume(prior, inputs)
    for name in DATAFLOW_RUNTIME_FILENAMES:
        staging.freeze(effective_runtime / name, f"{runtime_directory}/{name}")
    staging.freeze(script, script_file)
    if support is not None and support_file is not None:
        staging.freeze(support, support_file)
    command = _build_dataflow_command(
        input_path=RECIPE_INPUT
        if has_recipe
        else to_storage_api_path(options.input_path),
        output_dir=RECIPE_OUTPUT if has_recipe else staging.storage_path(),
        llm_env={},
        convert_format=options.convert_format,
        frozen_script_name=script_file,
        execution_revision=revision,
        runtime_dir_name=runtime_directory,
        runtime_storage_dir=staging.storage_path(runtime_directory),
        runtime_fingerprint=runtime_hash,
        resumed=resumed,
        output_schema=output_schema,
        reuse_assessment=options.reuse_assessment.model_dump()
        if options.reuse_assessment is not None
        else None,
        support_file_name=support_file,
    )
    if has_recipe:
        assert options.sft_export_script_path is not None
        sources: dict[
            Literal["prepare", "pipeline", "export", "support_file", "verifier"], Path
        ] = {
            "pipeline": script,
            "export": files.resolve(options.sft_export_script_path)[0],
            "verifier": runtime_dir / "verify_training_splits.py",
        }
        if options.prepare_script_path:
            sources["prepare"] = files.resolve(options.prepare_script_path)[0]
        if options.support_file_path:
            sources["support_file"] = files.resolve(options.support_file_path)[0]
        assets = []
        for role, source in sources.items():
            asset = recipe_asset(role, source, staging.storage_path("recipe-assets"))
            relative = PurePosixPath(to_storage_api_path(asset.path)).relative_to(
                staging.storage_path()
            )
            staging.freeze(source, str(relative))
            assets.append(asset)
        recipe, command = compose_recipe(
            input_path=to_storage_api_path(options.input_path),
            output_dir=staging.storage_path(),
            assets=tuple(assets),
            dataflow_command=command,
            llm_env=llm_env,
            fixed_validation_path=fixed_validation_path,
            replay_of_task_id=options.replay_of_task_id,
            batch_label_source=options.batch_label_source,
            prepared_gateway=uses_gateway,
        )
        staging.write(
            "dataset-recipe.json",
            json.dumps(
                recipe.model_dump(), ensure_ascii=False, sort_keys=True
            ).encode(),
        )
    elif model_profile != "none":
        environment = runtime_environment_prefix(
            [name for name, value in llm_env.items() if value], profile=model_profile
        )
        command = environment + "\n" + command
    workflow = _build_dataflow_workflow(options, run_id, command)
    if has_recipe and model_profile != "none" and uses_gateway:
        workflow["nodes"][0]["data"]["config"]["param"] = dataset_key_parameter(
            model_profile, llm_env
        )
    validate_pipeline_topology(workflow)
    staging.write(
        f"pipeline-revisions/{revision:08d}.json"
        if resumed
        else "pipeline-inputs.json",
        inputs.model_dump_json().encode(),
    )
    return workflow


def prepare_edp_render(
    options: RenderOptions,
    *,
    files: LocalFiles,
    staging: StagingDirectory,
    runtime_dir: Path,
    run_id: UUID,
    source_schema: Any = None,
) -> dict[str, Any]:
    """Freeze a render template and preserve the existing column preflight."""
    template = files.resolve(options.template_path)[0]
    document = json.loads(template.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Render template must be a JSON object")
    if source_schema is not None:
        errors = _check_column_refs(
            source_schema, _collect_column_refs(document.get("fields") or {})
        )
        if errors:
            raise ValueError("Render template preflight failed: " + "; ".join(errors))
    staging.write(
        "render-inputs.json",
        json.dumps(
            {
                "run_id": str(run_id),
                "data_source": to_storage_api_path(options.data_source),
                "shard_size": options.shard_size,
                "limit": options.limit,
            },
            sort_keys=True,
        ).encode(),
    )
    staging.freeze(template, RENDER_TEMPLATE_FILENAME)
    staging.freeze(runtime_dir / RENDER_FILENAME)
    staging.freeze(runtime_dir / "pod_runtime" / RENDER_REQUIREMENTS)
    command = build_render_command(
        output_dir=staging.storage_path(),
        data_source=to_storage_api_path(options.data_source),
        shard_size=options.shard_size,
        limit=options.limit,
    )
    return build_edp_workflow(
        run_id=run_id, command=command, cpu=options.cpu, memory=options.memory
    )


def prepare_edp_aggregate(
    options: AggregateOptions,
    *,
    staging: StagingDirectory,
    runtime_dir: Path,
    run_id: UUID,
) -> dict[str, Any]:
    """Freeze aggregation code; finished-shard selection remains caller-owned."""
    if to_storage_api_path(options.out_dir) != staging.storage_path():
        raise ValueError("Aggregate out_dir must match its staging Storage directory")
    for name in (AGGREGATE_FILENAME, SLIME_CONVERTER_FILENAME, SFT_CONVERTER_FILENAME):
        staging.freeze(runtime_dir / name)
    command = build_aggregate_command(
        out_dir=staging.storage_path(),
        run_dirs=[to_storage_api_path(path) for path in options.run_dirs],
        protocol=options.protocol,
        min_reward=options.min_reward,
        system_prompt=options.system_prompt,
        limit=options.limit,
    )
    return build_edp_workflow(
        run_id=run_id, command=command, cpu=options.cpu, memory=options.memory
    )
