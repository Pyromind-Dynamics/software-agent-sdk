"""Verified, bounded human-review checkpoints for delegated tasks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath

from pyromind_runtime.domain.execution_policy import ExecutionCheckpoint
from pyromind_runtime.domain.experiment_plans import PlanOwner


async def verify_checkpoint(
    owner: PlanOwner,
    row: dict,
    scope: dict,
    arguments: dict,
    read: Callable[..., Awaitable[bytes]],
    inspect: Callable[..., Awaitable[dict]] | None,
) -> dict:
    policy = (row.get("task_spec") or {}).get("execution_policy") or {}
    document = ExecutionCheckpoint.model_validate(arguments)
    if document.run_id != row["id"] or not policy.get("require_review"):
        raise PermissionError("checkpoint does not belong to this execution")
    if document.sample_count != min(
        policy.get("sample_limit", 3), document.source_count
    ):
        raise ValueError(
            "sample must contain the configured number of available records"
        )
    root = PurePosixPath("public_data/experiment-runs") / row["id"]
    evidence = {}
    bodies = {}
    for name in ("script_path", "sample_input", "sample_output"):
        path = getattr(document, name)
        colocated_input = (
            name == "sample_input"
            and PurePosixPath(path).parent == PurePosixPath(document.source_path).parent
            and row["id"] in PurePosixPath(path).name
        )
        if (
            not PurePosixPath(path).is_relative_to(root) and not colocated_input
        ) or ".." in PurePosixPath(path).parts:
            raise PermissionError("checkpoint artifacts must belong to this task")
        body = await read(owner, path)
        if len(body) > 65536:
            raise ValueError("sample artifacts must be bounded")
        bodies[name] = body
        evidence[path] = hashlib.sha256(body).hexdigest()
    inputs = [
        json.loads(line) for line in bodies["sample_input"].splitlines() if line.strip()
    ]
    outputs = [
        json.loads(line)
        for line in bodies["sample_output"].splitlines()
        if line.strip()
    ]
    if (
        len(inputs) != document.sample_count
        or len(outputs) != document.retained_count
        or len(outputs) > 3
    ):
        raise ValueError("sample counts do not match the actual artifacts")
    if document.validation.get("status") != "passed":
        raise ValueError("sample validation must pass before review")
    if (
        document.configuration.get("input_path") != document.source_path
        or document.configuration.get("script_path") != document.script_path
        or document.configuration.get("mode", "full") != "full"
    ):
        raise ValueError(
            "checkpoint.source_path must equal configuration.input_path; "
            "script_path must equal configuration.script_path and mode must be full"
        )
    if inspect is None:
        raise ValueError("resource verification is unavailable")
    source = await inspect(owner, document.source_path, "file")
    if (
        document.source_path.endswith(".jsonl")
        and source.get("nonempty_lines") is not None
        and source["nonempty_lines"] != document.source_count
    ):
        raise ValueError("source count does not match the actual JSONL input")
    task_inputs = row["task_spec"]["task"].get("inputs", {})
    source_inputs = [
        value
        for value in task_inputs.values()
        if isinstance(value, str) and value.startswith("/")
    ]
    if source_inputs and not any(
        value == document.source_path
        or (
            (value.endswith("/") or task_inputs.get("input_kind") == "directory")
            and PurePosixPath(document.source_path).is_relative_to(PurePosixPath(value))
        )
        for value in source_inputs
    ):
        raise ValueError(
            "sample source differs from the frozen task input; "
            "the source must match a frozen file path or be inside a directory "
            "declared with input_kind=directory or a trailing slash"
        )
    evidence[document.source_path] = source["sha256"]
    support = document.configuration.get("support_file_path")
    if isinstance(support, str):
        evidence[support] = hashlib.sha256(await read(owner, support)).hexdigest()
    for path in document.implementation_files:
        if (
            not PurePosixPath(path).is_relative_to(root)
            or ".." in PurePosixPath(path).parts
        ):
            raise PermissionError("implementation files must belong to this task")
        body = await read(owner, path)
        if not body or len(body) > 65536:
            raise ValueError("implementation files must be nonempty and bounded")
        evidence[path] = hashlib.sha256(body).hexdigest()
    document = ExecutionCheckpoint.model_validate(
        {
            **document.model_dump(mode="json"),
            "preview": ({"input_records": inputs, "output_records": outputs},),
        }
    )
    frozen = {
        "document": document.model_dump(mode="json"),
        "evidence": evidence,
    }
    checkpoint = {
        **frozen,
        "checkpoint_id": hashlib.sha256(
            json.dumps(
                {**frozen, "work_id": scope["work_id"]},
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest(),
        "status": "pending",
    }
    return checkpoint


async def require_approved_checkpoint(
    owner: PlanOwner,
    row: dict,
    arguments: dict,
    read: Callable[..., Awaitable[bytes]],
    inspect: Callable[..., Awaitable[dict]] | None,
) -> None:
    checkpoint = (row.get("attributes") or {}).get("checkpoint") or {}
    if checkpoint.get("status") != "approved" or arguments != checkpoint.get(
        "document", {}
    ).get("configuration"):
        raise PermissionError(
            "full submission must match the approved sample configuration"
        )
    for path, expected in checkpoint["evidence"].items():
        if path == checkpoint["document"]["source_path"]:
            if inspect is None:
                raise ValueError("resource verification is unavailable")
            actual = (await inspect(owner, path, "file"))["sha256"]
        else:
            actual = hashlib.sha256(await read(owner, path)).hexdigest()
        if actual != expected:
            raise ValueError(
                "sample inputs or implementation changed; run and confirm a new sample"
            )
