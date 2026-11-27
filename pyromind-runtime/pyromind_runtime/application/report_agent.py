"""Evidence and delivery tools bound to one approved report-producing attempt."""

import asyncio
import codecs
from collections.abc import Callable
from pathlib import PurePosixPath

from pyromind_runtime.application.plan_artifact_delivery import (
    InvalidPlanDelivery,
    validate_plan_report,
)
from pyromind_runtime.application.stage_analysis import ROLES
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.experiment_plans import ExperimentTask
from pyromind_runtime.domain.storage_paths import (
    from_storage_api_path,
    resolve_conversation_storage_path,
    to_storage_api_path,
)
from pyromind_runtime.domain.versions import canonical, check_persistable, fingerprint
from pyromind_runtime.ports.harness import ProjectReportAgent, SessionHandle
from pyromind_runtime.ports.storage import EvaluationStorage, EvidenceRangeReader


async def prepare_report_outputs(
    agent: ProjectReportAgent,
    handle: SessionHandle,
    context: RequestContext,
    storage: EvaluationStorage,
    task: ExperimentTask,
    run: ExperimentExecutionView,
    row: dict,
    directories: tuple[str, ...],
    on_tool: Callable[[str | None], None],
) -> dict[str, str]:
    primary = task.report_output
    assert primary is not None
    outputs = row.get("verified_outputs") or {}
    evidence_paths = row.get("evidence_paths") or []
    if not isinstance(outputs, dict) or not isinstance(evidence_paths, list):
        raise InvalidPlanDelivery("invalid_output", "evidence_paths")
    paths = {
        from_storage_api_path(v)
        for v in [*outputs.values(), *evidence_paths]
        if isinstance(v, str) and v.startswith("/")
    }
    roots = {PurePosixPath(from_storage_api_path(p)) for p in directories}
    ids: list[str] = list(
        dict.fromkeys(
            [
                *run.platform_task_ids,
                *([run.platform_task_id] if run.platform_task_id else []),
            ]
        )
    )
    approved_names = {o.name for o in task.outputs if o != primary}
    accepted: dict[str, str] = {}
    inspected_paths: set[str] = set()
    last_error = InvalidPlanDelivery("missing_output", "evidence_paths")

    def scope_path(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Expected an evidence path")
        path = from_storage_api_path(value)
        if path not in paths and not any(
            PurePosixPath(path).is_relative_to(root) for root in roots
        ):
            raise ValueError("Path outside the current task evidence")
        return path

    def execute(method: str, args: JsonObject) -> JsonObject:
        if method in {"list_evidence", "read_evidence"}:
            workspace_path = scope_path(args.get("path"))
            path = to_storage_api_path(workspace_path)
            if method == "list_evidence":
                names = sorted(storage.list(path))
                for name in names:
                    if name and name not in {".", ".."} and "/" not in name:
                        paths.add(f"{workspace_path}/{name}")
                return {"names": [*names[:1000]], "total": len(names)}
            offset = args.get("offset", 0)
            if type(offset) is not int or offset < 0:
                raise ValueError("Invalid offset")
            size = 16 * 1024
            if isinstance(storage, EvidenceRangeReader):
                chunk = storage.read_evidence_range(path, offset, size)
            else:
                chunk = storage.read(path)[offset : offset + size]
            decoder = codecs.getincrementaldecoder("utf-8")()
            text = decoder.decode(chunk, final=len(chunk) < size)
            if text.strip():
                inspected_paths.add(workspace_path)
            return {
                "text": text,
                "next_offset": offset + len(chunk) - len(decoder.getstate()[0])
                if len(chunk) == size
                else None,
            }
        if method != "submit_report":
            raise ValueError("Unknown report tool")
        if accepted:
            return {"accepted": True}
        body, other = args.get("report"), args.get("outputs")
        if not inspected_paths:
            raise InvalidPlanDelivery("missing_output", "evidence_paths")
        if (
            not isinstance(body, dict)
            or not isinstance(other, dict)
            or set(other) != approved_names
        ):
            raise InvalidPlanDelivery("invalid_output", "verified_outputs")
        mapped = {key: scope_path(value) for key, value in other.items()}
        # Bind the source and destination in the host. Model output cannot select
        # another attempt or overwrite its files.
        body = dict(body)
        if ids:
            source: JsonObject = (
                {"platform_task_id": ids[0]}
                if len(ids) == 1
                else {"platform_task_ids": [*ids]}
            )
            body["source"] = source
        files = body.get("files")
        if (run.stage_snapshot or {}).get("kind") == "dataset" and isinstance(
            files, list
        ):
            for file in files:
                if isinstance(file, dict) and isinstance(file.get("path"), str):
                    scope_path(file["path"])
        check_persistable(body)
        payload = canonical(body).encode()
        if len(payload) > 48 * 1024:
            raise InvalidPlanDelivery("invalid_report", primary.name)
        attempt_dir = fingerprint([run.plan_sequence, run.run_id])
        report_path = resolve_conversation_storage_path(
            handle.session_id,
            f"reports/{attempt_dir}/{fingerprint(body)}.json",
        )
        mapped[primary.name] = report_path
        validate_plan_report(
            storage,
            task,
            run,
            {k: to_storage_api_path(v) for k, v in mapped.items()},
            require_schema_version=True,
            draft_report=body,
        )
        if (run.stage_snapshot or {}).get("kind") == "dataset":
            if not isinstance(files, list):
                raise InvalidPlanDelivery("invalid_report", "files")
            for index, file in enumerate(files):
                if not isinstance(file, dict):
                    raise InvalidPlanDelivery("invalid_report", f"files[{index}]")
                groups = file.get("distributions")
                labels = groups.get("label") if isinstance(groups, dict) else None
                if (
                    not labels
                    and file.get("label_status") in ("unlabeled", "unavailable")
                    and not body.get("distribution")
                ):
                    continue
                field = f"files[{index}].distributions.label"
                if not isinstance(labels, dict) or not labels:
                    raise InvalidPlanDelivery("invalid_report", field)
                total = 0
                for label, count in labels.items():
                    if not label or type(count) is not int or count < 0:
                        raise InvalidPlanDelivery("invalid_report", field)
                    total += count
                sample_count = file.get("sample_count")
                if type(sample_count) is not int or (
                    file.get("label_mode") != "multilabel" and total != sample_count
                ):
                    raise InvalidPlanDelivery("invalid_report", field)
        if (run.stage_snapshot or {}).get("kind") == "init" and task.kind == "analysis":
            if not body.get("findings"):
                raise InvalidPlanDelivery("invalid_report", "findings")
            references = body.get("evidence_paths")
            if (
                not isinstance(references, list)
                or not references
                or not all(
                    isinstance(p, str) and p in inspected_paths for p in references
                )
            ):
                raise InvalidPlanDelivery("invalid_report", "evidence_paths")
        storage.write(to_storage_api_path(report_path), payload)
        accepted.update(mapped)
        return {"accepted": True}

    async def tool(method: str, args: JsonObject) -> JsonObject:
        nonlocal last_error
        if method not in {"list_evidence", "read_evidence", "submit_report"}:
            return {"error": "Unknown report tool"}
        on_tool(method)
        try:
            return await asyncio.to_thread(execute, method, args)
        except InvalidPlanDelivery as exc:
            last_error = exc
            return {"error": exc.code, "field": exc.field}
        except (ValueError, LookupError) as exc:
            return {"error": str(exc)[:240]}
        except Exception:
            # Transport exceptions can carry a signed download URL.
            return {"error": "storage_unavailable"}
        finally:
            on_tool(None)

    role = ROLES[str((run.stage_snapshot or {}).get("kind"))]
    evidence = {
        "task": task.model_dump(mode="json"),
        "platform_task_ids": ids,
        "output_directories": sorted(map(str, roots)),
        "evidence_paths": sorted(paths),
        "known_outputs": outputs,
        "language": context.accept_language or "zh-CN",
    }
    await agent.prepare_project_report(handle, role, evidence, context, tool)
    if not accepted:
        raise last_error
    return accepted
