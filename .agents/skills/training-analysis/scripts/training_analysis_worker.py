"""Single-call worker for the Pi training_analysis business tool."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import train_analysis


_SECRET = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|secret|token|"
    r"credential|private[_-]?key|access[_-]?key|refresh[_-]?token|cluster)",
    re.IGNORECASE,
)
_INLINE_SECRET = re.compile(
    r"((?:api[_-]?key|authorization|cookie|password|secret|token|cluster)"
    r"\s*[:=]\s*)"
    r"([^\s,;\"']+)",
    re.IGNORECASE,
)
_MAX_ERROR_LENGTH = 2000


def _sanitized(
    value: Any,
    key: str = "",
    depth: int = 0,
    secrets: tuple[str, ...] = (),
) -> Any:
    if _SECRET.search(key):
        return "[REDACTED]"
    if depth >= 10:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            str(name): _sanitized(child, str(name), depth + 1, secrets)
            for name, child in list(value.items())[:200]
        }
    if isinstance(value, list):
        return [_sanitized(child, key, depth + 1, secrets) for child in value[:1000]]
    if isinstance(value, str):
        limit = _MAX_ERROR_LENGTH if key == "error_message" else 100_000
        return _redact_text(value, secrets)[:limit]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _secret_values(value: Any, key: str = "") -> tuple[str, ...]:
    found: set[str] = set()
    if isinstance(value, dict):
        for name, child in value.items():
            name_text = str(name)
            if _SECRET.search(name_text) and isinstance(child, str) and child:
                found.add(child)
            else:
                found.update(_secret_values(child, name_text))
    elif isinstance(value, list):
        for child in value:
            found.update(_secret_values(child, key))
    return tuple(sorted(found, key=len, reverse=True))


def _credential_values(creds: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for name, value in creds.items()
                if value and _SECRET.search(str(name))
            },
            key=len,
            reverse=True,
        )
    )


def _redact_text(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    return _INLINE_SECRET.sub(r"\1[REDACTED]", value)


def _validate_payload(payload: dict[str, Any]) -> None:
    operation = payload.get("operation")
    if operation not in {"probe", "analyze", "report"}:
        raise ValueError("operation must be probe, analyze, or report")
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id is required")
    keys = payload.get("keys")
    if keys is not None and (
        not isinstance(keys, list)
        or len(keys) > 20
        or any(not isinstance(key, str) for key in keys)
    ):
        raise ValueError("keys must be a list of at most 20 strings")
    if operation != "report" and payload.get("output_path") is not None:
        raise ValueError("output_path is only valid for operation=report")
    if operation == "report" and payload.get("output_path") == "":
        raise ValueError("output_path must not be empty")
    output_relative = payload.get("output_relative")
    if output_relative is not None:
        if not isinstance(output_relative, str):
            raise ValueError("output_relative must be a string")
        relative = Path(output_relative)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2] != ("public_data", "training-analysis")
        ):
            raise ValueError(
                "output_path must stay under public_data/training-analysis"
            )


def _validate_report_target(payload: dict[str, Any]) -> None:
    """Re-check the executor's path boundary inside the worker process."""
    relative_text = payload.get("output_relative")
    output_text = payload.get("output_path")
    if not isinstance(relative_text, str) or not relative_text:
        return
    workspace = Path.cwd().resolve()
    relative = Path(relative_text)
    root = workspace / "public_data" / "training-analysis"
    current = workspace
    for part in relative.parts:
        if part == ".":
            continue
        current /= part
        if current.is_symlink():
            raise ValueError(
                "output_path must not traverse symbolic links under "
                "public_data/training-analysis"
            )
    target = (workspace / relative).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError("output_path must stay under public_data/training-analysis")
    if isinstance(output_text, str) and output_text:
        if Path(output_text).resolve() != target:
            raise ValueError("output_path and output_relative do not match")


def _resolve_from_task(payload: dict[str, Any], creds_path: Path) -> dict[str, Any]:
    headers = payload.get("headers") if isinstance(payload.get("headers"), dict) else {}
    target = train_analysis.TrainingAnalysisService().resolve_target(
        api_base=payload.get("api_base") or train_analysis._default_api_base(),
        cookie=str(headers.get("cookie") or ""),
        cluster=str(headers.get("x-cluster") or ""),
        authorization=str(headers.get("authorization") or ""),
        api_key="",
        creds_file="",
        entity="",
        data_source=str(payload.get("data_source") or ""),
        task_id=str(payload["task_id"]),
        run_url=str(payload.get("run_url") or ""),
        creds_out=str(creds_path),
    )
    if creds_path.exists():
        os.chmod(creds_path, 0o600)
    return target


def _operation(
    payload: dict[str, Any],
    target: dict[str, Any],
    creds_path: Path,
) -> dict[str, Any]:
    operation = str(payload["operation"])
    entity = str(target.get("entity") or "")
    project = str(target.get("project") or "")
    run_id = str(target.get("run_id") or "")
    if not entity or not project or not run_id:
        raise ValueError("resolved training target is missing entity/project/run_id")
    common = {
        "creds_file": creds_path,
        "data_source": str(
            target.get("data_source") or payload.get("data_source") or "wandb"
        ),
    }
    entity_project = f"{entity}/{project}"
    if operation == "probe":
        return train_analysis.TrainingAnalysisService().probe(
            **common, entity_project=entity_project, run_id=run_id
        )
    if operation == "analyze":
        service = train_analysis.TrainingAnalysisService()
        analyze = getattr(service, "analyze", None) or getattr(
            service, "analyze_run"
        )
        return analyze(
            **common,
            entity_project=entity_project,
            run_id=run_id,
            metric=str(payload.get("metric") or ""),
            keys=[str(key) for key in payload.get("keys") or []],
        )
    if operation == "report":
        output = str(payload.get("output_path") or "")
        message = train_analysis.TrainingAnalysisService().report(
            **common,
            entity_project=entity_project,
            run_id=run_id,
            metric=str(payload.get("metric") or ""),
            output_path=output,
        )
        return {"message": message}
    raise ValueError(f"unsupported operation: {operation}")


def main() -> int:
    payload: dict[str, Any] = {}
    creds: dict[str, str] = {}
    try:
        try:
            loaded = json.load(sys.stdin)
        except json.JSONDecodeError as exc:
            response = {
                "ok": False,
                "failure_stage": "worker_input",
                "error_code": "invalid_worker_input",
                "error_message": str(exc)[:_MAX_ERROR_LENGTH],
            }
            print(json.dumps(response, ensure_ascii=False))
            return 2
        if not isinstance(loaded, dict):
            response = {
                "ok": False,
                "failure_stage": "worker_input",
                "error_code": "invalid_worker_input",
                "error_message": "worker input must be a JSON object",
            }
            print(json.dumps(response, ensure_ascii=False))
            return 2
        payload = loaded
        _validate_payload(payload)
        _validate_report_target(payload)
        with tempfile.TemporaryDirectory(prefix="pyromind-training-") as directory:
            creds_path = Path(directory) / "creds.json"
            target = _resolve_from_task(payload, creds_path)
            creds = train_analysis._load_creds(creds_path)
            result = _operation(payload, target, creds_path)
            report_path = payload.get("output_path")
            if isinstance(report_path, str) and report_path:
                report_file = Path(report_path)
                if report_file.is_file():
                    report = report_file.read_text(encoding="utf-8")
                    report_file.write_text(
                        _redact_text(
                            report,
                            tuple(
                                sorted(
                                    set(_secret_values(payload))
                                    | set(_credential_values(creds)),
                                    key=len,
                                    reverse=True,
                                )
                            ),
                        ),
                        encoding="utf-8",
                    )
        response = {
            "ok": True,
            "target": {
                key: value
                for key, value in target.items()
                if key in {"task_id", "data_source", "entity", "project", "run_id"}
            },
            "result": result,
            "report_path": payload.get("output_relative"),
        }
        print(
            json.dumps(
                _sanitized(
                    response,
                    secrets=tuple(
                        sorted(
                            set(_secret_values(payload))
                            | set(_credential_values(creds)),
                            key=len,
                            reverse=True,
                        )
                    ),
                ),
                ensure_ascii=False,
            )
        )
        return 0
    except PermissionError as exc:
        response = {
            "ok": False,
            "failure_stage": "credentials",
            "error_code": "training_credentials_unavailable",
            "error_message": str(exc)[:_MAX_ERROR_LENGTH],
        }
    except train_analysis.WandbAnalysisError as exc:
        response = {
            "ok": False,
            "failure_stage": "target_resolution",
            "error_code": "training_target_resolution_failed",
            "error_message": str(exc)[:_MAX_ERROR_LENGTH],
        }
    except (ValueError, json.JSONDecodeError) as exc:
        response = {
            "ok": False,
            "failure_stage": "target_resolution",
            "error_code": "training_target_invalid",
            "error_message": str(exc)[:_MAX_ERROR_LENGTH],
        }
    except Exception as exc:
        response = {
            "ok": False,
            "failure_stage": "analysis_execution",
            "error_code": "training_analysis_failed",
            "error_message": f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_LENGTH],
        }
    print(
        json.dumps(
            _sanitized(
                response,
                secrets=tuple(
                    sorted(
                        set(_secret_values(payload))
                        | set(_credential_values(creds)),
                        key=len,
                        reverse=True,
                    )
                ),
            ),
            ensure_ascii=False,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
