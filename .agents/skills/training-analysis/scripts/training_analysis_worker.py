"""Single-call worker for the Pi training_analysis business tool."""

from __future__ import annotations

import json
import math
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
_CONFIG_SUMMARY_KEYS = {
    "_name_or_path",
    "model_type",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_steps",
    "warmup_ratio",
    "weight_decay",
    "optim",
    "eval_strategy",
    "logging_steps",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "max_length",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "seed",
    "bf16",
    "fp16",
}
_PEFT_SUMMARY_KEYS = {
    "peft_type",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "modules_to_save",
}


def _sanitized(
    value: Any,
    key: str = "",
    depth: int = 0,
    secrets: tuple[str, ...] = (),
) -> Any:
    public_count = key in {
        "max_tokens",
        "max_new_tokens",
        "min_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    if public_count and isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
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
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
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


def _error_description(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    for _ in range(2):
        if cause is None:
            break
        parts.append(f"caused by {type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return "; ".join(parts)[:_MAX_ERROR_LENGTH]


def _wandb_api_access_forbidden(exc: Exception) -> bool:
    from wandb.errors import AuthenticationError
    from wandb.sdk.lib.service.service_connection import WandbApiFailedError

    cause = exc.__cause__
    if not isinstance(exc, AuthenticationError) or not isinstance(
        cause, WandbApiFailedError
    ):
        return False
    response = cause.response
    return (
        response is not None
        and response.http_status == 403
        and "<!doctype html>" in str(cause).lower()
    )


def _compact_analysis_result(
    result: dict[str, Any], metric: str, keys: list[str]
) -> dict[str, Any]:
    compact = dict(result)
    config = result.get("config")
    if isinstance(config, dict):
        config_summary = {
            key: value for key, value in config.items() if key in _CONFIG_SUMMARY_KEYS
        }
        peft_config = config.get("peft_config")
        if isinstance(peft_config, dict):
            default_peft = peft_config.get("default")
            if isinstance(default_peft, dict):
                selected_peft = {
                    key: value
                    for key, value in default_peft.items()
                    if key in _PEFT_SUMMARY_KEYS
                }
                if selected_peft:
                    config_summary["peft_config"] = selected_peft
        compact["config"] = config_summary
        compact["omitted_config_key_count"] = len(config) - len(config_summary)

    requested_metrics = set(keys) | {metric}
    if "train/loss" in requested_metrics:
        requested_metrics.add("train_loss")
    summary = result.get("summary")
    if isinstance(summary, dict):
        compact["summary"] = {
            key: value for key, value in summary.items() if key in requested_metrics
        }

    metric_series = result.get("metric_series")
    if isinstance(metric_series, dict):
        summaries: dict[str, Any] = {}
        for name, points in metric_series.items():
            if not isinstance(points, list):
                continue
            finite = [
                point
                for point in points
                if isinstance(point, dict)
                and isinstance(point.get("value"), (int, float))
                and not isinstance(point.get("value"), bool)
                and math.isfinite(point["value"])
            ]
            curve: dict[str, Any] = {
                "count": len(points),
                "nonfinite_count": len(points) - len(finite),
            }
            if finite:
                best = min(finite, key=lambda point: point["value"])
                final = finite[-1]
                curve.update(
                    {
                        "first": {
                            "step": finite[0].get("step"),
                            "value": finite[0]["value"],
                        },
                        "minimum": {
                            "step": best.get("step"),
                            "value": best["value"],
                        },
                        "maximum": {
                            "step": max(finite, key=lambda point: point["value"]).get(
                                "step"
                            ),
                            "value": max(point["value"] for point in finite),
                        },
                        "final": {
                            "step": final.get("step"),
                            "value": final["value"],
                        },
                    }
                )
            if name in {"eval/loss", "eval_loss", "val_loss"}:
                rebound = train_analysis._validation_loss_rebound(points)
                if rebound:
                    curve["sustained_rebound"] = rebound
            summaries[name] = curve
        compact["metric_summaries"] = summaries
        compact.pop("metric_series", None)
    compact.pop("multi_metrics", None)
    compact["detail_level"] = "summary"
    return compact


def _validate_payload(payload: dict[str, Any]) -> None:
    operation = payload.get("operation")
    if operation not in {"probe", "analyze", "report", "artifacts"}:
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
    include_history = payload.get("include_history", False)
    if not isinstance(include_history, bool):
        raise ValueError("include_history must be a boolean")
    if operation != "analyze" and include_history:
        raise ValueError("include_history is only valid for operation=analyze")
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
        training_node_id=payload.get("training_node_id"),
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
        analyze = getattr(service, "analyze", None) or getattr(service, "analyze_run")
        result = analyze(
            **common,
            entity_project=entity_project,
            run_id=run_id,
            metric=str(payload.get("metric") or ""),
            keys=[str(key) for key in payload.get("keys") or []],
        )
        if payload.get("include_history") is True:
            result["detail_level"] = "full"
            return result
        return _compact_analysis_result(
            result,
            str(payload.get("metric") or ""),
            [str(key) for key in payload.get("keys") or []],
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
        if payload["operation"] == "artifacts":
            from training_artifacts import resolve_artifacts

            headers = payload.get("headers") or {}
            client = train_analysis.PlatformClient(
                base_url=payload.get("api_base") or train_analysis._default_api_base(),
                cookie=headers.get("cookie", ""),
                authorization=headers.get("authorization", ""),
                cluster=headers.get("x-cluster", ""),
            )
            result = resolve_artifacts(
                client, payload["task_id"], payload.get("training_node_id")
            )
            print(
                json.dumps(
                    _sanitized(
                        {
                            "ok": True,
                            "target": {"task_id": payload["task_id"]},
                            "result": result,
                        },
                        secrets=_secret_values(payload),
                    ),
                    ensure_ascii=False,
                )
            )
            return 0
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
    except train_analysis.InvalidPlatformTaskId as exc:
        response = {
            "ok": False,
            "failure_stage": "input_resolution",
            "error_code": "invalid_platform_task_id",
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
        if _wandb_api_access_forbidden(exc):
            response = {
                "ok": False,
                "failure_stage": "external_service",
                "error_code": "wandb_api_access_forbidden",
                "error_message": (
                    "W&B API returned HTTP 403 Forbidden HTML while verifying "
                    "credentials. This does not establish whether the API key is "
                    "valid. Check W&B API access from the local backend host."
                ),
            }
        else:
            response = {
                "ok": False,
                "failure_stage": "analysis_execution",
                "error_code": "training_analysis_failed",
                "error_message": _error_description(exc),
            }
    print(
        json.dumps(
            _sanitized(
                response,
                secrets=tuple(
                    sorted(
                        set(_secret_values(payload)) | set(_credential_values(creds)),
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
