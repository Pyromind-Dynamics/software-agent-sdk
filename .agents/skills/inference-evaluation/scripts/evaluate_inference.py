#!/usr/bin/env python3
"""Run dataset-driven evaluation against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import html
import json
import mimetypes
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any


RequestBuilder = Callable[[dict[str, Any], Path, str, dict[str, Any]], dict[str, Any]]


_AGENT_RUBRIC_MODE = "agent_rubric"
_MISSING = object()


def _json_object(value: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _chat_completions_endpoint(endpoint: str) -> str:
    value = endpoint.strip().rstrip("/")
    if not value:
        raise ValueError("endpoint must not be empty")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise
        if not isinstance(row, dict):
            raise TypeError("checkpoint rows must be objects")
        rows.append(row)
    return rows


def _required_field(
    sample: dict[str, Any], config: dict[str, Any], config_name: str
) -> Any:
    field = config.get(config_name)
    if not isinstance(field, str) or not field:
        raise ValueError(f"dataset_config_json.{config_name} is required")
    if field not in sample:
        raise KeyError(f"sample is missing configured field {field!r}")
    return sample[field]


def _optional_field(
    sample: dict[str, Any], config: dict[str, Any], config_name: str
) -> Any:
    field = config.get(config_name)
    if not isinstance(field, str) or not field:
        return None
    return sample.get(field)


def _media_values(sample: dict[str, Any], config: dict[str, Any]) -> list[str]:
    media_field = config.get("media_field", config.get("image_field"))
    if not isinstance(media_field, str) or not media_field:
        return []
    value = sample.get(media_field)
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    if not all(isinstance(item, str) for item in values):
        raise TypeError("configured media field must contain string paths or URLs")
    image_order = config.get("image_order")
    if isinstance(image_order, list) and all(
        isinstance(item, str) for item in image_order
    ):
        order = {name: index for index, name in enumerate(image_order)}
        values = sorted(
            values,
            key=lambda item: order.get(Path(item).name, len(order)),
        )
    return values


def _input_media(sample: dict[str, Any], config: dict[str, Any]) -> list[str]:
    values = list(_media_values(sample, config))
    messages = sample.get(config.get("messages_field", ""), [])
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or not isinstance(
                message.get("content"), list
            ):
                continue
            for part in message["content"]:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image" and isinstance(part.get("path"), str):
                    values.append(part["path"])
                image = part.get("image_url")
                if isinstance(image, dict) and isinstance(image.get("url"), str):
                    values.append(image["url"])
    return values


def _media_path(value: str, dataset_file: Path, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base_dir = config.get("media_base_dir", config.get("image_root"))
    if isinstance(base_dir, str) and base_dir:
        configured = Path(base_dir)
        root = (
            configured if configured.is_absolute() else dataset_file.parent / configured
        )
    else:
        root = dataset_file.parent
    return (root / path).resolve()


def _media_url(value: str, dataset_file: Path, config: dict[str, Any]) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = _media_path(value, dataset_file, config)
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_content(content: Any, dataset_file: Path, config: dict[str, Any]) -> Any:
    if not isinstance(content, list):
        return content
    normalized: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            normalized.append(part)
            continue
        kind = part.get("type")
        if kind == "image" and isinstance(part.get("path"), str):
            normalized.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _media_url(part["path"], dataset_file, config)
                    },
                }
            )
            continue
        image_url = part.get("image_url")
        if kind == "image_url" and isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str):
                normalized.append(
                    {
                        **part,
                        "image_url": {
                            **image_url,
                            "url": _media_url(url, dataset_file, config),
                        },
                    }
                )
                continue
        normalized.append(part)
    return normalized


def _default_request(
    sample: dict[str, Any],
    dataset_file: Path,
    model: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    messages_field = config.get("messages_field")
    if isinstance(messages_field, str) and messages_field:
        messages = sample.get(messages_field)
        if not isinstance(messages, list):
            raise TypeError(f"{messages_field!r} must contain a messages list")
        normalized_messages = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("each message must be an object")
            normalized_messages.append(
                {
                    **message,
                    "content": _normalize_content(
                        message.get("content"), dataset_file, config
                    ),
                }
            )
    else:
        prompt = _required_field(sample, config, "user_prompt_field")
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": _media_url(value, dataset_file, config)},
            }
            for value in _media_values(sample, config)
        ]
        content.append({"type": "text", "text": str(prompt)})
        normalized_messages = []
        system_prompt = _optional_field(sample, config, "system_prompt_field")
        if system_prompt is not None:
            normalized_messages.append(
                {"role": "system", "content": str(system_prompt)}
            )
        normalized_messages.append({"role": "user", "content": content})
    return {"model": model, "messages": normalized_messages}


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    objects: list[tuple[int, dict[str, Any]]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append((end, value))
    return max(objects, key=lambda item: item[0])[1] if objects else None


def _configured_rubrics(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rubrics = config.get("rubrics")
    if not isinstance(raw_rubrics, list) or not raw_rubrics:
        raise ValueError("agent_rubric requires a non-empty 'rubrics' list")
    if len(raw_rubrics) > 20:
        raise ValueError("agent_rubric supports at most 20 rubrics")
    rubrics: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in raw_rubrics:
        if not isinstance(raw, dict):
            raise TypeError("each configured rubric must be an object")
        name = raw.get("name")
        criterion = raw.get("criterion")
        weight = raw.get("weight")
        evaluator = raw.get("evaluator")
        required = raw.get("required", False)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each configured rubric requires a name")
        if name in names:
            raise ValueError(f"duplicate configured rubric name: {name!r}")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"configured rubric {name!r} requires a criterion")
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"configured rubric {name!r} requires a positive weight")
        if not isinstance(evaluator, dict):
            raise ValueError(f"configured rubric {name!r} requires an evaluator")
        if not isinstance(required, bool):
            raise ValueError(f"configured rubric {name!r} required must be boolean")
        names.add(name)
        rubrics.append(
            {
                "name": name.strip(),
                "criterion": criterion.strip(),
                "weight": float(weight),
                "evaluator": evaluator,
                "required": required,
            }
        )
    return rubrics


def _structured_value(value: Any) -> Any:
    if isinstance(value, str):
        parsed = _last_json_object(value)
        return parsed if parsed is not None else value.strip()
    return value


def _path_value(value: Any, path: Any) -> Any:
    if path in (None, ""):
        return value
    if not isinstance(path, str):
        raise TypeError("rubric evaluator path must be a string")
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _evidence_value(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    return rendered if len(rendered) <= 240 else rendered[:237] + "..."


def _box_coordinates(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        nested = value.get("bbox", value.get("box"))
        if nested is not None:
            return _box_coordinates(nested)
        values = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    elif isinstance(value, list) and len(value) == 4:
        values = value
    else:
        return None
    if not all(isinstance(item, (int, float)) for item in values):
        return None
    x1, y1, x2, y2 = (float(item) for item in values)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _bbox_score(prediction: Any, reference: Any, threshold: float) -> float:
    if not isinstance(prediction, list) or not isinstance(reference, list):
        return 0.0
    predicted = [box for item in prediction if (box := _box_coordinates(item))]
    expected = [box for item in reference if (box := _box_coordinates(item))]
    if len(predicted) != len(prediction) or len(expected) != len(reference):
        return 0.0
    if not expected or not predicted:
        return 1.0 if not expected and not predicted else 0.0
    unmatched = set(range(len(predicted)))
    matches = 0
    for expected_box in expected:
        candidates = [
            (index, _intersection_over_union(expected_box, predicted[index]))
            for index in unmatched
        ]
        if not candidates:
            continue
        index, score = max(candidates, key=lambda item: item[1])
        if score >= threshold:
            unmatched.remove(index)
            matches += 1
    precision = matches / len(predicted)
    recall = matches / len(expected)
    return 2 * precision * recall / (precision + recall) if matches else 0.0


def _score_configured_rubric(
    rubric: dict[str, Any], reference: Any, prediction: str
) -> tuple[float, list[str], str | None]:
    evaluator = rubric["evaluator"]
    kind = evaluator.get("type")
    expected_root = _structured_value(reference)
    predicted_root = _structured_value(prediction)
    prediction_path = evaluator.get("prediction_path", "")
    reference_path = evaluator.get("reference_path", "")
    actual = _path_value(predicted_root, prediction_path)
    expected = _path_value(expected_root, reference_path)

    if kind == "exact_match":
        left, right = str(reference).strip(), prediction.strip()
        if not evaluator.get("case_sensitive", False):
            left, right = left.casefold(), right.casefold()
        score = float(left == right)
    elif kind == "json_valid":
        score = float(isinstance(predicted_root, dict))
    elif kind == "required_fields":
        paths = evaluator.get("prediction_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("required_fields requires prediction_paths")
        score = sum(_path_value(predicted_root, path) is not _MISSING for path in paths)
        score /= len(paths)
    elif kind == "field_equals":
        score = float(
            actual is not _MISSING and expected is not _MISSING and actual == expected
        )
    elif kind == "non_empty":
        score = float(actual is not _MISSING and actual not in (None, "", [], {}))
    elif kind == "number_range":
        minimum = evaluator.get("min", float("-inf"))
        maximum = evaluator.get("max", float("inf"))
        if not isinstance(minimum, (int, float)) or not isinstance(
            maximum, (int, float)
        ):
            raise TypeError("number_range min and max must be numeric")
        score = float(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and minimum <= actual <= maximum
        )
    elif kind == "list_count_match":
        score = float(
            isinstance(actual, list)
            and isinstance(expected, list)
            and len(actual) == len(expected)
        )
    elif kind == "bbox_iou":
        threshold = evaluator.get("iou_threshold", 0.5)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("bbox_iou iou_threshold must be between 0 and 1")
        score = _bbox_score(actual, expected, float(threshold))
    else:
        raise ValueError(f"unsupported rubric evaluator type: {kind!r}")

    evidence = [
        f"prediction[{prediction_path or '<root>'}]={_evidence_value(actual)}",
        f"reference[{reference_path or '<root>'}]={_evidence_value(expected)}",
    ]
    failure_reason = (
        None if score == 1 else f"criterion not fully satisfied ({score:.3f})"
    )
    return score, evidence, failure_reason


def _agent_rubric_evaluation(
    reference: Any, prediction: str, config: dict[str, Any]
) -> dict[str, Any]:
    rubrics = _configured_rubrics(config)
    pass_threshold = float(config.get("pass_threshold", 0.7))
    rubric_threshold = float(config.get("rubric_pass_threshold", 0.7))
    if not 0 <= pass_threshold <= 1 or not 0 <= rubric_threshold <= 1:
        raise ValueError("rubric score thresholds must be between 0 and 1")
    results = []
    for rubric in rubrics:
        score, evidence, failure_reason = _score_configured_rubric(
            rubric, reference, prediction
        )
        results.append(
            {
                **rubric,
                "score": round(score, 6),
                "passed": score >= rubric_threshold,
                "evidence": evidence,
                "failure_reason": failure_reason,
            }
        )
    total_weight = sum(item["weight"] for item in results)
    overall_score = sum(item["score"] * item["weight"] for item in results)
    overall_score /= total_weight
    failures = [item["name"] for item in results if not item["passed"]]
    required_failures = [
        item["name"] for item in results if item["required"] and not item["passed"]
    ]
    return {
        "passed": overall_score >= pass_threshold and not required_failures,
        "overall_score": round(overall_score, 6),
        "rubric_results": results,
        "metrics": {
            "overall_score": overall_score,
            **{f"rubric.{item['name']}": item["score"] for item in results},
        },
        "details": {
            "failure_tags": failures,
            "required_failures": required_failures,
            "rubric_source": "agent",
        },
    }


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no assistant message")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        values = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text", part.get("value"))
            if isinstance(text, str) and text.strip():
                values.append(text.strip())
        if values:
            return "\n".join(values)
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    raise ValueError("assistant message has no text")


def _call_api(
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], float]:
    headers = {"Content-Type": "application/json", "User-Agent": "eval-node/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}") from exc
    value = json.loads(body)
    if not isinstance(value, dict):
        raise TypeError("API response is not an object")
    return value, time.monotonic() - started


def _validate_evaluation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("evaluator must return an object")
    if not isinstance(value.get("passed"), bool):
        raise TypeError("evaluator result must contain boolean 'passed'")
    metrics = value.get("metrics", {})
    if not isinstance(metrics, dict) or not all(
        isinstance(key, str)
        and isinstance(item, (int, float, bool))
        and not isinstance(item, complex)
        for key, item in metrics.items()
    ):
        raise TypeError("evaluator metrics must contain numeric or boolean values")
    details = value.get("details", {})
    if not isinstance(details, dict):
        raise TypeError("evaluator details must be an object")
    overall_score = value.get("overall_score")
    if overall_score is not None and (
        not isinstance(overall_score, (int, float)) or not 0 <= overall_score <= 1
    ):
        raise TypeError("evaluator overall_score must be between 0 and 1")
    rubric_results = value.get("rubric_results", [])
    if not isinstance(rubric_results, list):
        raise TypeError("evaluator rubric_results must be a list")
    return {
        "passed": value["passed"],
        "overall_score": overall_score,
        "rubric_results": rubric_results,
        "metrics": metrics,
        "details": details,
    }


def _case_id(sample: dict[str, Any], config: dict[str, Any], index: int) -> str:
    field = config.get("id_field", "id")
    value = sample.get(field) if isinstance(field, str) else None
    return str(value) if value is not None else f"case-{index:06d}"


def _has_reusable_prediction(result: dict[str, Any] | None) -> bool:
    if result is None:
        return False
    prediction = result.get("prediction")
    return isinstance(prediction, str) and bool(prediction.strip())


def _error_result(
    case_id: str,
    sample: dict[str, Any],
    reference: Any,
    prediction: str | None,
    media: list[str],
    error: Exception,
    *,
    latency_seconds: float | None = None,
    usage: Any = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "passed": False,
        "overall_score": None,
        "rubric_results": [],
        "reference": reference,
        "prediction": prediction,
        "metrics": {},
        "details": {},
        "latency_seconds": latency_seconds,
        "usage": usage,
        "media": media,
        "sample": sample,
        "error": f"{type(error).__name__}: {error}",
    }


def _evaluate_case(
    index: int,
    sample: dict[str, Any],
    dataset_file: Path,
    dataset_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    builder: RequestBuilder,
    endpoint: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
    on_request: Callable[[], None],
) -> dict[str, Any]:
    case_id = _case_id(sample, dataset_config, index)
    try:
        reference = _required_field(sample, dataset_config, "reference_field")
        media = _input_media(sample, dataset_config)
        payload = builder(sample, dataset_file, model, dataset_config)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return _error_result(case_id, sample, None, None, [], exc)
    payload.setdefault("model", model)
    payload.setdefault("max_tokens", max_tokens)
    payload.setdefault("temperature", temperature)
    last_error: Exception | None = None
    response: dict[str, Any] | None = None
    prediction: str | None = None
    elapsed: float | None = None
    for attempt in range(max_retries + 1):
        try:
            on_request()
            response, elapsed = _call_api(endpoint, payload, api_key, timeout_seconds)
            prediction = _response_text(response)
            if not prediction.strip():
                raise ValueError("empty model prediction")
            break
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            last_error = exc
            response, prediction, elapsed = None, None, None
            if attempt < max_retries:
                time.sleep(2**attempt)
    if response is None or prediction is None or elapsed is None:
        assert last_error is not None
        return _error_result(case_id, sample, reference, None, media, last_error)
    try:
        evaluation_latency = 0.0
        evaluation = _validate_evaluation(
            _agent_rubric_evaluation(reference, prediction, evaluation_config)
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        return _error_result(
            case_id,
            sample,
            reference,
            prediction,
            media,
            exc,
            latency_seconds=round(elapsed, 3),
            usage=response.get("usage"),
        )
    return {
        "id": case_id,
        "passed": evaluation["passed"],
        "overall_score": evaluation["overall_score"],
        "rubric_results": evaluation["rubric_results"],
        "reference": reference,
        "prediction": prediction,
        "metrics": evaluation["metrics"],
        "details": evaluation["details"],
        "latency_seconds": round(elapsed, 3),
        "evaluation_latency_seconds": round(evaluation_latency, 3),
        "usage": response.get("usage"),
        "media": media,
        "sample": sample,
        "error": None,
    }


def _metrics(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(row["passed"] for row in rows)
    errors = sum(row["error"] is not None for row in rows)
    latencies = [
        row["latency_seconds"]
        for row in rows
        if isinstance(row["latency_seconds"], (int, float))
    ]
    names = sorted({name for row in rows for name in row["metrics"]})
    aggregate: dict[str, float] = {}
    for name in names:
        values = [
            float(row["metrics"][name])
            for row in rows
            if isinstance(row["metrics"].get(name), (int, float, bool))
        ]
        if values:
            aggregate[name] = round(statistics.mean(values), 6)
    rubric_values: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for rubric in row.get("rubric_results", []):
            rubric_values.setdefault(rubric["name"], []).append(rubric)
    rubric_metrics = {
        name: {
            "count": len(values),
            "mean_score": round(
                statistics.mean(float(value["score"]) for value in values), 6
            ),
            "pass_rate": round(
                sum(bool(value["passed"]) for value in values) / len(values), 6
            ),
        }
        for name, values in sorted(rubric_values.items())
    }
    overall_scores = [
        float(row["overall_score"])
        for row in rows
        if isinstance(row.get("overall_score"), (int, float))
    ]
    return {
        **metadata,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "api_or_evaluation_errors": errors,
        "mean_latency_seconds": (
            round(statistics.mean(latencies), 3) if latencies else None
        ),
        "aggregate_metrics": aggregate,
        "mean_overall_score": (
            round(statistics.mean(overall_scores), 6) if overall_scores else None
        ),
        "rubric_metrics": rubric_metrics,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _pretty(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def _media_html(
    row: dict[str, Any], dataset_file: Path, dataset_config: dict[str, Any]
) -> str:
    figures = []
    for index, value in enumerate(row.get("media", []), 1):
        try:
            url = _media_url(value, dataset_file, dataset_config)
        except (OSError, ValueError):
            continue
        figures.append(
            f'<figure><img loading="lazy" src="{html.escape(url)}" '
            f'alt="media {index}"><figcaption>{html.escape(value)}</figcaption>'
            "</figure>"
        )
    return f'<div class="media">{"".join(figures)}</div>' if figures else ""


def _rubrics_html(row: dict[str, Any]) -> str:
    items = []
    for rubric in row.get("rubric_results", []):
        status = "passed" if rubric["passed"] else "failed"
        evidence = "".join(
            f"<li>{html.escape(item)}</li>" for item in rubric.get("evidence", [])
        )
        failure = html.escape(rubric.get("failure_reason") or "")
        items.append(
            f'<section class="rubric {status}"><header>'
            f"<strong>{html.escape(rubric['name'])}</strong>"
            f"<span>{rubric['score']:.2f} × {rubric['weight']}</span></header>"
            f"<p>{html.escape(rubric['criterion'])}</p>"
            f"<ul>{evidence}</ul>"
            f'<p class="failure">{failure}</p></section>'
        )
    if not items:
        return ""
    score = row.get("overall_score")
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
    return (
        '<div class="rubrics"><h3>Rubric evaluation '
        f"· overall {score_text}</h3>{''.join(items)}</div>"
    )


def _build_html(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    dataset_file: Path,
    dataset_config: dict[str, Any],
    output: Path,
) -> None:
    cards = []
    for row in sorted(rows, key=lambda item: (item["passed"], item["id"])):
        status = "passed" if row["passed"] else "failed"
        error = html.escape(row["error"] or "none")
        cards.append(
            f'<article class="case {status}" data-status="{status}">'
            f"<header><code>{html.escape(row['id'])}</code>"
            f"<strong>{status}</strong></header>"
            f"{_media_html(row, dataset_file, dataset_config)}"
            f"{_rubrics_html(row)}"
            '<div class="columns"><section><h3>Reference</h3><pre>'
            f"{_pretty(row['reference'])}</pre></section>"
            "<section><h3>Prediction</h3><pre>"
            f"{html.escape(row['prediction'] or '')}</pre></section></div>"
            "<details><summary>Metrics and details</summary><pre>"
            f"{_pretty({'metrics': row['metrics'], 'details': row['details']})}"
            f'</pre></details><p class="meta">Latency: {row["latency_seconds"]}s · '
            f"Error: {error}</p></article>"
        )
    aggregate = _pretty(metrics["aggregate_metrics"])
    rubric_aggregate = _pretty(metrics["rubric_metrics"])
    css = """
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #f3f6fa; color: #172033; }
    main { max-width: 1200px; margin: auto; padding: 28px 18px 60px; }
    .hero, .case { background: white; border: 1px solid #dbe3ee;
      border-radius: 14px; box-shadow: 0 8px 24px #14203712; }
    .hero { padding: 24px; }
    .summary { display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 10px; margin: 16px 0; }
    .summary div { background: #f8fafc; border-radius: 10px; padding: 12px; }
    .summary strong { display: block; font-size: 24px; }
    .toolbar { position: sticky; top: 0; padding: 10px 0;
      background: #f3f6faf2; }
    button { border: 1px solid #cbd5e1; border-radius: 999px;
      background: white; padding: 7px 12px; cursor: pointer; }
    button.active { background: #172033; color: white; }
    .case { margin-top: 14px; overflow: hidden; }
    .case.passed { border-left: 5px solid #15803d; }
    .case.failed { border-left: 5px solid #b42318; }
    .case header { display: flex; justify-content: space-between;
      padding: 12px 14px; background: #f8fafc; }
    .columns, .media { display: grid; grid-template-columns: repeat(2, 1fr); }
    .columns section, details, .meta { padding: 12px 14px; }
    .rubrics { padding: 12px 14px; border-top: 1px solid #e2e8f0; }
    .rubric { margin: 8px 0; padding: 10px 12px; border-radius: 9px;
      background: #f8fafc; border-left: 4px solid #15803d; }
    .rubric.failed { border-left-color: #b42318; }
    .rubric header { padding: 0; background: transparent; }
    .rubric p, .rubric ul { margin: 7px 0; }
    .failure { color: #b42318; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; }
    figure { margin: 0; padding: 10px; border-top: 1px solid #e2e8f0; }
    img { width: 100%; max-height: 420px; object-fit: contain; background: #111827; }
    figcaption, .meta { color: #64748b; font-size: 12px; }
    .hidden { display: none; }
    @media (max-width: 700px) {
      .summary, .columns, .media { grid-template-columns: 1fr; }
    }
    """
    script = """
    const cards = [...document.querySelectorAll('.case')];
    const buttons = [...document.querySelectorAll('button[data-filter]')];
    function apply(filter) {
      cards.forEach(card => card.classList.toggle('hidden',
        filter !== 'all' && card.dataset.status !== filter));
      buttons.forEach(button => button.classList.toggle('active',
        button.dataset.filter === filter));
    }
    buttons.forEach(button => button.addEventListener('click',
      () => apply(button.dataset.filter)));
    apply('failed');
    """
    document = f"""
    <!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Inference evaluation report</title><style>{css}</style></head><body>
    <main><section class="hero"><h1>Inference evaluation report</h1>
    <p>{html.escape(str(metrics["dataset_path"]))}</p>
    <div class="summary"><div>Total<strong>{metrics["total"]}</strong></div>
    <div>Passed<strong>{metrics["passed"]}</strong></div>
    <div>Failed<strong>{metrics["failed"]}</strong></div>
    <div>Pass rate<strong>{metrics["pass_rate"] * 100:.1f}%</strong></div></div>
    <details><summary>Aggregate metrics</summary><pre>{aggregate}</pre></details>
    <details><summary>Rubric metrics</summary><pre>{rubric_aggregate}</pre></details>
    </section><div class="toolbar"><button data-filter="all">All</button>
    <button data-filter="failed">Failed</button>
    <button data-filter="passed">Passed</button></div>{"".join(cards)}</main>
    <script>{script}</script></body></html>
    """
    output.write_text(document, encoding="utf-8")


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _rescore(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        evaluation = _validate_evaluation(
            _agent_rubric_evaluation(row["reference"], row["prediction"], config)
        )
        return {**row, **evaluation, "error": None}
    except (TypeError, ValueError, KeyError) as exc:
        return {**row, "error": f"{type(exc).__name__}: {exc}"}


def evaluate_inference(
    endpoint: str,
    model: str,
    model_reference: str,
    dataset_path: str,
    output_dir: str,
    dataset_config_json: str,
    evaluation_config_json: str,
    *,
    workers: int = 4,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_seconds: int = 240,
    max_retries: int = 3,
    limit: int = 0,
) -> dict[str, Any]:
    """Execute frozen rubrics, preserving predictions across identical runs."""
    if workers < 1 or max_tokens < 1 or timeout_seconds < 1 or max_retries < 0:
        raise ValueError("invalid inference request settings")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    dataset_file = Path(dataset_path).resolve()
    config = _json_object(dataset_config_json, "dataset config")
    evaluation = _json_object(evaluation_config_json, "evaluation config")
    if evaluation.get("mode") != _AGENT_RUBRIC_MODE:
        raise ValueError("only agent_rubric evaluation is supported")
    _configured_rubrics(evaluation)
    rows = _load_jsonl(dataset_file)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("dataset contains no evaluation cases")
    ids = [_case_id(row, config, index) for index, row in enumerate(rows, 1)]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset case IDs must be unique")
    media_digests = {}
    for row in rows:
        for value in _input_media(row, config):
            if value.startswith(("http://", "https://")):
                raise ValueError("evaluation media must be frozen Storage files")
            if not value.startswith("data:"):
                media = _media_path(value, dataset_file, config)
                media_digests[str(media)] = _file_digest(media)
    manifest = {
        "schema_version": 1,
        "dataset_sha256": _file_digest(dataset_file),
        "media_sha256": media_digests,
        "dataset_config": config,
        "evaluation_config": evaluation,
        "model": model,
        "model_reference": model_reference,
        "runtime_sha256": _file_digest(Path(__file__)),
        "generation": {
            "workers": workers,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "limit": limit,
        },
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("evaluation inputs changed; start a new full run")
    else:
        _atomic_json(manifest_path, manifest)
    checkpoint = output / "predictions.partial.jsonl"
    completed = (
        {row["id"]: row for row in _load_checkpoint(checkpoint)}
        if checkpoint.exists()
        else {}
    )
    reused = {
        key: _rescore(completed[key], evaluation)
        for key in ids
        if _has_reusable_prediction(completed.get(key))
    }
    completed = dict(reused)
    pending = [
        (index, row)
        for index, row in enumerate(rows, 1)
        if _case_id(row, config, index) not in reused
    ]
    ledger = output / "requests.jsonl"
    request_history = _load_checkpoint(ledger)
    previous_requests = len(request_history)
    _write_jsonl(ledger, request_history)
    _write_jsonl(checkpoint, list(completed.values()))
    current_requests = 0
    started = time.monotonic()
    lock = threading.Lock()
    _atomic_json(output / "report.json", {"status": "running"})

    def record_request() -> None:
        nonlocal current_requests
        with lock:
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"at": time.time()}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            current_requests += 1

    def write_progress() -> None:
        elapsed = time.monotonic() - started
        done = list(completed.values())
        succeeded = sum(row.get("error") is None for row in done)
        rate = max(0, len(done) - len(reused)) / elapsed if elapsed else 0
        _atomic_json(
            output / "progress.json",
            {
                "total": len(rows),
                "processed": len(done),
                "succeeded": succeeded,
                "failed": len(done) - succeeded,
                "elapsed_ms": round(elapsed * 1000),
                "records_per_second": rate,
                "eta_ms": round((len(rows) - len(done)) / rate * 1000)
                if rate
                else None,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    _write_jsonl(output / "processed.jsonl", list(reused.values()))
    write_progress()
    builder = _default_request
    endpoint_url = _chat_completions_endpoint(endpoint)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _evaluate_case,
                index,
                row,
                dataset_file,
                config,
                evaluation,
                builder,
                endpoint_url,
                model,
                "",
                max_tokens,
                temperature,
                timeout_seconds,
                max_retries,
                record_request,
            )
            for index, row in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed[row["id"]] = row
            for path in (checkpoint, output / "processed.jsonl"):
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            write_progress()
    results = [completed[key] for key in ids]
    metrics = _metrics(
        results,
        {
            "model": model,
            "model_reference": model_reference,
            "dataset_path": str(dataset_file),
            "evaluation_mode": "agent_rubric",
        },
    )
    metrics.update(
        {
            "request_count": previous_requests + current_requests,
            "current_request_count": current_requests,
            "reused_predictions": len(reused),
            "successful_predictions": sum(
                _has_reusable_prediction(row) for row in results
            ),
            "evaluated_cases": sum(row.get("error") is None for row in results),
        }
    )
    for name in ("predictions.jsonl", "processed.jsonl"):
        _write_jsonl(output / name, results)
    _atomic_json(output / "metrics.json", metrics)
    report_path = output / "evaluation_report.html"
    valid = metrics["successful_predictions"] > 0 and metrics["evaluated_cases"] > 0
    if valid:
        _build_html(results, metrics, dataset_file, config, report_path)
    error = None if valid else "inference produced no successfully scored predictions"
    storage_dir = str(output).removeprefix("/workspace")
    summary = {
        "status": "succeeded" if valid else "failed",
        "metrics": metrics,
        "error": error,
        "artifacts": {
            name: f"{storage_dir}/{name}"
            for name in ("metrics.json", "predictions.jsonl", "evaluation_report.html")
            if (output / name).exists()
        },
    }
    _atomic_json(output / "report.json", summary)
    if not valid:
        raise RuntimeError(error)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="default")
    parser.add_argument("--model-reference", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    try:
        config = json.loads(Path(args.evaluation_config).read_text())
        result = evaluate_inference(
            endpoint=args.endpoint,
            model=args.model,
            model_reference=args.model_reference,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            dataset_config_json=Path(args.dataset_config).read_text(),
            evaluation_config_json=json.dumps(config),
            **config.get("generation", {}),
        )
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            output / "report.json",
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
