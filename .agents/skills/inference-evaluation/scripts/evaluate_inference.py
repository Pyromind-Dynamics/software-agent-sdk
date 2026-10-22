#!/usr/bin/env python3
"""Run dataset-driven evaluation against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
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


def _bbox_counts(
    prediction: Any, reference: Any, threshold: float
) -> tuple[int, int, int] | None:
    if not isinstance(prediction, list) or not isinstance(reference, list):
        return None
    predicted = [box for item in prediction if (box := _box_coordinates(item))]
    expected = [box for item in reference if (box := _box_coordinates(item))]
    if len(predicted) != len(prediction) or len(expected) != len(reference):
        return None
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
    return matches, len(predicted), len(expected)


def _bbox_score(prediction: Any, reference: Any, threshold: float) -> float:
    counts = _bbox_counts(prediction, reference, threshold)
    if counts is None:
        return 0.0
    matches, predicted, expected = counts
    return 2 * matches / (predicted + expected) if predicted + expected else 1.0


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


def _classification_metrics(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    # Missing/invalid predictions remain in GT support, so failed requests and
    # malformed answers cannot improve recall by disappearing from the sample.
    valid = [
        (actual, expected) for actual, expected in pairs if isinstance(expected, str)
    ]
    supports = Counter(expected for _, expected in valid)
    predictions = Counter(actual for actual, _ in valid if isinstance(actual, str))
    labels = sorted(supports)
    confusion = {label: {label: 0} for label in labels}
    per_class = {}
    for actual, expected in valid:
        if isinstance(actual, str):
            confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
    for label in labels:
        support = supports[label]
        predicted = predictions[label]
        correct = confusion[label][label]
        per_class[label] = {
            "support": support,
            "precision": correct / predicted if predicted else 0.0,
            "recall": correct / support,
            "f1": 2 * correct / (predicted + support),
            "unclassified": support - sum(confusion[label].values()),
            "prediction_rates": {
                key: count / support for key, count in confusion[label].items()
            },
        }
    return {
        "labels": labels,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "accuracy": (
            sum(actual == expected for actual, expected in valid) / len(valid)
            if valid
            else None
        ),
        "macro_f1": (
            statistics.mean(value["f1"] for value in per_class.values())
            if per_class
            else None
        ),
        "excluded_references": len(pairs) - len(valid),
    }


def _localization_metrics(
    pairs: list[tuple[Any, Any]], threshold: float
) -> dict[str, Any]:
    matched = predicted = expected = invalid = excluded = 0
    for actual, reference in pairs:
        if not isinstance(reference, list) or any(
            _box_coordinates(box) is None for box in reference
        ):
            excluded += 1
            continue
        counts = _bbox_counts(actual, reference, threshold)
        if counts is None:
            invalid += 1
            counts = (0, len(actual) if isinstance(actual, list) else 0, len(reference))
        matches, prediction_count, reference_count = counts
        matched += matches
        predicted += prediction_count
        expected += reference_count
    return {
        "iou_threshold": threshold,
        "matched_boxes": matched,
        "predicted_boxes": predicted,
        "reference_boxes": expected,
        "precision": matched / predicted if predicted else 0.0,
        "recall": matched / expected if expected else 0.0,
        "f1": 2 * matched / (predicted + expected) if predicted + expected else 0.0,
        "invalid_predictions": invalid,
        "excluded_references": excluded,
    }


def _metrics(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    rubrics: list[dict[str, Any]],
) -> dict[str, Any]:
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
    for rubric in rubrics:
        evaluator = rubric["evaluator"]
        kind = evaluator["type"]
        pairs = [
            (
                _path_value(
                    _structured_value(row.get("prediction")),
                    evaluator.get("prediction_path", ""),
                )
                if row.get("error") is None
                else _MISSING,
                _path_value(
                    _structured_value(row.get("reference")),
                    evaluator.get("reference_path", ""),
                ),
            )
            for row in rows
        ]
        summary = rubric_metrics.setdefault(
            rubric["name"], {"count": 0, "mean_score": None, "pass_rate": None}
        )
        if kind == "field_equals" and any(isinstance(ref, str) for _, ref in pairs):
            summary["classification"] = _classification_metrics(pairs)
        elif kind == "bbox_iou":
            summary["localization"] = _localization_metrics(
                pairs, float(evaluator.get("iou_threshold", 0.5))
            )
        elif kind == "json_valid":
            summary["valid_rate"] = (
                sum(
                    isinstance(_structured_value(row.get("prediction")), dict)
                    for row in rows
                )
                / total
                if total
                else 0.0
            )
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


REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inference evaluation report</title>
<style>__REPORT_CSS__</style>
</head>
<body>
<main id="report"></main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>__REPORT_JS__</script>
<script>
window.EvaluationReport.mount(document.getElementById("report"), {
  dataElement: document.getElementById("report-data")
});
</script>
</body>
</html>
"""

REPORT_ASSET_NAMES = ("evaluation_report.css", "evaluation_report.js")


def _read_report_asset(name: str, assets_dir: Path | None = None) -> str:
    directory = assets_dir if assets_dir is not None else Path(__file__).parent
    return (directory / name).read_text(encoding="utf-8")


def _runtime_sha256(assets_dir: Path | None = None) -> str:
    digest = hashlib.sha256()
    for name in sorted((Path(__file__).name, *REPORT_ASSET_NAMES)):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        path = (
            assets_dir / name
            if assets_dir is not None and name in REPORT_ASSET_NAMES
            else Path(__file__).with_name(name)
        )
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


TEMPLATE_REGISTRY: dict[str, dict[str, Any]] = {
    "evaluation-basic-v1": {
        "template_version": "1.2.0",
        "required_capabilities": ("structured_comparison",),
        "allowed_panels": ("rubric", "annotation", "raw_output"),
    },
    "evaluation-media-v1": {
        "template_version": "1.2.0",
        "required_capabilities": ("structured_comparison", "media_gallery"),
        "allowed_panels": ("rubric", "annotation", "raw_output", "media_compare"),
    },
}
# Only machine-failed cases are reviewable in this release; the field exists so
# a future review_scope="all" needs no annotation-record migration.
REVIEW_SCOPE = "machine_failed"
ANNOTATION_PRODUCT_STEMS = ("evaluation_annotations", "labeled_evaluation_results")
REPORT_SPEC_FIELDS = frozenset(
    {
        "template_id",
        "template_version",
        "title",
        "enabled_panels",
        "media_roles",
        "annotation_schema",
        "review_scope",
    }
)


def _report_case(
    row: dict[str, Any],
    dataset_file: Path,
    dataset_config: dict[str, Any],
    include_media_urls: bool,
    validation_errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    media: list[dict[str, Any]] = []
    for value in row.get("media", []):
        if not include_media_urls:
            media.append({"caption": value})
            continue
        try:
            url: str | None = _media_url(value, dataset_file, dataset_config)
        except (OSError, ValueError):
            url = None
        if url is None:
            warnings.append(
                f"case {row['id']}: media {value} could not be read;"
                " showing the media path only"
            )
            media.append({"caption": value})
        else:
            media.append({"url": url, "caption": value})
    return {
        "id": row["id"],
        "status": "passed" if row["passed"] else "failed",
        "input": {
            field: row.get("sample", {}).get(field)
            for key in ("messages_field", "user_prompt_field", "system_prompt_field")
            if isinstance(field := dataset_config.get(key), str)
            and field in row.get("sample", {})
        }
        or row.get("sample", {}),
        "media": media,
        "rubric_items": [
            {
                "name": rubric["name"],
                "passed": bool(rubric["passed"]),
                "score": rubric["score"],
                "weight": rubric["weight"],
                "criterion": rubric["criterion"],
                "evidence": list(rubric.get("evidence", [])),
                "failure_reason": rubric.get("failure_reason") or "",
            }
            for rubric in row.get("rubric_results", [])
        ],
        "overall_score": row.get("overall_score"),
        "reference": row["reference"],
        "prediction": row["prediction"],
        "detail": {"metrics": row["metrics"], "details": row["details"]},
        "latency_seconds": row["latency_seconds"],
        "error": row["error"],
        "validation_errors": validation_errors,
    }


def _report_capabilities(rows: list[dict[str, Any]]) -> set[str]:
    capabilities = {"structured_comparison"}
    if any(row.get("media") for row in rows):
        capabilities.add("media_gallery")
    return capabilities


def _case_validation_errors(
    row: dict[str, Any], rubrics: list[dict[str, Any]]
) -> list[str]:
    """Reference/GT-side extraction and type checks for one evaluated case.

    Execution-error rows are reviewable badcases rather than config defects, so
    they yield nothing here. Prediction-side misses reflect model output quality
    (already visible as ``<missing>`` evidence and ``passed=False``) and do not
    invalidate the report; only the fixed GT/reference contract does.
    """
    if row.get("error") is not None:
        return []
    reference_root = _structured_value(row.get("reference"))
    errors: list[str] = []
    for rubric in rubrics:
        evaluator = rubric["evaluator"]
        kind = evaluator.get("type")
        name = rubric["name"]
        reference_path = evaluator.get("reference_path", "")
        expected = _path_value(reference_root, reference_path)
        if reference_path and expected is _MISSING:
            errors.append(
                f"rubric '{name}': reference_path '{reference_path}' did not"
                " resolve (reference <missing>)"
            )
        elif kind in ("list_count_match", "bbox_iou") and not isinstance(
            expected, list
        ):
            errors.append(
                f"rubric '{name}': {kind} expects a list reference at"
                f" '{reference_path or '<root>'}', got {type(expected).__name__}"
            )
    return errors


def _prediction_path_errors(
    results: list[dict[str, Any]], rubrics: list[dict[str, Any]]
) -> list[str]:
    """Flag prediction paths that never resolve, indicating a misconfigured rubric.

    A prediction path missing on some cases is ordinary model variance; missing
    on every evaluated case means the configured path does not match the model
    output contract.
    """
    evaluated = [row for row in results if row.get("error") is None]
    if not evaluated:
        return []
    errors: list[str] = []
    for rubric in rubrics:
        evaluator = rubric["evaluator"]
        kind = evaluator.get("type")
        prediction_path = evaluator.get("prediction_path", "")
        if not prediction_path or kind in ("exact_match", "json_valid"):
            continue
        if all(
            _path_value(_structured_value(row.get("prediction")), prediction_path)
            is _MISSING
            for row in evaluated
        ):
            errors.append(
                f"rubric '{rubric['name']}': prediction_path '{prediction_path}'"
                f" never resolved across {len(evaluated)} cases (likely"
                " misconfigured)"
            )
    return errors


def _evaluation_status(
    results: list[dict[str, Any]],
    rubrics: list[dict[str, Any]],
    metrics: dict[str, Any],
    predictions_rows: int,
) -> tuple[str, dict[str, list[str]], list[str]]:
    """Derive validated/invalid plus per-case and report-level validation errors.

    Completeness compares the accounted rows (evaluated plus execution errors)
    and the on-disk predictions.jsonl line count against the result row count,
    so execution-error cases stay reviewable instead of invalidating the report.
    """
    case_errors = {
        row["id"]: errors
        for row in results
        if (errors := _case_validation_errors(row, rubrics))
    }
    report_errors = _prediction_path_errors(results, rubrics)
    total = len(results)
    if predictions_rows != total:
        report_errors.append(
            f"completeness: predictions.jsonl has {predictions_rows} rows,"
            f" expected {total}"
        )
    accounted = metrics["evaluated_cases"] + metrics["api_or_evaluation_errors"]
    if accounted != total:
        report_errors.append(
            f"completeness: evaluated_cases ({metrics['evaluated_cases']}) +"
            f" errors ({metrics['api_or_evaluation_errors']}) != {total} rows"
        )
    status = "invalid" if case_errors or report_errors else "validated"
    return status, case_errors, report_errors


def _spec_for(template_id: str) -> dict[str, Any]:
    entry = TEMPLATE_REGISTRY[template_id]
    return {
        "template_id": template_id,
        "template_version": entry["template_version"],
        "title": "Inference evaluation report",
        "enabled_panels": list(entry["allowed_panels"]),
    }


def _default_spec(capabilities: set[str]) -> dict[str, Any]:
    return _spec_for(
        "evaluation-media-v1"
        if "media_gallery" in capabilities
        else "evaluation-basic-v1"
    )


def _valid_labels(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def _resolve_report_spec(
    report_spec_json: str | None, capabilities: set[str]
) -> tuple[dict[str, Any], list[str]]:
    """Validate the agent-authored spec; any violation falls back to basic.

    A missing spec keeps the capability-derived default (media when the results
    carry media, otherwise basic); a present-but-invalid spec always falls back
    to ``evaluation-basic-v1`` so an unrequested panel is never auto-enabled.
    """
    if not report_spec_json:
        return _default_spec(capabilities), []
    try:
        value = json.loads(report_spec_json)
    except json.JSONDecodeError as exc:
        return _spec_for("evaluation-basic-v1"), [
            f"report spec is not valid JSON ({exc}); using default template"
        ]
    if not isinstance(value, dict):
        return _spec_for("evaluation-basic-v1"), [
            "report spec must be a JSON object; using default template"
        ]
    unknown = sorted(set(value) - REPORT_SPEC_FIELDS)
    if unknown:
        return _spec_for("evaluation-basic-v1"), [
            f"report spec has unknown fields {unknown}; using default template"
        ]
    template_id = value.get("template_id")
    if not isinstance(template_id, str) or not template_id.strip():
        return _spec_for("evaluation-basic-v1"), [
            f"report spec template id {template_id!r} is not a non-empty"
            " string; using default template"
        ]
    entry = TEMPLATE_REGISTRY.get(template_id)
    if entry is None:
        return _spec_for("evaluation-basic-v1"), [
            f"report spec template {template_id!r} is not registered;"
            " using default template"
        ]
    version = value.get("template_version")
    if version is not None and version != entry["template_version"]:
        return _spec_for("evaluation-basic-v1"), [
            f"report spec template version {version!r} does not match"
            " the registry; using default template"
        ]
    missing = [
        capability
        for capability in entry["required_capabilities"]
        if capability not in capabilities
    ]
    if missing:
        return _spec_for("evaluation-basic-v1"), [
            f"report spec template {template_id!r} requires missing"
            f" capabilities {missing}; using default template"
        ]
    panels = value.get("enabled_panels")
    if panels is None:
        panels = list(entry["allowed_panels"])
    if (
        not isinstance(panels, list)
        or not all(isinstance(panel, str) for panel in panels)
        or not set(panels) <= set(entry["allowed_panels"])
    ):
        return _spec_for("evaluation-basic-v1"), [
            "report spec has unregistered panels; using default template"
        ]
    title = value.get("title", "Inference evaluation report")
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        return _spec_for("evaluation-basic-v1"), [
            "report spec title must be a non-empty string of at most"
            " 200 characters; using default template"
        ]
    media_roles = value.get("media_roles", {})
    if not isinstance(media_roles, dict) or not all(
        isinstance(key, str) and isinstance(role, str) and role.strip()
        for key, role in media_roles.items()
    ):
        return _spec_for("evaluation-basic-v1"), [
            "report spec media roles must map strings to strings;"
            " using default template"
        ]
    annotation_schema = value.get("annotation_schema", {})
    if (
        not isinstance(annotation_schema, dict)
        or not set(annotation_schema)
        <= {
            "taxonomy_labels",
            "dispositions",
        }
        or not all(_valid_labels(items) for items in annotation_schema.values())
    ):
        return _spec_for("evaluation-basic-v1"), [
            "report spec annotation schema allows only string lists under"
            " taxonomy_labels and dispositions; using default template"
        ]
    review_scope = value.get("review_scope", REVIEW_SCOPE)
    if review_scope != REVIEW_SCOPE:
        return _spec_for("evaluation-basic-v1"), [
            f"report spec review_scope {review_scope!r} is not supported"
            f" (only {REVIEW_SCOPE!r}); using default template"
        ]
    spec = {
        "template_id": template_id,
        "template_version": entry["template_version"],
        "title": title,
        "enabled_panels": list(panels),
        "review_scope": review_scope,
    }
    if media_roles:
        spec["media_roles"] = media_roles
    if annotation_schema:
        spec["annotation_schema"] = annotation_schema
    return spec, []


REPORT_HTML_NAME = "evaluation_report.html"
REPORT_CASES_MANIFEST_NAME = "evaluation_cases_manifest.json"
REPORT_CASES_INDEX_NAME = "evaluation_cases_index.json"
REPORT_CASES_DIRECTORY = "evaluation_cases"
REPORT_MEDIA_DIRECTORY = "evaluation_media"
REPORT_CASE_CHUNK_SIZE = 20


def _media_suffix(mime_type: str) -> str:
    return {
        "image/gif": ".gif",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime_type, ".bin")


def _externalize_report_media(
    case_payload: dict[str, Any],
    *,
    case_index: int,
    media_output: Path,
) -> dict[str, Any]:
    media_items: list[dict[str, Any]] = []
    for media_index, media in enumerate(case_payload.get("media", [])):
        external = dict(media)
        url = media.get("url")
        if isinstance(url, str) and url.startswith("data:"):
            header, separator, encoded = url.partition(",")
            mime_type = header.removeprefix("data:").removesuffix(";base64")
            if separator and header.endswith(";base64"):
                media_id = (
                    f"{case_index:08d}-{media_index:03d}{_media_suffix(mime_type)}"
                )
                (media_output / media_id).write_bytes(
                    base64.b64decode(encoded, validate=True)
                )
                external.pop("url", None)
                external.update({"media_id": media_id, "mime_type": mime_type})
        media_items.append(external)
    return {**case_payload, "media": media_items}


def _write_report_case_products(
    case_payloads: list[dict[str, Any]], output: Path
) -> None:
    cases_output = output / REPORT_CASES_DIRECTORY
    media_output = output / REPORT_MEDIA_DIRECTORY
    cases_output.mkdir(exist_ok=True)
    media_output.mkdir(exist_ok=True)

    externalized = [
        _externalize_report_media(
            case_payload,
            case_index=case_index,
            media_output=media_output,
        )
        for case_index, case_payload in enumerate(case_payloads)
    ]
    chunks: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(
        range(0, len(externalized), REPORT_CASE_CHUNK_SIZE)
    ):
        items = externalized[start : start + REPORT_CASE_CHUNK_SIZE]
        name = f"chunk-{chunk_index:05d}.json"
        _atomic_json(cases_output / name, {"items": items})
        chunks.append(
            {
                "name": f"{REPORT_CASES_DIRECTORY}/{name}",
                "start": start,
                "count": len(items),
            }
        )
        index.extend(
            {
                "id": item["id"],
                "chunk": chunk_index,
                "offset": offset,
                "media": [
                    {
                        "media_id": media["media_id"],
                        "mime_type": media["mime_type"],
                    }
                    for media in item.get("media", [])
                    if "media_id" in media and "mime_type" in media
                ],
            }
            for offset, item in enumerate(items)
        )

    _atomic_json(
        output / REPORT_CASES_INDEX_NAME,
        {"schema_version": 1, "cases": index},
    )
    _atomic_json(
        output / REPORT_CASES_MANIFEST_NAME,
        {
            "schema_version": 1,
            "total": len(externalized),
            "chunk_size": REPORT_CASE_CHUNK_SIZE,
            "chunks": chunks,
        },
    )


def _build_html(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    dataset_file: Path,
    dataset_config: dict[str, Any],
    output: Path,
    spec: dict[str, Any],
    evaluation_status: str,
    case_errors: dict[str, list[str]],
    run_id: str,
    evaluation_config_sha256: str,
    assets_dir: Path | None = None,
) -> list[str]:
    """Render the single-file badcase review report; return warnings.

    Only machine-failed cases (including execution-error rows) are rendered:
    annotation targets badcases, and skipping passed cases keeps the page
    small and quick to load. Media are embedded as original bytes; files
    that cannot be read degrade to a path caption.
    """
    include_media_urls = "media_compare" in spec["enabled_panels"]
    warnings: list[str] = []
    badcases = [row for row in rows if not row["passed"]]
    case_payloads = [
        _report_case(
            row,
            dataset_file,
            dataset_config,
            include_media_urls,
            case_errors.get(row["id"], []),
            warnings,
        )
        for row in badcases
    ]
    _write_report_case_products(case_payloads, output)
    # allow_nan=False turns a NaN/Infinity anywhere in the payload into an
    # explicit generation failure instead of a report page that dies in
    # JSON.parse and renders blank.
    payload = (
        json.dumps(
            {
                "title": spec["title"],
                "template_id": spec["template_id"],
                "template_version": spec["template_version"],
                "panels": spec["enabled_panels"],
                "evaluation_status": evaluation_status,
                "run_id": run_id,
                "evaluation_config_sha256": evaluation_config_sha256,
                "review_scope": spec.get("review_scope", REVIEW_SCOPE),
                "annotation_schema": spec.get("annotation_schema", {}),
                "dataset_path": str(metrics["dataset_path"]),
                "summary": {
                    "total": metrics["total"],
                    "passed": metrics["passed"],
                    "failed": metrics["failed"],
                    "pass_rate": metrics["pass_rate"],
                },
                "dataset_distribution": metrics.get("dataset_distribution", {}),
                "review_only": metrics.get("review_only", False),
                "metrics_available": metrics.get("metrics_available", True),
                "classification": metrics.get("classification"),
                "original_label_metrics": metrics.get("original_label_metrics"),
                "integrity_issues": metrics.get("integrity_issues", []),
                "aggregate_metrics": metrics["aggregate_metrics"],
                "rubric_metrics": metrics["rubric_metrics"],
                "mean_latency_seconds": metrics.get("mean_latency_seconds"),
                "case_delivery": "paged",
                "case_count": len(case_payloads),
                "cases": [],
                "warnings": warnings,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    document = (
        REPORT_TEMPLATE.replace(
            "__REPORT_CSS__", _read_report_asset("evaluation_report.css", assets_dir)
        )
        .replace(
            "__REPORT_JS__", _read_report_asset("evaluation_report.js", assets_dir)
        )
        .replace("__REPORT_DATA__", payload)
    )
    (output / REPORT_HTML_NAME).write_bytes(document.encode("utf-8"))
    for stale in output.glob("evaluation_report_page_*.html"):
        stale.unlink()
    return warnings


def canonical_annotation_jsonl(records: list[dict[str, Any]]) -> bytes:
    if not records:
        return b""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _machine_result(row: dict[str, Any]) -> dict[str, Any]:
    """Authoritative machine result for one case, rebuilt from predictions."""
    return {
        "passed": bool(row.get("passed")),
        "reference": row.get("reference"),
        "prediction": row.get("prediction"),
        "overall_score": row.get("overall_score"),
        "failure_tags": [
            rubric["name"]
            for rubric in row.get("rubric_results") or []
            if isinstance(rubric, dict)
            and rubric.get("name")
            and not rubric.get("passed")
        ],
    }


def _initial_annotation_records(
    rows: list[dict[str, Any]], run_id: str, evaluation_config_sha256: str
) -> list[dict[str, Any]]:
    """Revision-0 annotation records: the failed set, all unreviewed.

    Bytes are a pure function of predictions (no timestamps), so re-running
    the evaluation reproduces identical r0 objects.
    """
    return [
        {
            "schema_version": 1,
            "run_id": run_id,
            "evaluation_config_sha256": evaluation_config_sha256,
            "case_id": row["id"],
            "review_status": "unreviewed",
            "machine_result": _machine_result(row),
            "human_annotation": None,
        }
        for row in rows
        if not row["passed"]
    ]


def _labeled_result_records(
    rows: list[dict[str, Any]], run_id: str, evaluation_config_sha256: str
) -> list[dict[str, Any]]:
    """Revision-0 merged records covering the full evaluation set.

    Each record is the complete original prediction row (inputs, rubric
    results, errors, media, timings) plus run binding and the human-review
    slot — not a reduced machine-result summary.
    """
    return [
        {
            **row,
            "run_id": run_id,
            "evaluation_config_sha256": evaluation_config_sha256,
            "human_review": {
                "review_status": (
                    "unreviewed" if not row["passed"] else "not_in_scope"
                ),
                "annotation": None,
            },
        }
        for row in rows
    ]


def _write_initial_annotation_products(
    output: Path,
    results: list[dict[str, Any]],
    run_id: str,
    evaluation_config_sha256: str,
) -> None:
    """Seed revision-0 annotation objects plus canonical-name views.

    The ``*.r0.jsonl`` objects are the authoritative layer the annotation
    save API builds on; the canonical-name files are convenience views
    refreshed on every commit by the server. report.json deliberately does
    not list them (revision-suffixed names change with every save).
    """
    # Heuristic guard: the eval pod cannot read the Product Store commit
    # state, so file existence is the only signal available here. Any r{N>0}
    # object — committed or left behind by a failed save — means the save
    # service owns these artifacts; skipping is the conservative direction
    # (init only matters for fresh runs) and never overwrites anything.
    present = [
        int(path.stem.rsplit(".r", 1)[1])
        for path in output.glob("evaluation_annotations.r*.jsonl")
    ]
    if present and max(present) > 0:
        return
    annotation_records = _initial_annotation_records(
        results, run_id, evaluation_config_sha256
    )
    labeled_records = _labeled_result_records(results, run_id, evaluation_config_sha256)
    annotation_jsonl = canonical_annotation_jsonl(annotation_records)
    labeled_jsonl = canonical_annotation_jsonl(labeled_records)
    snapshot = {
        "schema_version": 1,
        "run_id": run_id,
        "evaluation_config_sha256": evaluation_config_sha256,
        "revision": 0,
        "exported_at": datetime.now(UTC).isoformat(),
        "review_scope": REVIEW_SCOPE,
        "total": len(annotation_records),
        "reviewed": 0,
        "in_progress": 0,
        "unreviewed": len(annotation_records),
        "annotations": annotation_records,
    }
    (output / "evaluation_annotations.r0.jsonl").write_bytes(annotation_jsonl)
    (output / "labeled_evaluation_results.r0.jsonl").write_bytes(labeled_jsonl)
    (output / "evaluation_annotations.jsonl").write_bytes(annotation_jsonl)
    (output / "labeled_evaluation_results.jsonl").write_bytes(labeled_jsonl)
    (output / "evaluation_annotations.json").write_bytes(
        (json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _config_digest(value: dict[str, Any]) -> str:
    canonical = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
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
    run_id: str,
    report_spec_json: str | None = None,
    report_assets_dir: str | None = None,
    workers: int = 4,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_seconds: int = 240,
    max_retries: int = 3,
    limit: int = 0,
    business_scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute frozen rubrics, preserving predictions across identical runs."""
    if not run_id:
        raise ValueError("run_id is required")
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
    business_cases = {}
    business_module = None
    if business_scoring is not None:
        import types

        from evaluate_predictions import stable_case, validate_script

        if (
            limit
            or _file_digest(dataset_file) != business_scoring["test_source"]["sha256"]
        ):
            raise ValueError("final test must use the complete frozen test dataset")
        business_module = types.ModuleType("frozen_test_scoring")
        # This entry point runs in the existing CPU worker, never in Runtime/HTTP.
        exec(
            compile(business_scoring["script"], "score_eval.py", "exec"),
            business_module.__dict__,
        )
        validate_script(business_module, business_scoring["spec"])
        for identity, raw in zip(ids, rows, strict=True):
            case = stable_case(raw, business_module, {"source": "test"})
            if case["id"] != identity:
                raise ValueError(
                    "test mapping and frozen scoring script disagree on IDs"
                )
            business_cases[identity] = case
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
        "run_id": run_id,
        "evaluation_config_sha256": _config_digest(
            business_scoring["spec"] if business_scoring else evaluation
        ),
        "dataset_sha256": _file_digest(dataset_file),
        "media_sha256": media_digests,
        "dataset_config": config,
        "evaluation_config": evaluation,
        "model": model,
        "model_reference": model_reference,
        "runtime_sha256": _runtime_sha256(
            Path(report_assets_dir) if report_assets_dir is not None else None
        ),
        "generation": {
            "workers": workers,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "limit": limit,
        },
        **(
            {"business_scoring": business_scoring, "dataset_role": "test"}
            if business_scoring is not None
            else {}
        ),
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
    if business_module is not None and business_scoring is not None:
        from evaluate_predictions import score

        rescored = []
        for row in results:
            case = {
                **business_cases[row["id"]],
                "prediction": None if row.get("error") else row["prediction"],
                "media": row.get("media", []),
                "sample": row.get("sample", {}),
            }
            result = score(case, business_module, business_scoring["spec"])
            rescored.append(
                {
                    **row,
                    **result,
                    "error": row.get("error"),
                    "latency_seconds": row.get("latency_seconds"),
                    "rubric_results": result["rubric_results"],
                }
            )
        results = rescored
    metrics = _metrics(
        results,
        {
            "model": model,
            "model_reference": model_reference,
            "dataset_path": str(dataset_file),
            "evaluation_mode": "agent_rubric",
        },
        [] if business_scoring else _configured_rubrics(evaluation),
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
    if business_scoring is not None:
        metrics.update(
            dataset_role="test",
            selection_run_id=business_scoring["selection_run_id"],
            scoring_spec_sha256=_config_digest(business_scoring["spec"]),
            classification=_classification_metrics(
                [
                    (row.get("prediction_label"), row.get("reference_label"))
                    for row in results
                ]
            ),
        )
        metrics["macro_f1"] = metrics["classification"]["macro_f1"]
    return write_evaluation_products(
        results,
        metrics,
        dataset_file,
        config,
        output,
        run_id,
        manifest["evaluation_config_sha256"],
        [] if business_scoring else _configured_rubrics(evaluation),
        report_spec_json,
        report_assets_dir,
    )


def write_evaluation_products(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    dataset_file: Path,
    config: dict[str, Any],
    output: Path,
    run_id: str,
    evaluation_config_sha256: str,
    rubrics: list[dict[str, Any]],
    report_spec_json: str | None = None,
    report_assets_dir: str | None = None,
    integrity_issues: list[str] | None = None,
    render_invalid: bool = False,
    render_report: bool = True,
) -> dict[str, Any]:
    """Write scores and optionally render a report; never initialize human review."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.jsonl", "processed.jsonl"):
        _write_jsonl(output / name, results)
    _atomic_json(output / "metrics.json", metrics)
    valid = metrics["evaluated_cases"] > 0 and (
        metrics["successful_predictions"] > 0
        or metrics.get("evaluation_mode") == "saved_predictions"
    )
    with (output / "predictions.jsonl").open(encoding="utf-8") as stream:
        predictions_rows = sum(1 for line in stream if line.strip())
    evaluation_status, case_errors, report_errors = _evaluation_status(
        results, rubrics, metrics, predictions_rows
    )
    if not valid or integrity_issues:
        evaluation_status = "invalid"
    report_errors.extend(integrity_issues or [])
    badcases = [row for row in results if not row["passed"]]
    spec, report_warnings = _resolve_report_spec(
        report_spec_json, _report_capabilities(badcases)
    )
    # Human editing belongs to independently created review tasks.
    spec["enabled_panels"] = [
        panel for panel in spec["enabled_panels"] if panel != "annotation"
    ]
    report_status = "not_requested"
    try:
        if render_report and (valid or render_invalid):
            report_warnings.extend(
                _build_html(
                    results,
                    metrics,
                    dataset_file,
                    config,
                    output,
                    spec,
                    evaluation_status,
                    case_errors,
                    run_id,
                    evaluation_config_sha256,
                    Path(report_assets_dir) if report_assets_dir is not None else None,
                )
            )
            _atomic_json(output / "resolved_report_spec.json", spec)
        if render_report and (valid or render_invalid):
            report_status = "ready"
    except Exception as exc:
        report_status = "failed"
        report_errors.append(f"report_render_error:{type(exc).__name__}:{exc}")
    error = None if valid else "inference produced no successfully scored predictions"
    storage_dir = str(output).removeprefix("/workspace")
    summary = {
        "status": "succeeded" if valid else "failed",
        "evaluation_status": evaluation_status,
        "report_status": report_status,
        "metrics": metrics,
        "error": error,
        "report_spec": {
            "template_id": spec["template_id"],
            "template_version": spec["template_version"],
        },
        "badcase_count": len(badcases),
        "validation": {
            "invalid_cases": sorted(case_errors),
            "case_errors": case_errors,
            "report_errors": report_errors,
        },
        "warnings": report_warnings,
        "artifacts": {
            name: f"{storage_dir}/{name}"
            for name in (
                "metrics.json",
                "predictions.jsonl",
                "resolved_report_spec.json",
                REPORT_HTML_NAME,
                REPORT_CASES_MANIFEST_NAME,
                REPORT_CASES_INDEX_NAME,
            )
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-spec")
    parser.add_argument("--report-assets-dir")
    parser.add_argument("--business-scoring")
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
            run_id=args.run_id,
            report_assets_dir=args.report_assets_dir,
            business_scoring=(
                json.loads(Path(args.business_scoring).read_text())
                if args.business_scoring
                else None
            ),
            report_spec_json=(
                Path(args.report_spec).read_text() if args.report_spec else None
            ),
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
