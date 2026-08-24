#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible inference endpoint and build an HTML report."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import difflib
import html
import json
import math
import mimetypes
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


LABEL_KEYS = ("result", "label", "answer", "prediction", "class", "category")
BOX_KEYS = ("boxes", "bboxes", "bbox")
VALIDATION_NAMES = ("val.jsonl", "validation.jsonl", "eval.jsonl", "test.jsonl")
MEDIA_TYPES = {"image", "image_url", "input_image"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="INFERENCE_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--task-type",
        choices=("auto", "classification", "generation", "structured"),
        default="auto",
    )
    parser.add_argument("--labels", help="Comma-separated classification labels")
    parser.add_argument("--title")
    parser.add_argument("--report-name", default="validation_report.html")
    parser.add_argument("--id-field")
    parser.add_argument("--gt-field")
    parser.add_argument("--system-field")
    parser.add_argument("--user-field")
    parser.add_argument("--images-field")
    parser.add_argument("--messages-field")
    parser.add_argument("--box-image-index", type=int, default=0)
    parser.add_argument("--box-scale", type=float)
    parser.add_argument("--responses-jsonl", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def resolve_dataset(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return path, path.parent
    if not path.is_dir():
        raise FileNotFoundError(path)
    matches = [path / name for name in VALIDATION_NAMES if (path / name).is_file()]
    if not matches:
        matches = sorted(path.glob("*.jsonl"))
    if len(matches) != 1:
        names = ", ".join(str(item) for item in matches) or "none"
        raise ValueError(f"expected one validation JSONL in {path}; found {names}")
    return matches[0], path


def first_field(
    row: dict[str, Any], explicit: str | None, names: tuple[str, ...]
) -> str | None:
    if explicit:
        return explicit if explicit in row else None
    return next((name for name in names if name in row), None)


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return "" if value is None else str(value)
    parts = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
            text = item.get("text", item.get("value", ""))
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def structured_object(text: str) -> dict[str, Any] | None:
    objects = json_objects(text)
    return objects[-1] if objects else None


def answer_value(text: str) -> str:
    obj = structured_object(text)
    if obj:
        for key in LABEL_KEYS:
            value = obj.get(key)
            if isinstance(value, (str, int, float, bool)):
                return str(value).strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def token_list(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[\w]+", normalize_text(text))


def token_f1(reference: str, prediction: str) -> float:
    ref = Counter(token_list(reference))
    pred = Counter(token_list(prediction))
    overlap = sum((ref & pred).values())
    if not ref and not pred:
        return 1.0
    precision = overlap / sum(pred.values()) if pred else 0.0
    recall = overlap / sum(ref.values()) if ref else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalize_boxes(value: Any) -> list[list[float]]:
    if isinstance(value, list) and len(value) == 4 and not isinstance(value[0], list):
        value = [value]
    if not isinstance(value, list):
        return []
    boxes = []
    for box in value:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            coords = [float(item) for item in box]
        except (TypeError, ValueError):
            continue
        if coords[0] < coords[2] and coords[1] < coords[3]:
            boxes.append(coords)
    return boxes


def boxes_from_text(text: str) -> list[list[float]]:
    obj = structured_object(text)
    if not obj:
        return []
    for key in BOX_KEYS:
        boxes = normalize_boxes(obj.get(key))
        if boxes:
            return boxes
    return []


def path_from_media_part(part: dict[str, Any]) -> str | None:
    value = part.get("image", part.get("value", part.get("url")))
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        value = image_url
    elif isinstance(image_url, dict):
        value = image_url.get("url")
    return value if isinstance(value, str) else None


def media_paths_from_content(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    paths = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in MEDIA_TYPES:
            value = path_from_media_part(part)
            if value:
                paths.append(value)
    return paths


def resolve_media(value: str, root: Path) -> str:
    if value.startswith(("data:", "http://", "https://")):
        return value
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.resolve())


def data_uri(value: str) -> str:
    if value.startswith(("data:", "http://", "https://")):
        return value
    path = Path(value)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def normalize_content(content: Any, root: Path) -> Any:
    if not isinstance(content, list):
        return content
    normalized = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in MEDIA_TYPES:
            normalized.append(part)
            continue
        path_value = path_from_media_part(part)
        if not path_value:
            normalized.append(part)
            continue
        resolved = resolve_media(path_value, root)
        normalized.append(
            {"type": "image_url", "image_url": {"url": data_uri(resolved)}}
        )
    return normalized


def row_messages(
    row: dict[str, Any], root: Path, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], str, list[str], dict[str, str | None]]:
    messages_field = first_field(row, args.messages_field, ("messages", "conversation"))
    gt_field = first_field(
        row,
        args.gt_field,
        ("gt", "ground_truth", "expected", "answer", "label", "target"),
    )
    media_field = first_field(
        row, args.images_field, ("images", "image_paths", "image")
    )
    media_values = row.get(media_field, []) if media_field else []
    if isinstance(media_values, str):
        media_values = [media_values]
    top_level_media = [resolve_media(str(value), root) for value in media_values]
    message_media: list[str] = []

    messages_value = row.get(messages_field) if messages_field else None
    messages = []
    target = content_text(row.get(gt_field)) if gt_field else ""
    if isinstance(messages_value, list):
        for original in messages_value:
            if not isinstance(original, dict):
                continue
            role = str(original.get("role", "user"))
            content = original.get("content", "")
            message_media.extend(
                resolve_media(value, root)
                for value in media_paths_from_content(content)
            )
            if role == "assistant":
                target = content_text(content)
                continue
            messages.append({"role": role, "content": normalize_content(content, root)})
    else:
        system_field = first_field(row, args.system_field, ("system_prompt", "system"))
        user_field = first_field(
            row,
            args.user_field,
            ("user_prompt", "prompt", "instruction", "question", "input"),
        )
        if system_field:
            messages.append({"role": "system", "content": str(row[system_field])})
        user_text = str(row.get(user_field, "")) if user_field else ""
        messages.append({"role": "user", "content": user_text})

    media = top_level_media if top_level_media else message_media
    has_message_media = any(
        media_paths_from_content(message.get("content")) for message in messages
    )
    if media and not has_message_media:
        user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index]["role"] == "user"
            ),
            None,
        )
        if user_index is None:
            messages.append({"role": "user", "content": ""})
            user_index = len(messages) - 1
        text = content_text(messages[user_index]["content"])
        content = [
            {"type": "image_url", "image_url": {"url": data_uri(value)}}
            for value in media
        ]
        content.append({"type": "text", "text": text})
        messages[user_index]["content"] = content

    if not target:
        raise ValueError("could not find ground-truth response")
    fields = {
        "messages": messages_field,
        "ground_truth": gt_field or "assistant message",
        "media": media_field,
    }
    return messages, target, media, fields


def identify_task(targets: list[str], requested: str) -> tuple[str, list[str]]:
    if requested != "auto":
        labels = sorted({answer_value(target) for target in targets})
        return requested, labels if requested == "classification" else []
    objects = [structured_object(target) for target in targets]
    if sum(obj is not None for obj in objects) / len(targets) >= 0.8:
        answers = [answer_value(target) for target in targets]
        if len(set(answers)) <= min(30, max(2, math.ceil(len(targets) ** 0.5) * 2)):
            return "classification", sorted(set(answers))
        return "structured", []
    answers = [answer_value(target) for target in targets]
    short = all("\n" not in answer and len(answer) <= 80 for answer in answers)
    if short and len(set(answers)) <= min(
        30, max(2, math.ceil(len(targets) ** 0.5) * 2)
    ):
        return "classification", sorted(set(answers))
    return "generation", []


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no message")
    content = message.get("content")
    text = content_text(content)
    if text.strip():
        return text.strip()
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    raise ValueError("assistant response has no text")


def call_endpoint(
    payload: dict[str, Any], api_key: str, args: argparse.Namespace
) -> tuple[str, float, Any]:
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "inference-validation-report/1.0",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}") from exc
    return response_text(value), time.monotonic() - started, value.get("usage")


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_boxes(
    reference: list[list[float]], predicted: list[list[float]]
) -> list[float]:
    candidates = sorted(
        (
            (box_iou(left, right), left_index, right_index)
            for left_index, left in enumerate(reference)
            for right_index, right in enumerate(predicted)
        ),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = []
    for iou, left_index, right_index in candidates:
        if iou < 0.5:
            break
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append(iou)
    return matches


def score_output(
    target: str, output: str, task: str, labels: list[str]
) -> dict[str, Any]:
    target_answer = answer_value(target)
    output_answer = answer_value(output)
    target_obj = structured_object(target)
    output_obj = structured_object(output)
    target_boxes = boxes_from_text(target)
    predicted_boxes = boxes_from_text(output)
    matched_ious = match_boxes(target_boxes, predicted_boxes)
    normalized_labels = {normalize_text(label): label for label in labels}
    prediction = normalized_labels.get(normalize_text(output_answer))
    reference = normalized_labels.get(normalize_text(target_answer), target_answer)
    if task == "classification":
        correct = prediction == reference
    elif task == "structured":
        correct = target_obj == output_obj
    else:
        correct = normalize_text(target_answer) == normalize_text(output_answer)
    return {
        "reference": reference,
        "prediction": prediction if task == "classification" else output_answer,
        "correct": correct,
        "exact_match": target_answer == output_answer,
        "normalized_match": normalize_text(target_answer)
        == normalize_text(output_answer),
        "token_f1": token_f1(target_answer, output_answer),
        "similarity": difflib.SequenceMatcher(
            None, normalize_text(target_answer), normalize_text(output_answer)
        ).ratio(),
        "structured_valid": output_obj is not None,
        "structured_exact": target_obj is not None and target_obj == output_obj,
        "target_boxes": target_boxes,
        "predicted_boxes": predicted_boxes,
        "matched_ious": matched_ious,
    }


def evaluate_one(
    sample: dict[str, Any],
    api_key: str,
    args: argparse.Namespace,
    task: str,
    labels: list[str],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            output, latency, usage = call_endpoint(sample["payload"], api_key, args)
            return {
                **sample["public"],
                "raw_response": output,
                **score_output(sample["target"], output, task, labels),
                "latency_seconds": round(latency, 3),
                "usage": usage,
                "error": None,
            }
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(2 ** (attempt - 1))
    return {
        **sample["public"],
        "raw_response": None,
        **score_output(sample["target"], "", task, labels),
        "latency_seconds": None,
        "usage": None,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def prepare_samples(
    rows: list[dict[str, Any]], root: Path, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = []
    observed_fields: Counter[str] = Counter()
    for index, row in enumerate(rows):
        messages, target, media, fields = row_messages(row, root, args)
        observed_fields.update(value for value in fields.values() if value)
        id_field = first_field(row, args.id_field, ("id", "sample_id", "uid", "name"))
        sample_id = str(row.get(id_field, index)) if id_field else str(index)
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        samples.append(
            {
                "target": target,
                "payload": payload,
                "public": {
                    "id": sample_id,
                    "media": media,
                    "prompt": "\n\n".join(
                        content_text(message["content"])
                        for message in messages
                        if message["role"] != "system"
                    ),
                    "ground_truth_response": target,
                },
            }
        )
    profile = {
        "rows": len(samples),
        "rows_with_media": sum(bool(sample["public"]["media"]) for sample in samples),
        "media_count": sum(len(sample["public"]["media"]) for sample in samples),
        "observed_fields": dict(observed_fields),
    }
    return samples, profile


def classification_metrics(
    rows: list[dict[str, Any]], labels: list[str]
) -> dict[str, Any]:
    columns = [*labels, "invalid"]
    confusion = {label: {column: 0 for column in columns} for label in labels}
    for row in rows:
        prediction = row["prediction"] if row["prediction"] in labels else "invalid"
        confusion[row["reference"]][prediction] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][column] for column in columns if column != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "labels": labels,
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "macro_f1": statistics.mean(value["f1"] for value in per_class.values()),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "ground_truth_distribution": dict(Counter(row["reference"] for row in rows)),
        "prediction_distribution": dict(
            Counter(row["prediction"] or "invalid" for row in rows)
        ),
    }


def calculate_metrics(
    rows: list[dict[str, Any]], task: str, labels: list[str], profile: dict[str, Any]
) -> dict[str, Any]:
    latencies = [row["latency_seconds"] for row in rows if row["latency_seconds"]]
    metrics: dict[str, Any] = {
        "task_type": task,
        "total": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "api_errors": sum(row["error"] is not None for row in rows),
        "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "dataset_profile": profile,
    }
    if task == "classification":
        metrics.update(classification_metrics(rows, labels))
    else:
        metrics.update(
            {
                "exact_match": statistics.mean(row["exact_match"] for row in rows),
                "normalized_match": statistics.mean(
                    row["normalized_match"] for row in rows
                ),
                "mean_token_f1": statistics.mean(row["token_f1"] for row in rows),
                "mean_similarity": statistics.mean(row["similarity"] for row in rows),
                "structured_valid_rate": statistics.mean(
                    row["structured_valid"] for row in rows
                ),
            }
        )
        if task == "structured":
            metrics["structured_exact_match"] = statistics.mean(
                row["structured_exact"] for row in rows
            )
    gt_boxes = sum(len(row["target_boxes"]) for row in rows)
    predicted_boxes = sum(len(row["predicted_boxes"]) for row in rows)
    matched = sum(len(row["matched_ious"]) for row in rows)
    if gt_boxes or predicted_boxes:
        metrics["localization"] = {
            "iou_threshold": 0.5,
            "ground_truth_boxes": gt_boxes,
            "prediction_boxes": predicted_boxes,
            "matched_boxes": matched,
            "precision": matched / predicted_boxes if predicted_boxes else 0.0,
            "recall": matched / gt_boxes if gt_boxes else 0.0,
        }
    return metrics


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_cards(metrics: dict[str, Any]) -> str:
    if metrics["task_type"] == "classification":
        values = [
            (
                "准确率",
                percentage(metrics["accuracy"]),
                f"{metrics['correct']} / {metrics['total']}",
            ),
            ("Macro-F1", percentage(metrics["macro_f1"]), "各类别等权"),
        ]
        for label, value in metrics["per_class"].items():
            values.append(
                (
                    f"{label} Recall",
                    percentage(value["recall"]),
                    f"support {value['support']}",
                )
            )
    else:
        values = [
            ("Exact Match", percentage(metrics["exact_match"]), "原文完全一致"),
            (
                "Normalized Match",
                percentage(metrics["normalized_match"]),
                "忽略空白与大小写",
            ),
            ("Token-F1", percentage(metrics["mean_token_f1"]), "字符/词级重合"),
            ("相似度", percentage(metrics["mean_similarity"]), "SequenceMatcher"),
        ]
    values.extend(
        [
            (
                "API 成功率",
                percentage(1 - metrics["api_errors"] / metrics["total"]),
                f"错误 {metrics['api_errors']}",
            ),
            (
                "平均延迟",
                f"{metrics['mean_latency_seconds']:.2f}s"
                if metrics["mean_latency_seconds"]
                else "—",
                "每条样本",
            ),
        ]
    )
    if "localization" in metrics:
        loc = metrics["localization"]
        values.append(
            (
                "定位 Recall@0.5",
                percentage(loc["recall"]),
                f"{loc['matched_boxes']} / {loc['ground_truth_boxes']} boxes",
            )
        )
    return "".join(
        (
            '<article class="metric">'
            f"<span>{html.escape(name)}</span>"
            f"<strong>{value}</strong>"
            f"<small>{html.escape(note)}</small>"
            "</article>"
        )
        for name, value, note in values
    )


def classification_tables(metrics: dict[str, Any]) -> str:
    labels = metrics["labels"]
    rows = "".join(
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        f"<td>{value['support']}</td>"
        f"<td>{percentage(value['precision'])}</td>"
        f"<td>{percentage(value['recall'])}</td>"
        f"<td>{percentage(value['f1'])}</td>"
        "</tr>"
        for label, value in metrics["per_class"].items()
    )
    columns = [*labels, "invalid"]
    matrix_rows = "".join(
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        + "".join(
            (
                f'<td class="{"diag" if label == column else ""}">'
                f"{metrics['confusion_matrix'][label][column]}</td>"
            )
            for column in columns
        )
        + "</tr>"
        for label in labels
    )
    return (
        '<div class="tables"><div><h3>分类指标</h3><table><thead><tr>'
        "<th>类别</th><th>样本</th><th>Precision</th><th>Recall</th><th>F1</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
        "<div><h3>混淆矩阵</h3><table><thead><tr><th>GT ↓ / 预测 →</th>"
        + "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        + f"</tr></thead><tbody>{matrix_rows}</tbody></table></div></div>"
    )


def box_scale(boxes: list[list[float]], requested: float | None) -> float | None:
    if requested:
        return requested
    maximum = max((max(box) for box in boxes), default=0.0)
    if maximum <= 1:
        return 1.0
    if maximum <= 1000:
        return 1000.0
    return None


def box_overlays(boxes: list[list[float]], css_class: str, scale: float | None) -> str:
    if not scale:
        return ""
    overlays = []
    for box in boxes:
        left, top = box[0] / scale * 100, box[1] / scale * 100
        width, height = (box[2] - box[0]) / scale * 100, (box[3] - box[1]) / scale * 100
        overlays.append(
            f'<span class="box {css_class}" '
            f'style="left:{left:.3f}%;top:{top:.3f}%;'
            f'width:{width:.3f}%;height:{height:.3f}%"></span>'
        )
    return "".join(overlays)


def media_gallery(row: dict[str, Any], args: argparse.Namespace) -> str:
    figures = []
    boxes = [*row["target_boxes"], *row["predicted_boxes"]]
    scale = box_scale(boxes, args.box_scale)
    for index, value in enumerate(row["media"]):
        uri = data_uri(value)
        mime = uri[5:].split(";", 1)[0] if uri.startswith("data:") else "image"
        overlays = ""
        if index == args.box_image_index:
            overlays = box_overlays(row["target_boxes"], "gt", scale)
            overlays += box_overlays(row["predicted_boxes"], "pred", scale)
        if mime.startswith("video/"):
            media = f'<video controls preload="metadata" src="{uri}"></video>'
        else:
            media = (
                '<div class="media-wrap">'
                f'<img loading="lazy" src="{uri}">{overlays}</div>'
            )
        figures.append(
            f"<figure>{media}<figcaption>媒体 {index + 1}</figcaption></figure>"
        )
    return f'<div class="gallery">{"".join(figures)}</div>' if figures else ""


def sample_card(row: dict[str, Any], index: int, args: argparse.Namespace) -> str:
    status = "correct" if row["correct"] else "wrong"
    prediction = row["prediction"] if row["prediction"] is not None else "invalid"
    return (
        f'<article class="sample {status}" data-status="{status}"><header>'
        f"<b>#{index:03d}</b><code>{html.escape(row['id'])}</code>"
        f"<span>GT: {html.escape(str(row['reference']))}</span>"
        f"<span>预测: {html.escape(str(prediction))}</span>"
        f"<strong>{'正确' if row['correct'] else '不一致'}</strong></header>"
        + media_gallery(row, args)
        + '<div class="sample-grid"><details><summary>输入</summary><pre>'
        + html.escape(row["prompt"])
        + "</pre></details><details><summary>标准答案</summary><pre>"
        + html.escape(row["ground_truth_response"])
        + "</pre></details><details open><summary>模型输出</summary><pre>"
        + html.escape(row["raw_response"] or row["error"] or "—")
        + f"</pre></details></div><footer>Token-F1 {row['token_f1']:.3f} · "
        f"相似度 {row['similarity']:.3f} · "
        f"延迟 {row['latency_seconds'] or '—'}s</footer></article>"
    )


def build_html(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    dataset: Path,
) -> str:
    title = args.title or f"{args.model} 验证集评测"
    cards = "".join(
        sample_card(row, index, args)
        for index, row in enumerate(sorted(rows, key=lambda item: item["correct"]), 1)
    )
    tables = (
        classification_tables(metrics)
        if metrics["task_type"] == "classification"
        else ""
    )
    css = """
    :root { --bg:#f2f5fa; --card:#fff; --text:#172033; --muted:#657087;
      --line:#d9e1ec; --green:#16803d; --red:#c52b28; --blue:#263f78; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text);
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
        sans-serif; }
    main { max-width:1280px; margin:auto; padding:28px 18px 70px; }
    .hero { padding:42px; color:#fff; background:var(--blue); border-radius:36px; }
    h1 { margin:0 0 20px; font-size:42px; } .hero p { opacity:.85; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
      gap:14px; margin:22px 0; }
    .metric,.section,.sample { background:var(--card); border:1px solid var(--line);
      border-radius:18px; box-shadow:0 8px 24px rgba(20,35,60,.06); }
    .metric { padding:22px; } .metric span,.metric small { color:var(--muted); }
    .metric strong { display:block; margin:12px 0; font-size:34px; }
    .section { padding:22px; margin:18px 0; }
    .tables { display:grid; grid-template-columns:1fr 1fr; gap:22px; overflow:auto; }
    table { width:100%; border-collapse:collapse; } th,td { padding:9px;
      text-align:center; border-bottom:1px solid var(--line); }
    .diag { background:#e7f6eb; color:var(--green); font-weight:700; }
    .toolbar { position:sticky; top:0; z-index:5; padding:10px 0;
      background:rgba(242,245,250,.92); backdrop-filter:blur(8px); }
    button { padding:7px 13px; margin-right:6px; border:1px solid var(--line);
      border-radius:999px; background:#fff; cursor:pointer; }
    .sample { margin:16px 0; overflow:hidden; border-left:5px solid var(--green); }
    .sample.wrong { border-left-color:var(--red); }
    .sample header { display:flex; gap:12px; flex-wrap:wrap; align-items:center;
      padding:14px; background:#f8fafc; border-bottom:1px solid var(--line); }
    .sample header code { margin-right:auto; } .sample footer { padding:10px 14px;
      color:var(--muted); border-top:1px solid var(--line); }
    .gallery { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
      gap:1px; background:var(--line); }
    figure { margin:0; padding:12px; background:#fff; } img,video { width:100%;
      max-height:430px; object-fit:contain; background:#101625; border-radius:10px; }
    figcaption { text-align:center; color:var(--muted); margin-top:6px; }
    .media-wrap { position:relative; line-height:0; } .box { position:absolute;
      border:3px solid; pointer-events:none; } .box.gt { border-color:#16a34a; }
    .box.pred { border-color:#dc2626; }
    .sample-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1px;
      background:var(--line); } details { padding:14px; background:#fff; }
    summary { cursor:pointer; font-weight:700; } pre { white-space:pre-wrap;
      word-break:break-word; font:13px/1.55 ui-monospace,SFMono-Regular,monospace; }
    .hidden { display:none; }
    @media(max-width:800px) { .hero { padding:26px; } h1 { font-size:30px; }
      .tables,.sample-grid { grid-template-columns:1fr; } }
    """
    script = """
    const cards=[...document.querySelectorAll('.sample')];
    function filter(status){cards.forEach(card=>card.classList.toggle('hidden',
      status!=='all'&&card.dataset.status!==status));}
    """
    profile = metrics["dataset_profile"]
    metrics_json = html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))
    body = f"""
    <main><section class="hero"><h1>{html.escape(title)}</h1>
    <p>{html.escape(dataset.name)} · {metrics["total"]} 条样本 ·
    {html.escape(metrics["task_type"])} · 媒体样本 {profile["rows_with_media"]}</p>
    <p>模型：{html.escape(str(args.model))}　温度：{args.temperature}　
    生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}</p></section>
    <section class="metrics">{metric_cards(metrics)}</section>
    <section class="section"><h2>总体结果</h2>{tables}
    <pre>{metrics_json}</pre></section>
    <section class="section"><h2>逐样本复核</h2><div class="toolbar">
    <button onclick="filter('all')">全部</button>
    <button onclick="filter('wrong')">只看不一致</button>
    <button onclick="filter('correct')">只看一致</button></div>{cards}</section></main>
    """
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{css}</style></head>"
        f"<body>{body}<script>{script}</script></body></html>"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_offline_responses(path: Path) -> dict[str, str]:
    responses = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("id"))
        output = row.get("raw_response", row.get("response", row.get("output")))
        if isinstance(output, str):
            responses[sample_id] = output
    return responses


def main() -> None:
    args = parse_args()
    dataset, root = resolve_dataset(args.dataset)
    rows = load_jsonl(dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("validation dataset is empty")
    samples, profile = prepare_samples(rows, root, args)
    requested_labels = [
        item.strip() for item in (args.labels or "").split(",") if item.strip()
    ]
    task, labels = identify_task(
        [sample["target"] for sample in samples], args.task_type
    )
    if requested_labels:
        task, labels = "classification", requested_labels
    if task == "classification":
        label_map = {normalize_text(label): label for label in labels}
        for sample in samples:
            reference = answer_value(sample["target"])
            normalized = label_map.get(normalize_text(reference))
            if normalized is None:
                raise ValueError(f"unknown ground-truth label {reference!r}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_profile.json").write_text(
        json.dumps(
            {**profile, "task_type": task, "labels": labels},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint = args.output / "predictions.partial.jsonl"
    completed = (
        {row["id"]: row for row in load_jsonl(checkpoint)}
        if checkpoint.exists() and not args.no_resume
        else {}
    )
    offline = (
        load_offline_responses(args.responses_jsonl) if args.responses_jsonl else None
    )
    if offline is None:
        if not args.endpoint or not args.model:
            raise ValueError("--endpoint and --model are required")
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"environment variable {args.api_key_env} is required")
    else:
        api_key = ""

    pending = [sample for sample in samples if sample["public"]["id"] not in completed]
    lock = threading.Lock()

    def run(sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = sample["public"]["id"]
        if offline is not None:
            if sample_id not in offline:
                raise KeyError(f"offline response missing id {sample_id}")
            output = offline[sample_id]
            return {
                **sample["public"],
                "raw_response": output,
                **score_output(sample["target"], output, task, labels),
                "latency_seconds": None,
                "usage": None,
                "error": None,
            }
        return evaluate_one(sample, api_key, args, task, labels)

    print(
        f"task={task} total={len(samples)} completed={len(completed)} "
        f"pending={len(pending)}"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, sample): sample for sample in pending}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            with lock:
                with checkpoint.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed[result["id"]] = result
            print(
                f"[{index}/{len(pending)}] {result['id']} correct={result['correct']}"
            )

    results = [completed[sample["public"]["id"]] for sample in samples]
    write_jsonl(args.output / "predictions.jsonl", results)
    metrics = calculate_metrics(results, task, labels, profile)
    metrics.update(
        {
            "dataset": str(dataset),
            "endpoint": args.endpoint,
            "model": args.model,
            "temperature": args.temperature,
        }
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = build_html(results, metrics, args, dataset)
    (args.output / args.report_name).write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
