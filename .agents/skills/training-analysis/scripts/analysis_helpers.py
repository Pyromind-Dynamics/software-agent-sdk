"""Data-source neutral diagnostics preserving metric and training step identity."""

from __future__ import annotations

import math
import statistics
from typing import Any


TRAINING_STEP_KEYS = ("trainer/global_step", "train/global_step", "global_step")


def metric_series(history: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    points = []
    for row in history:
        if metric not in row or row[metric] is None:
            continue
        raw = row[metric]
        if isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (ValueError, TypeError):
            continue
        step_key = next(
            (
                key
                for key in TRAINING_STEP_KEYS
                if isinstance(row.get(key), (int, float))
                and not isinstance(row[key], bool)
                and math.isfinite(row[key])
                and row[key] >= 0
                and int(row[key]) == row[key]
            ),
            None,
        )
        points.append(
            {
                "step": int(row[step_key]) if step_key else None,
                "step_key": step_key,
                "log_step": row.get("_step"),
                "value": value if math.isfinite(value) else None,
                "nonfinite": str(value) if not math.isfinite(value) else None,
            }
        )
    return points


def diagnose_run(
    data: dict[str, Any],
    train_key: str = "loss",
    val_key: str | None = "val_loss",
    max_steps: int | None = None,
) -> dict[str, Any]:
    points = metric_series(data["history"], train_key)[:max_steps]
    finite = [point for point in points if point["value"] is not None]
    if not points:
        return {
            "error": f"No observations for {train_key}",
            "summary_value": data["summary"].get(train_key),
        }
    best = min(finite, key=lambda point: point["value"]) if finite else None
    values = [point["value"] for point in finite]
    result = {
        "total_steps": len(values),
        "observations": len(points),
        "final_value": values[-1] if values else None,
        "min_value": best["value"] if best else None,
        "min_value_step": best["step"] if best else None,
        "min_value_log_step": best["log_step"] if best else None,
        "has_nan": any(point["nonfinite"] == "nan" for point in points),
        "has_inf": any(point["nonfinite"] in {"inf", "-inf"} for point in points),
        "step_alignment": "training_step"
        if all(p["step"] is not None for p in points)
        else "unverified",
        "final_10pct_mean": statistics.mean(values[-max(1, len(values) // 10) :])
        if values
        else None,
    }
    if len(values) > 100:
        tail = values[-max(1, len(values) // 10) :]
        result["converged"] = (
            statistics.pstdev(tail) < max(abs(statistics.mean(tail)), 1e-12) * 0.01
        )
    if val_key:
        train = {p["step"]: p["value"] for p in finite if p["step"] is not None}
        validation = [
            p
            for p in metric_series(data["history"], val_key)
            if p["value"] is not None and p["step"] in train
        ]
        if len(validation) > 10:
            tail = validation[-max(1, len(validation) // 5) :]
            gap = statistics.mean(p["value"] - train[p["step"]] for p in tail)
            result["train_val_gap"] = gap
            result["likely_overfit"] = (
                gap > abs(statistics.mean(train[p["step"]] for p in tail)) * 0.2
            )
    return result
