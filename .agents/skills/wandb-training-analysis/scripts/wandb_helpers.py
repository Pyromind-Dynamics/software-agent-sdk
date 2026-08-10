# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from wandb-primary skill (wandb_helpers_impl.py).
# https://github.com/coreweave/skills/tree/main/skills/wandb-primary

"""Helpers for W&B training data analysis, adapted from wandb-primary skill.

Key differences from the original:
- probe_project excludes Weave traces (irrelevant for Pyromind training analysis)
- All functions accept wandb.Api / Run objects directly
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Smart history scan
# ---------------------------------------------------------------------------


def scan_history(
    run: Any,
    keys: list[str],
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Read exact history rows from a run with explicit metric keys.

    IMPORTANT: keys is required. Never call without explicit keys on large
    projects — runs with 1K+ metrics will 502 or timeout without key filtering.

    Args:
        run: A W&B Run object.
        keys: Metric keys to fetch. REQUIRED.
        max_rows: Stop after this many rows. None = all rows.

    Returns:
        List of dicts with the requested keys + _step.
    """
    if not keys:
        raise ValueError(
            "keys is required — never scan without explicit keys on large projects"
        )

    rows: list[dict[str, Any]] = []
    scanner = run.scan_history(keys=keys, page_size=min(max_rows or 10_000, 10_000))
    for row in scanner:
        rows.append(dict(row))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# Run diagnostics
# ---------------------------------------------------------------------------


def diagnose_run(
    run: Any,
    train_key: str = "loss",
    val_key: str | None = "val_loss",
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Quick diagnostic summary of a training run.

    Checks for convergence, overfitting, NaN values, and other common
    training issues. Uses scan_history with explicit metric keys.

    Args:
        run: A W&B Run object from api.run().
        train_key: Primary training metric key (default "loss").
        val_key: Validation metric key (default "val_loss"). None to skip.
        max_steps: Limit rows read. None = all.

    Returns:
        Dict with diagnostic keys. Returns {"error": ...} if the
        requested keys don't exist.
    """
    import numpy as np
    import pandas as pd

    # Verify keys exist in summary before scanning history
    available_keys = set(run.summary_metrics.keys())
    if train_key not in available_keys:
        available = sorted(k for k in available_keys if not k.startswith("_"))[:20]
        return {
            "error": (f"Key '{train_key}' not in run summary. Available: {available}")
        }

    keys = [train_key]
    if val_key and val_key in available_keys:
        keys.append(val_key)
    elif val_key and val_key not in available_keys:
        val_key = None  # skip val check

    rows = scan_history(run, keys=keys, max_rows=max_steps)
    if not rows:
        return {
            "error": "No history rows found",
            "summary_value": run.summary_metrics.get(train_key),
        }

    df = pd.DataFrame(rows)
    if train_key not in df.columns:
        return {
            "error": f"Key '{train_key}' not in history columns: {list(df.columns)}"
        }

    loss = df[train_key].dropna()
    loss_arr = loss.to_numpy()

    diagnostics: dict[str, Any] = {
        "total_steps": len(loss),
        "final_value": float(loss_arr[-1]) if len(loss_arr) else None,
        "min_value": float(loss_arr.min()) if len(loss_arr) else None,
        "min_value_step": int(loss_arr.argmin()) if len(loss_arr) else None,
        "has_nan": bool(pd.isna(df[train_key]).any()),
        "final_10pct_mean": float(np.mean(loss_arr[-max(1, len(loss_arr) // 10) :]))
        if len(loss_arr)
        else None,
    }

    # Overfitting check
    if val_key and val_key in df.columns:
        val = df[val_key].dropna()
        val_arr = val.to_numpy()
        if len(val_arr) > 10:
            tail_size = max(1, len(val_arr) // 5)
            train_tail = float(np.mean(loss_arr[-tail_size:]))
            val_tail = float(np.mean(val_arr[-tail_size:]))
            diagnostics["train_val_gap"] = round(val_tail - train_tail, 6)
            diagnostics["likely_overfit"] = val_tail > train_tail * 1.2

    # Convergence check
    if len(loss_arr) > 100:
        last_pct = loss_arr[-max(1, len(loss_arr) // 10) :]
        diagnostics["converged"] = bool(np.std(last_pct) < np.mean(last_pct) * 0.01)

    return diagnostics


# ---------------------------------------------------------------------------
# Config comparison
# ---------------------------------------------------------------------------


def compare_configs(
    run_a: Any,
    run_b: Any,
    keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Side-by-side config comparison between two W&B runs.

    Args:
        run_a: First W&B Run object.
        run_b: Second W&B Run object.
        keys: Specific config keys to compare. None = all non-internal keys.

    Returns:
        List of dicts with differing keys and their values per run.
    """
    if keys is not None:
        config_a = {k: run_a.config.get(k) for k in keys}
        config_b = {k: run_b.config.get(k) for k in keys}
    else:
        config_a = {k: v for k, v in run_a.config.items() if not k.startswith("_")}
        config_b = {k: v for k, v in run_b.config.items() if not k.startswith("_")}

    all_keys = sorted(set(config_a) | set(config_b))
    diffs = []
    for k in all_keys:
        val_a = config_a.get(k)
        val_b = config_b.get(k)
        if val_a != val_b:
            diffs.append(
                {
                    "key": k,
                    run_a.name: val_a,
                    run_b.name: val_b,
                }
            )
    return diffs


# ---------------------------------------------------------------------------
# Project probe
# ---------------------------------------------------------------------------


def probe_project(api: Any, path: str, sample_size: int = 3) -> dict[str, Any]:
    """Discover project characteristics before running queries.

    Call this FIRST on an unfamiliar project. It returns the project scale,
    available metric keys, config shape, and whether runs have step history.

    Args:
        api: wandb.Api instance.
        path: "entity/project" string.
        sample_size: Number of runs to sample for metric/config inspection.

    Returns:
        Dict with: run_count_estimate, sample_metrics, sample_config_keys,
        has_step_history, recommended_per_page, warnings.
    """
    result: dict[str, Any] = {"path": path, "warnings": []}

    runs = api.runs(
        path, filters={"state": "finished"}, order="-created_at", per_page=sample_size
    )
    sample = runs[:sample_size]
    if not sample:
        result["run_count_estimate"] = 0
        result["warnings"].append("No finished runs found")
        return result

    all_metric_keys: set[str] = set()
    all_config_keys: set[str] = set()
    has_history = False

    for run in sample:
        metric_keys = {k for k in run.summary_metrics.keys() if not k.startswith("_")}
        config_keys = {k for k in run.config.keys() if not k.startswith("_")}
        all_metric_keys |= metric_keys
        all_config_keys |= config_keys
        if getattr(run, "lastHistoryStep", -1) >= 0:
            has_history = True

    n_metrics = len(all_metric_keys)
    result["sample_metric_count"] = n_metrics
    result["sample_metric_keys"] = sorted(all_metric_keys)[:50]
    result["sample_config_keys"] = sorted(all_config_keys)[:50]
    result["has_step_history"] = has_history

    if n_metrics > 500:
        result["warnings"].append(
            f"Runs have {n_metrics} metrics — ALWAYS pass keys= to history/scan_history"
        )
    if n_metrics > 5000:
        result["warnings"].append(
            f"Runs have {n_metrics} metrics — history() without keys WILL 502"
        )

    if n_metrics > 1000:
        result["recommended_per_page"] = 10
    elif n_metrics > 100:
        result["recommended_per_page"] = 50
    else:
        result["recommended_per_page"] = 100

    return result
