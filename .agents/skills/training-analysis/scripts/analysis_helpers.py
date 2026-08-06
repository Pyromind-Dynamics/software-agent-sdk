# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from wandb-primary skill (wandb_helpers_impl.py).
# https://github.com/coreweave/skills/tree/main/skills/wandb-primary

"""数据源无关的训练分析函数。

所有函数只消费标准 ``RunData``(见 data_sources/base.py)或纯 dict,
不依赖任何具体数据源 SDK。新增数据源(如 SwanLab)无需改动本模块。

原 scan_history / probe_project 依赖 wandb API 对象,已迁至
data_sources/wandb.py(WandbDataSource)数据获取层。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from data_sources.base import RunData


def diagnose_run(
    data: RunData,
    train_key: str = "loss",
    val_key: str | None = "val_loss",
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Quick diagnostic summary of a training run.

    Checks for convergence, overfitting, NaN values, and other common
    training issues. Consumes only RunData (summary + history rows).

    Args:
        data: Standard run data (data_sources.base.RunData).
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
    available_keys = set(data["summary"])
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

    rows = [r for r in data["history"] if r.get("_step") is not None][:max_steps]
    if not rows:
        return {
            "error": "No history rows found",
            "summary_value": data["summary"].get(train_key),
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
