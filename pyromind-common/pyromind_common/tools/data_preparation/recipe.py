"""Compatibility models for the shared frozen Dataset recipe builder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pyromind_common.platform.recipe import (
    RECIPE_INPUT as RECIPE_INPUT,
    RECIPE_OUTPUT as RECIPE_OUTPUT,
    RecipeAsset,
    compose_recipe as _compose_recipe,
    recipe_asset as _recipe_asset,
)
from pyromind_runtime.domain.workflow_execution import (
    BatchLabelSource,
    DatasetRecipeAsset,
    DatasetRecipeReceipt,
)


def recipe_asset(
    role: Literal["prepare", "pipeline", "export", "support_file", "verifier"],
    local: Path,
    asset_root: str,
) -> DatasetRecipeAsset:
    return DatasetRecipeAsset.model_validate(
        _recipe_asset(role, local, asset_root).model_dump()
    )


def compose_recipe(
    *,
    input_path: str,
    output_dir: str,
    assets: tuple[DatasetRecipeAsset, ...],
    dataflow_command: str,
    llm_env: dict[str, str],
    fixed_validation_path: str | None,
    replay_of_task_id: str | None,
    batch_label_source: BatchLabelSource | None = None,
) -> tuple[DatasetRecipeReceipt, str]:
    receipt, command = _compose_recipe(
        input_path=input_path,
        output_dir=output_dir,
        assets=tuple(
            RecipeAsset.model_validate(asset.model_dump()) for asset in assets
        ),
        dataflow_command=dataflow_command,
        llm_env=llm_env,
        fixed_validation_path=fixed_validation_path,
        replay_of_task_id=replay_of_task_id,
        batch_label_source=batch_label_source,
    )
    return DatasetRecipeReceipt.model_validate(receipt.model_dump()), command
