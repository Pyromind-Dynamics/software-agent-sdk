"""Frozen positional Dataset recipe assembly, independent of product persistence."""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Literal

from pydantic import BaseModel

from pyromind_common.platform.paths import to_agent_storage_path
from pyromind_common.platform.scene_options import BatchLabelSource


class RecipeAsset(BaseModel):
    role: Literal["prepare", "pipeline", "export", "support_file", "verifier"]
    path: str
    sha256: str


class RecipeReceipt(BaseModel):
    schema_version: Literal[1] = 1
    source_path: str
    output_directory: str
    command_template: str
    assets: tuple[RecipeAsset, ...]
    batch_label_source: BatchLabelSource | None = None
    fixed_validation_path: str | None = None
    replay_of_task_id: str | None = None
    uses_labeling_gateway: bool


RECIPE_INPUT = "/__pyromind_recipe_input__"


RECIPE_OUTPUT = "/__pyromind_recipe_output__"


_RECIPE_COMMENT = "# pyromind-dataset-recipe-v1 "
_GATEWAY_EXPORT = 'export DF_API_KEY="$param"'
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"\b(?:DF_API_KEY|PYROMIND_API_KEY|PYROMIND_AUTH_TOKEN|LLM_AUTH_TOKEN|WANDB_API_KEY)\s*="
)
_REPLAY_LIMIT = (
    "awk 'NF && ++n > 5 { exit 1 } END { if (!n) exit 1 }' "
    '"$recipe_input"'
    ' || { echo "Replay requires 1-5 manifest rows" >&2; exit 1; }'
)


def dataset_public_profile(values: Mapping[str, str]) -> dict[str, str]:
    return {
        name: values[name]
        for name in ("DF_API_URL", "DF_API_BASE_URL", "DF_MODEL_NAME")
        if values.get(name)
    }


def dataset_gateway_exports(public_profile: Mapping[str, str]) -> list[str]:
    return [
        "unset DF_API_KEY DF_API_URL DF_API_BASE_URL DF_MODEL_NAME",
        *(
            f"export {name}={shlex.quote(value)}"
            for name, value in public_profile.items()
        ),
    ]


def _development_command(receipt: RecipeReceipt) -> str:
    fields = []
    for _, field, spec, conversion in Formatter().parse(receipt.command_template):
        if field is None:
            continue
        if field not in {"source_batch", "output_uri"} or spec or conversion:
            raise ValueError("Invalid Dataset recipe template field")
        fields.append(field)
    if set(fields) != {"source_batch", "output_uri"}:
        raise ValueError("Missing Dataset recipe inputs")
    command = receipt.command_template.format(
        source_batch=shlex.quote(receipt.source_path),
        output_uri=shlex.quote(receipt.output_directory),
    )
    if receipt.replay_of_task_id:
        # A development replay checks up to five rows before DataFlow calls the model.
        before, pipeline_start, rest = command.partition("\n(\n")
        if not pipeline_start:
            raise ValueError("Replay recipe has no pipeline step")
        return before + "\n" + _REPLAY_LIMIT + pipeline_start + rest
    return command


def read_prepared_dataset_recipe(command: str) -> RecipeReceipt | None:
    if not command.startswith(_RECIPE_COMMENT):
        return None
    header, newline, body = command.partition("\n")
    metadata = header.removeprefix(_RECIPE_COMMENT)
    try:
        receipt = RecipeReceipt.model_validate_json(metadata)
        rendered = _development_command(receipt)
    except (AttributeError, IndexError, KeyError, ValueError):
        raise ValueError("Invalid SDK Dataset recipe declaration") from None
    if not newline or metadata != receipt.model_dump_json() or body != rendered:
        raise ValueError("Dataset command does not match its declared recipe")
    allowed = _GATEWAY_EXPORT + "\n"
    if rendered.count(allowed) != int(
        receipt.uses_labeling_gateway
    ) or _CREDENTIAL_ASSIGNMENT.search(rendered.replace(allowed, "")):
        raise ValueError("Dataset recipes can only refer to the worker credential")
    return receipt


def recipe_asset(
    role: Literal["prepare", "pipeline", "export", "support_file", "verifier"],
    local: Path,
    asset_root: str,
) -> RecipeAsset:
    with local.open("rb") as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    filename = "support.json" if role == "support_file" else f"{role}.py"
    return RecipeAsset(
        role=role,
        path=to_agent_storage_path(str(PurePosixPath(asset_root, digest, filename))),
        sha256=digest,
    )


def compose_recipe(
    *,
    input_path: str,
    output_dir: str,
    assets: tuple[RecipeAsset, ...],
    dataflow_command: str,
    llm_env: dict[str, str],
    fixed_validation_path: str | None,
    replay_of_task_id: str | None,
    batch_label_source: BatchLabelSource | None = None,
    development_environment: str | None = None,
    prepared_gateway: bool | None = None,
) -> tuple[RecipeReceipt, str]:
    source_path = to_agent_storage_path(input_path)
    output_directory = to_agent_storage_path(output_dir)
    key = llm_env.get("DF_API_KEY")
    gateway = bool(key) if prepared_gateway is None else prepared_gateway
    public_profile = dataset_public_profile(llm_env)
    if gateway and not all(
        public_profile.get(k) for k in ("DF_API_URL", "DF_MODEL_NAME")
    ):
        raise ValueError("Dataset replay needs a stable DataFlow endpoint and model")
    if any(
        "?" in url or "@" in url
        for name, url in public_profile.items()
        if "URL" in name
    ):
        raise ValueError(
            "Dataset replay requires an endpoint without embedded credentials"
        )
    prefix = [
        f"export {name}={shlex.quote(value)}" for name, value in public_profile.items()
    ]
    if prepared_gateway is not None and public_profile:
        prefix = dataset_gateway_exports(public_profile)
    if gateway:
        prefix += [_GATEWAY_EXPORT, "unset param"]

    paths = {}
    checks = {}
    body = [
        'recipe_source="/target-workspace${source_path#/workspace}"',
        'recipe_output="/target-workspace${output_path#/workspace}"',
        'mkdir -p "$recipe_output"',
    ]
    for asset in assets:
        mounted = "/target-workspace" + asset.path.removeprefix("/workspace")
        path = shlex.quote(mounted)
        paths[asset.role] = path
        checks[asset.role] = f'test "$(sha256sum {path} | cut -c1-64)" = {asset.sha256}'
        body.append(checks[asset.role])
    body.append(f'cp {paths["pipeline"]} "$recipe_output/pipeline.py"')
    if "support_file" in paths:
        body.append(f'cp {paths["support_file"]} "$recipe_output/job-spec.json"')
    if "prepare" in paths:
        body += [
            'recipe_input="$recipe_output/manifest.jsonl"',
            f'python3 {paths["prepare"]} "$recipe_source" "$recipe_input"',
        ]
    else:
        body.append('recipe_input="$recipe_source"')
    pipeline = dataflow_command.replace(
        f"/target-workspace{RECIPE_OUTPUT}", '"$recipe_output"'
    ).replace(f"/target-workspace{RECIPE_INPUT}", '"$recipe_input"')
    pipeline_step = f"(\n{pipeline}\n)"
    body.append(pipeline_step)
    fixed = ""
    if fixed_validation_path:
        fixed = shlex.quote(
            "/target-workspace" + fixed_validation_path.removeprefix("/workspace")
        )
    body.append(
        f"/tmp/df-venv/bin/python {paths['export']} "
        '"$recipe_output/processed.jsonl" "$recipe_output"'
        + (f" {fixed}" if fixed else "")
    )
    body.append(checks["verifier"])
    body.append(
        f"/tmp/df-venv/bin/python {paths['verifier']} "
        '--source-batch "$recipe_source" '
        '--output-dir "$recipe_output"'
        + (f" --fixed-validation {fixed}" if fixed else "")
    )

    def escape(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    template = (
        escape("\n".join([*prefix, "set -eu"]))
        + "\nsource_path={source_batch}\n"
        + "output_path={output_uri}\n"
        + escape("\n".join(body))
    )
    receipt = RecipeReceipt(
        source_path=source_path,
        output_directory=output_directory,
        command_template=template,
        batch_label_source=batch_label_source,
        assets=assets,
        fixed_validation_path=fixed_validation_path,
        uses_labeling_gateway=gateway,
        replay_of_task_id=replay_of_task_id,
    )
    development = _development_command(receipt)
    if prepared_gateway is not None:
        return receipt, _RECIPE_COMMENT + receipt.model_dump_json() + "\n" + development
    if key:
        development = development.replace(
            'export DF_API_KEY="$param"',
            development_environment
            if development_environment is not None
            else f"export DF_API_KEY={shlex.quote(key)}",
            1,
        )
    elif development_environment is not None:
        development = development_environment + "\n" + development
    return receipt, development
