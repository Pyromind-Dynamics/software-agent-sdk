#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shlex
import tempfile
import zlib
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNotFoundError,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import (
    ProjectPublicationService,
    build_runtime_contract_draft,
)
from pyromind_common.tools.utils.training_source import (
    WANDB_BUILDER_NODE_TYPE,
    WANDB_BUILDER_NODE_TYPES,
)
from pyromind_common.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from pyromind_runtime.domain.storage_paths import normalize_workspace_path


SOURCE_PROJECT_ID = "6742edad-2fd1-419e-9e6c-979847463fe2"
SOURCE_CONVERSATION_ID = "375c0c97aa9b46a1a879effed4f0e904"
DATASET_TASK_ID = "10229"
DATASET_EXECUTION_ID = "5b20967e-a654-4b76-8b94-65381c429499"
DATASET_TRANSCRIPT = "9d413ece-151a-4919-af56-62cdee400df1_dataset_transcript.jsonl"
DATASET_PIPELINE_SHA256 = (
    "8393ce2afa871656db75df3a50fe4f8c46d34a7593c7bfe0325868b93ca9389d"
)
RAW_DATASET_PATH = "/workspace/datasets/hql_test/pcb_avi_recheck_0823_0828_test5"
RUN_OUTPUT_ROOT = (
    f"/workspace/.pyromind-agent/{SOURCE_CONVERSATION_ID}/echomind-runs/"
    "{graph_run_id}/{stage_run_id}"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an isolated EchoMind validation recipe from the 6742 runs."
        )
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workspace",
    )
    parser.add_argument("--source-project-id", default=SOURCE_PROJECT_ID)
    parser.add_argument("--train-task-id", default="10237")
    parser.add_argument(
        "--evaluation-workflow",
        type=Path,
        default=Path(
            "product-versions/snapshot-staging/"
            "64fe0a50cbb652566d07620f7dd333a2de669773c2f48f03ab4bd59601ddb87f/"
            ".pyromind-agent/375c0c97aa9b46a1a879effed4f0e904/versions/v52/"
            "f5ae11fe-be76-41d9-a6d3-1a43edac4e26/content/workflow.py"
        ),
    )
    parser.add_argument("--validation-key", default="pcb-avi-test5-platform-v5")
    parser.add_argument("--dataset-input", default=RAW_DATASET_PATH)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.train_task_id.isascii() or not args.train_task_id.isdecimal():
        parser.error("--train-task-id must be a numeric Studio task ID")

    workspace_root = args.workspace_root.resolve()
    projects_root = workspace_root / "projects"
    source = ProjectStore(projects_root).load(
        args.source_project_id,
        owner_user_id=None,
    )
    if source.project.conversation_id != SOURCE_CONVERSATION_ID:
        raise ValueError(
            "Source project conversation does not match the embedded artifact paths"
        )

    dataset_workflow = _recorded_workflow(
        workspace_root=workspace_root,
        conversation_id=source.project.conversation_id,
        task_id=DATASET_TASK_ID,
    )
    dataset_config = _dataset_runtime_config(
        dataset_workflow,
        _dataset_pipeline_source(workspace_root, source.project.conversation_id),
    )
    source_workflow = _recorded_workflow(
        workspace_root=workspace_root,
        conversation_id=source.project.conversation_id,
        task_id=args.train_task_id,
    )
    _replace_wandb_credentials_with_references(source_workflow)
    evaluation_dsl_path = _resolve_workspace_path(
        workspace_root,
        args.evaluation_workflow,
    )
    evaluation_workflow = convert_dsl_to_xyflow(
        evaluation_dsl_path.read_text(encoding="utf-8"),
        name="echomind-runtime-evaluation",
    )

    validation_id = str(
        uuid5(
            NAMESPACE_URL,
            f"echomind-runtime-validation-source:{args.validation_key}",
        )
    )
    nodes = [
        _validation_node(
            node=node,
            dataset_config=dataset_config,
            dataset_input=normalize_workspace_path(args.dataset_input),
            source_workflow=source_workflow,
            train_task_id=args.train_task_id,
            evaluation_workflow=evaluation_workflow,
            conversation_id=source.project.conversation_id,
        )
        for node in source.graph.nodes
    ]
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=validation_id,
            name=f"平台全量验证 · {args.validation_key}",
            description=(
                "Disposable production-runtime validation for Dataset, Train, "
                "and Evaluator."
            ),
            skill=source.project.skill,
            requirements=source.project.requirements,
            owner_user_id=source.project.owner_user_id,
            conversation_id=source.project.conversation_id,
            cluster=source.project.cluster,
            lifecycle="development",
            created_at=datetime.now(UTC),
        ),
        graph=ProjectGraph(
            graph_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"echomind-runtime-validation-graph:{validation_id}",
                )
            ),
            project_id=validation_id,
            template="echomind.runtime-validation/v1",
            lifecycle="development",
            nodes=nodes,
            edges=deepcopy(source.graph.edges),
        ),
    )

    if args.dry_run:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            if args.publish:
                store.save(source)
            _prepare_recipe(
                store=store,
                source=source,
                recipe=project,
                publish=args.publish,
            )
        return
    _prepare_recipe(
        store=ProjectStore(projects_root),
        source=source,
        recipe=project,
        publish=args.publish,
    )


def _prepare_recipe(
    *, store: ProjectStore, source: ProjectInfo, recipe: ProjectInfo, publish: bool
) -> None:
    draft = build_runtime_contract_draft(source, validation_project=recipe)
    if draft.issues:
        raise ValueError(
            "Runtime validation contract is invalid: "
            + ", ".join(f"{issue.code}: {issue.message}" for issue in draft.issues)
        )
    try:
        existing = store.load(
            recipe.project.project_id, owner_user_id=recipe.project.owner_user_id
        )
    except ProjectNotFoundError:
        store.save(recipe)
    else:
        if existing.graph != recipe.graph:
            raise ValueError(
                "Validation recipe has changed; use a new --validation-key."
            )
        if existing.project.name != recipe.project.name:
            store.save(
                ProjectInfo(
                    project=existing.project.model_copy(
                        update={"name": recipe.project.name}
                    ),
                    graph=existing.graph,
                )
            )

    print(f"source_project_id={source.project.project_id}")
    print(f"validation_recipe_id={recipe.project.project_id}")
    print(f"draft_sha256={draft.content_sha256}")
    if not publish:
        return

    publication = ProjectPublicationService(store.root).publish(
        project=source,
        published_by=source.project.owner_user_id or "runtime-validation",
        publication_mode="production",
        expected_contract_sha256=draft.content_sha256,
        validation_project_id=recipe.project.project_id,
    )
    print(f"published_project_id={publication.project_id}")
    print(f"release_id={publication.release_id}")
    print(f"release_digest={publication.content_sha256}")


def _validation_node(
    *,
    node: ProjectNode,
    dataset_config: dict[str, Any],
    dataset_input: str,
    source_workflow: dict[str, Any],
    train_task_id: str,
    evaluation_workflow: dict[str, Any],
    conversation_id: str,
) -> ProjectNode:
    config = deepcopy(node.config)
    if node.type == "dataset":
        config.update(
            {
                "dataset_runtime": dataset_config,
                "runtime_input_defaults": {"data": dataset_input},
            }
        )
        return node.model_copy(
            update={
                "executor": "dataset_runtime",
                "launch_mode": "executor",
                "config": config,
            }
        )
    if node.type == "train":
        config.update(
            {
                "train_runtime": {
                    "workflow": source_workflow,
                    "wandb_credential_task_id": train_task_id,
                    "workflow_bindings": {
                        "nodes[2].data.config.train_data_path": "train_dataset",
                        "nodes[2].data.config.val_data_path": "validation_dataset",
                        "nodes[3].data.config.model_path": "base_model",
                    },
                    "output_uri": f"{RUN_OUTPUT_ROOT}/candidate",
                    "artifact_selection": "latest_lora_checkpoint",
                    "output_node_id": "12",
                    "output_parameter": "output_path",
                }
            }
        )
        return node.model_copy(
            update={
                "executor": "train_runtime",
                "launch_mode": "executor",
                "config": config,
            }
        )
    if node.type != "evaluator":
        raise ValueError(f"Unsupported validation node type: {node.type}")

    script_path = (
        "/workspace/.pyromind-agent/"
        f"{conversation_id}/versions/v50/"
        "bea08a88-64c4-4c57-adae-10096e71b551/content/evaluate_inference.py"
    )
    dataset_config_path = (
        "/workspace/.pyromind-agent/"
        f"{conversation_id}/versions/v48/"
        "809fe94a-4a4b-442f-b096-8511727d480e/content/dataset_config.json"
    )
    evaluation_config_path = (
        "/workspace/.pyromind-agent/"
        f"{conversation_id}/versions/v49/"
        "86bc8eea-2813-43d0-90c9-b8adaa4f0d0d/content/evaluation_config.json"
    )
    output_dir = f"{RUN_OUTPUT_ROOT}/evaluation"
    merged_model_path = f"{RUN_OUTPUT_ROOT}/merged-model"
    command = (
        f"python3 {script_path} --run-id {{graph_run_id}} "
        '--endpoint "$param" --model default '
        f"--model-reference {{candidate_model}} "
        f"--dataset-path {{validation_dataset}} "
        f"--dataset-config {dataset_config_path} "
        f"--evaluation-config {evaluation_config_path} "
        f"--output-dir {output_dir} --report-assets-dir {output_dir}"
    )
    config.update(
        {
            "evaluator_runtime": {
                "workflow": evaluation_workflow,
                "workflow_bindings": {
                    "nodes[0].data.config.lora_path": "candidate_model",
                    "nodes[0].data.config.model_path": "base_model",
                },
                "workflow_templates": {
                    "nodes[0].data.config.output_path": merged_model_path,
                },
                "command_template": command,
                "command_binding_path": "nodes[2].data.config.command",
                "output_uri": f"{output_dir}/report.json",
                "artifact_selection": "evaluation_report",
            }
        }
    )
    return node.model_copy(
        update={
            "executor": "evaluator_runtime",
            "launch_mode": "executor",
            "config": config,
            "inputs": {
                **node.inputs,
                "base_model": ProjectNodeInput(
                    artifact_type="model.checkpoint",
                    schema_ref="echomind://schemas/model/v1",
                    binding=ProjectNodeInputBinding(
                        kind="run_input", name="base_model"
                    ),
                ),
            },
        }
    )


def _dataset_runtime_config(
    workflow: dict[str, Any], pipeline_source: str
) -> dict[str, Any]:
    workflow = deepcopy(workflow)
    (node,) = workflow["nodes"]
    config = node["data"]["config"]
    command = config["command"]
    historical_output = (
        f"/target-workspace/.pyromind-agent/{SOURCE_CONVERSATION_ID}"
        f"/data_preparation/{DATASET_EXECUTION_ID}"
    )
    historical_input = RAW_DATASET_PATH.replace("/workspace/", "/target-workspace/", 1)
    if not all(part in command for part in (historical_output, historical_input)):
        raise ValueError(
            "The recorded Dataset command does not match the validated run."
        )
    if any(
        key in command for key in ("DF_API_KEY=", "DF_API_URL=", "DF_API_BASE_URL=")
    ):
        raise ValueError("The recorded Dataset workflow embeds API credentials.")

    encoded = base64.b64encode(zlib.compress(pipeline_source.encode())).decode()
    write_pipeline = shlex.quote(
        "from pathlib import Path; import base64, sys, zlib; "
        "Path(sys.argv[1]).write_bytes(zlib.decompress(base64.b64decode(sys.argv[2])))"
    )
    body = (
        command.replace(historical_input, '"$pod_source_path"')
        .replace(historical_output, "$pod_output_path")
        .replace("{", "{{")
        .replace("}", "}}")
    )
    config["command"] = ""
    command_template = (
        "source_path={source_batch}; output_path={output_uri}; "
        'pod_source_path="/target-workspace/${{source_path#/workspace/}}"; '
        'pod_output_path="/target-workspace/${{output_path#/workspace/}}"; '
        'mkdir -p "$pod_output_path" && '
        f'python3 -c {write_pipeline} "$pod_output_path/pipeline.py" {encoded} '
        "|| exit 1; "
        f'{body} && test -s "$pod_output_path/train_messages.jsonl" '
        '&& test -s "$pod_output_path/val_messages.jsonl"'
    )
    return {
        "workflow": workflow,
        "output_uri": f"{RUN_OUTPUT_ROOT}/dataset",
        "command_template": command_template,
        "command_binding_path": "nodes[0].data.config.command",
    }


def _dataset_pipeline_source(workspace_root: Path, conversation_id: str) -> str:
    transcript = (
        workspace_root
        / "conversations"
        / conversation_id
        / "pi"
        / "subagent-artifacts"
        / DATASET_TRANSCRIPT
    )
    with transcript.open(encoding="utf-8") as file:
        for line in file:
            content = json.loads(line).get("message", {}).get("content", [])
            for item in content:
                source = item.get("arguments", {}).get("content")
                if (
                    isinstance(source, str)
                    and sha256(source.encode()).hexdigest() == DATASET_PIPELINE_SHA256
                ):
                    return _adapt_dataset_pipeline(source)
    raise ValueError("The validated Dataset pipeline snapshot is missing.")


def _adapt_dataset_pipeline(source: str) -> str:
    return (
        source.replace(
            "'train_messages.jsonl')\n    val_file =",
            "'train.jsonl')\n    val_file =",
            1,
        )
        .replace(
            "'val_messages.jsonl')\n\n    # Fall back",
            "'val.jsonl')\n\n    # Fall back",
            1,
        )
        .replace(
            "Fall back to processing raw train.jsonl/val.jsonl if messages not found",
            "Use a messages split when the raw split is absent",
        )
        .replace(
            "        train_file = os.path.join(base_dir, 'train.jsonl')",
            "        train_file = os.path.join(base_dir, 'train_messages.jsonl')",
        )
        .replace(
            "        val_file = os.path.join(base_dir, 'val.jsonl')",
            "        val_file = os.path.join(base_dir, 'val_messages.jsonl')",
        )
        .replace(
            "        issues = validate_record(rec, base_dir)",
            """        if 'messages' not in rec:
            assistant_output = rec.get('assistant_output')
            rec = _convert_to_messages(rec, base_dir)
            if isinstance(assistant_output, dict):
                assistant_output = json.dumps(assistant_output, ensure_ascii=False)
            if isinstance(assistant_output, str) and assistant_output.strip():
                rec['messages'][-1]['content'] = assistant_output
        issues = validate_record(rec, base_dir)""",
        )
        .replace(
            "full_path = os.path.join(base_dir, img_path)",
            "full_path = _pod_image_path(base_dir, img_path)",
        )
        .replace(
            "j = extract_result_json(assistant_content)",
            "j = extract_result_json(_message_text(assistant_content))",
        )
        .replace(
            "            clean_records.append(rec)",
            "            clean_records.append(_prepare_training_record(rec, base_dir))",
        )
        .replace(
            "                'boxes': [],",
            "                'boxes': rec.get('boxes', []),",
        )
        .replace(
            "if __name__ == '__main__':",
            _DATASET_MESSAGE_HELPERS + "\n\nif __name__ == '__main__':",
        )
    )


_DATASET_MESSAGE_HELPERS = """def _pod_image_path(base_dir, value):
    if value.startswith('/workspace/'):
        return '/target-workspace/' + value.removeprefix('/workspace/')
    return os.path.normpath(os.path.join(base_dir, value))


def _message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            part['text'] for part in content
            if isinstance(part, dict) and part.get('type') == 'text'
            and isinstance(part.get('text'), str)
        )
    return ''


def _prepare_training_record(record, base_dir):
    def storage_path(value):
        path = _pod_image_path(base_dir, value)
        return '/workspace/' + path.removeprefix('/target-workspace/') if (
            path.startswith('/target-workspace/')
        ) else path

    record['images'] = [storage_path(value) for value in record['images']]
    for message in record['messages']:
        content = message.get('content')
        if isinstance(content, str):
            message['content'] = [{'type': 'text', 'text': content}]
            continue
        if not isinstance(content, list):
            raise ValueError(f"Invalid content for message role {message.get('role')}")
        for part in content:
            if part.get('type') == 'image' and isinstance(part.get('path'), str):
                part['path'] = storage_path(part['path'])
    return record"""


def _replace_wandb_credentials_with_references(workflow: dict[str, Any]) -> None:
    builders = [
        node["data"]
        for node in workflow["nodes"]
        if node["data"].get("nodeType") in WANDB_BUILDER_NODE_TYPES
    ]
    if not builders:
        raise ValueError("The recorded Train workflow has no W&B config builder.")
    for builder in builders:
        config = builder["config"]
        key = config.get("wandb_api_key")
        if not isinstance(key, str) or not key.strip() or key == "WANDB_API_KEY":
            raise ValueError(
                "The recorded Train workflow has no reusable W&B credential."
            )
        builder["nodeType"] = WANDB_BUILDER_NODE_TYPE
        config["wandb_api_key"] = "WANDB_API_KEY"


def _recorded_workflow(
    *,
    workspace_root: Path,
    conversation_id: str,
    task_id: str,
) -> dict[str, Any]:
    path = (
        workspace_root
        / "conversations"
        / conversation_id
        / "product"
        / "workflow-execution-requests.json"
    )
    requests = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        request
        for request in requests.values()
        if isinstance(request, dict) and str(request.get("task_id")) == task_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one workflow request for platform task {task_id}")
    workflow = matches[0].get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError(f"Platform task {task_id} has no embedded workflow")
    return deepcopy(workflow)


def _resolve_workspace_path(workspace_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else workspace_root / value


if __name__ == "__main__":
    main()
