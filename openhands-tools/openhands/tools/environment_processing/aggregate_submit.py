"""Submit the result-aggregation task to Pyromind Studio.

The aggregation step ("merge the shards into training data") runs as its own
CustomCommandCPUNode so the agent never downloads verdicts or traces
locally: the node reads every shard run directory from its Storage mount,
merges verdicts, and converts usable records to the slime RL / SFT formats
with the same incremental append + progress.json contract as the other
platform steps. Pure standard-library execution (no pip install, no
credentials).
"""

from __future__ import annotations

import shlex
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import Field
from rich.text import Text

from openhands.sdk.conversation.state import ActiveLongTask
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    _resolve_conversation_headers,
    _resolve_secret_headers,
    upload_local_file_to_pyromind,
)
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
    submit_workflow_task,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


DATAFLOW_NODE_TYPE = "CustomCommandCPUNode"
AGGREGATE_TASK_KIND = "environment_processing_aggregate"
AGGREGATE_FILENAME = "aggregate_results.py"
SLIME_CONVERTER_FILENAME = "convert_to_slime.py"
SFT_CONVERTER_FILENAME = "convert_to_sft.py"

TOOL_DESCRIPTION = """\
Aggregate validated shard runs into training data on the platform.

Reads the per-shard run output_dirs returned by edp_submit (each holding
run/verdicts.jsonl + run/traces/), merges everything into one dataset and
converts in the same pass, writing straight to --out-dir on Storage:
- verdicts.jsonl : merged per-record verdicts (usable/error + reward)
- slime.jsonl    : usable records in slime RL format (reward-agnostic)
- sft.jsonl      : traces with reward >= min_reward in SFT format
- progress.json  : live snapshot (readable with df_check_progress)
- report.json    : final verdict distribution + conversion stats

The merged verdicts file doubles as the checkpoint: rerun with the same
out_dir (plus any new run_dirs) to resume — already-aggregated task_ids are
skipped, never duplicated. The outputs are the training files themselves:
after the terminal callback, hand <out_dir> over to training — no local
conversion step remains.
"""


class EdpAggregateAction(Action):
    """Submit the merge + training-format conversion task to Studio."""

    run_dirs: list[str] = Field(
        min_length=1,
        description=(
            "Per-shard output_dir values returned by edp_submit (Storage "
            "paths). Each must contain run/verdicts.jsonl next to its shard "
            "manifest."
        ),
    )
    out_dir: str = Field(
        description=(
            "Storage output directory for the merged dataset + training "
            "files (verdicts.jsonl / slime.jsonl / sft.jsonl / report.json). "
            "Reuse it to resume: already-aggregated task_ids are skipped."
        ),
    )
    protocol: str = Field(
        default="tmax",
        description="Protocol tag embedded in slime record metadata.",
    )
    min_reward: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum reward for the SFT set (default 1.0: solved "
            "trajectories only; slime keeps every usable record regardless)."
        ),
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Optional system message prepended to every SFT sample "
            "(coding-agent persona)."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on new records this pass (smoke runs).",
    )
    cpu: int = Field(default=4, ge=1, le=64)
    memory: int = Field(default=16, ge=1, le=256)

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Submit aggregate task: ", style="bold blue")
        content.append(f"{len(self.run_dirs)} shard run(s) -> {self.out_dir}")
        return content


class EdpAggregateObservation(Observation):
    """Result of the aggregation task submission."""

    status: str = Field(default="Unknown")
    task_id: str | None = Field(default=None)
    run_id: str | None = Field(default=None)
    out_dir: str | None = Field(default=None)

    @property
    def visualize(self) -> Text:
        text = Text()
        style = "green" if self.status != "Failed" else "red"
        text.append(f"Aggregate submit: {self.status}\n", style=style)
        if self.out_dir:
            text.append(f"out_dir={self.out_dir}\n")
        text.append(self.text)
        return text


def _pod_path(storage_path: str) -> str:
    return f"/target-workspace{storage_path}"


def build_aggregate_command(
    *,
    out_dir: str,
    run_dirs: Sequence[str],
    protocol: str,
    min_reward: float,
    system_prompt: str | None,
    limit: int | None,
) -> str:
    """Assemble the aggregation command: node python3, stdlib only."""
    pod_out = _pod_path(str(PurePosixPath("/" + out_dir.lstrip("/"))))
    pod_run_dirs = [
        _pod_path(str(PurePosixPath("/" + str(d).lstrip("/")))) for d in run_dirs
    ]
    script = f"{pod_out}/{AGGREGATE_FILENAME}"
    args = [
        "--run-dirs",
        *[shlex.quote(d) for d in pod_run_dirs],
        "--out-dir",
        shlex.quote(pod_out),
        "--protocol",
        shlex.quote(protocol),
        "--min-reward",
        str(float(min_reward)),
    ]
    if system_prompt is not None:
        args += ["--system-prompt", shlex.quote(system_prompt)]
    if limit is not None:
        args += ["--limit", str(int(limit))]
    steps = [
        # Leading no-op: the pod executor can drop the first command segment.
        "true",
        f"test -f {script}",
        f"test -f {pod_out}/{SLIME_CONVERTER_FILENAME}",
        f"test -f {pod_out}/{SFT_CONVERTER_FILENAME}",
        # Pure standard library: run with the node python directly. The
        # script's own directory is sys.path[0], so its sibling converter
        # imports resolve from the same staged directory.
        f"python3 {script} " + " ".join(args),
    ]
    return " && ".join(steps)


class EdpAggregateExecutor(ToolExecutor[EdpAggregateAction, EdpAggregateObservation]):
    """Stage the aggregation scripts and submit the task."""

    def __init__(
        self,
        *,
        env: str | None,
        cluster: str | None,
        headers: dict[str, str] | None,
        runtime_dir: str | None,
        storage_base_url: str | None,
        storage_headers: dict[str, str] | None,
        storage_secret_headers: dict[str, str] | None,
        timeout: int = 30,
    ) -> None:
        self._env = env
        self._cluster = cluster
        self._headers = headers or {}
        self._runtime_dir = runtime_dir
        self._storage_base_url = storage_base_url
        self._storage_headers = storage_headers or {}
        self._storage_secret_headers = storage_secret_headers or {}
        self._timeout = timeout

    def _storage_headers_for(self, conversation: BaseConversation) -> dict[str, str]:
        headers = {**self._storage_headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(
            _resolve_secret_headers(conversation, self._storage_secret_headers)
        )
        return headers

    def _stage(
        self,
        conversation: BaseConversation,
        local_path: Path,
        output_dir: str,
    ) -> None:
        upload_local_file_to_pyromind(
            local_path=local_path,
            target_dir=output_dir,
            storage_base_url=self._storage_base_url or _default_storage_base_url(),
            headers=self._storage_headers_for(conversation),
            timeout=float(self._timeout),
        )

    def _stage_scripts(self, conversation: BaseConversation, output_dir: str) -> None:
        if self._runtime_dir is None:
            raise ValueError("aggregate runtime_dir not configured.")
        runtime = Path(self._runtime_dir)
        scripts = [
            runtime / AGGREGATE_FILENAME,
            runtime / SLIME_CONVERTER_FILENAME,
            runtime / SFT_CONVERTER_FILENAME,
        ]
        missing = [str(p) for p in scripts if not p.is_file()]
        if missing:
            raise ValueError(
                f"aggregation scripts not found in runtime_dir: {', '.join(missing)}"
            )
        for script in scripts:
            self._stage(conversation, script, output_dir)

    def __call__(
        self,
        action: EdpAggregateAction,
        conversation: BaseConversation | None = None,
    ) -> EdpAggregateObservation:
        if conversation is None:
            return EdpAggregateObservation.from_text(
                text="edp_aggregate requires an active conversation.",
                status="Failed",
                is_error=True,
            )
        out_dir = str(PurePosixPath("/" + action.out_dir.lstrip("/")))

        try:
            state = cast("ConversationState", conversation.state)
            auth_token = state.secret_registry.get_secret_value(
                PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
            )
            if not auth_token:
                raise ValueError(
                    "Platform auth_token missing from conversation secrets."
                )
            platform_env = self._env or ""
            cluster = self._cluster or ""
            if not platform_env or not cluster:
                raise ValueError(
                    "env/cluster not wired into the edp_aggregate tool; the "
                    "agent-server Pyromind router should inject them."
                )

            self._stage_scripts(conversation, out_dir)

            run_id = uuid.uuid4()
            command = build_aggregate_command(
                out_dir=out_dir,
                run_dirs=action.run_dirs,
                protocol=action.protocol,
                min_reward=action.min_reward,
                system_prompt=action.system_prompt,
                limit=action.limit,
            )
            workflow = {
                "id": str(run_id),
                "name": f"agent-edp-aggregate-{str(run_id)[:8]}",
                "nodes": [
                    {
                        "id": "1",
                        "type": "default",
                        "position": {"x": 0, "y": 0},
                        "data": {
                            "display_name": "Aggregate Results",
                            "nodeType": DATAFLOW_NODE_TYPE,
                            "config": {
                                "command": command,
                                "cpu": action.cpu,
                                "memory": action.memory,
                            },
                        },
                    }
                ],
                "edges": [],
                "viewport": {"x": 0, "y": 0, "zoom": 1},
                "timestamp": datetime.now(UTC).isoformat(),
            }

            client = create_workflow_api_client(
                env=platform_env,
                cluster=cluster,
                auth_token=auth_token,
                headers=self._headers,
                timeout=self._timeout,
            )
            response = submit_workflow_task(
                client=client,
                workflow=workflow,
                name=str(workflow["name"]),
                conversation_id=str(conversation.id),
            )
            task_id = response.task_id
        except Exception as exc:  # noqa: BLE001
            return EdpAggregateObservation.from_text(
                text=f"Failed to submit aggregate task: {exc}",
                status="Failed",
                is_error=True,
            )

        conversation.register_active_long_task(
            ActiveLongTask(
                task_id=task_id,
                kind=AGGREGATE_TASK_KIND,
                status=response.status,
            )
        )
        return EdpAggregateObservation.from_text(
            text=(
                "Aggregate task submitted. "
                f"task_id={task_id}, out_dir={out_dir}. "
                "After the terminal callback, read "
                f"{out_dir}/report.json for the verdict distribution and "
                f"conversion stats; {out_dir}/slime.jsonl and "
                f"{out_dir}/sft.jsonl are the training files. Rerun with the "
                "same out_dir (and any new run_dirs) to resume."
            ),
            status=response.status,
            task_id=task_id,
            run_id=str(run_id),
            out_dir=out_dir,
        )


class EdpAggregateTool(ToolDefinition[EdpAggregateAction, EdpAggregateObservation]):
    """Submit the shard-result aggregation task to Pyromind Studio."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        env_value = params.pop("env", None)
        env = str(env_value) if env_value is not None else None
        cluster_value = params.pop("cluster", None)
        cluster = str(cluster_value) if cluster_value is not None else None
        params.pop("current_user", None)
        headers = params.pop("headers", None) or {}
        runtime_dir_value = params.pop("runtime_dir", None)
        runtime_dir = str(runtime_dir_value) if runtime_dir_value is not None else None
        storage_base_url_value = params.pop("storage_base_url", None)
        storage_base_url = (
            str(storage_base_url_value) if storage_base_url_value is not None else None
        )
        storage_headers = params.pop("storage_headers", None) or None
        storage_secret_headers = params.pop("storage_secret_headers", None) or None
        timeout = int(params.pop("timeout", 30))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"EdpAggregateTool got unknown params: {names}")
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=EdpAggregateAction,
                observation_type=EdpAggregateObservation,
                executor=EdpAggregateExecutor(
                    env=env,
                    cluster=cluster,
                    headers=headers,
                    runtime_dir=runtime_dir,
                    storage_base_url=storage_base_url,
                    storage_headers=storage_headers,
                    storage_secret_headers=storage_secret_headers,
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="edp_aggregate",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(EdpAggregateTool.name, EdpAggregateTool)
