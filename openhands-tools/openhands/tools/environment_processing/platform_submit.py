"""Submit environment-validation batches to Pyromind Studio for async execution.

Mirrors the data-preparation platform submission pattern (``df_submit_pipeline``):
stage the frozen runner + profile + pod runtime into a per-run Storage
directory, build one ``CustomCommandCPUNode`` workflow per shard manifest
whose command installs openhands-tools, injects credentials, and runs the
skill's ``sandbox_runner.py`` against the shard's Storage manifest, then
submit and let the Kafka terminal callback resume the conversation.

The runner writes its verdicts and traces into the pod's mounted Storage
directory, so after the callback everything is inspectable with
``preview_dataset`` — no local sandbox loop is involved.

Credentials come from the conversation secret registry exactly like every
other platform tool: the platform ``auth_token`` (key ``auth_token``) and the
coding-agent LLM triple (``LLM_BASE_URL`` / ``LLM_AUTH_TOKEN`` /
``LLM_MODEL``, falling back to ``DF_API_BASE_URL`` / ``DF_API_KEY`` /
``DF_MODEL_NAME`` and the legacy ``ANTHROPIC_*`` secrets). Values reach the
pod only via the node command environment and never land in verdicts
(runner contract).
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import Field, model_validator
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
    download_file_from_pyromind,
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
EDP_TASK_KIND = "environment_processing"

DEFAULT_PROFILE_NAME = "tmax-validation"
RUNNER_FILENAME = "sandbox_runner.py"
MANIFEST_FILENAME = "manifest.jsonl"
OUTPUT_SUBDIR = "run"  # runner --output-dir inside the pod mount
SHARDS_INDEX_MAX_BYTES = 4 * 1024 * 1024  # shards.json is a small path index

# The CustomCommandCPUNode pod runs Python 3.10 (conda), while the openhands
# distributions require Python >= 3.12. The pod runtime stages a minimal
# openhands namespace (processing_profile.py verbatim + a sandbox client
# shim) into the run's Storage mount, and the node installs only
# pyromind-sdk from PyPI.
POD_RUNTIME_DIR = "pod_runtime"
POD_REQUIREMENTS_FILENAME = "pod_requirements.txt"

LLM_ENV_KEYS = (
    "LLM_BASE_URL",
    "LLM_AUTH_TOKEN",
    "LLM_MODEL",
)
# Fallback sources mirroring the skill's credential contract: platform DF_*
# vars first, then the legacy ANTHROPIC_* names from the Claude Code chain.
LLM_FALLBACK_KEYS = {
    "LLM_BASE_URL": ("DF_API_BASE_URL", "ANTHROPIC_BASE_URL"),
    "LLM_AUTH_TOKEN": ("DF_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "LLM_MODEL": ("DF_MODEL_NAME", "ANTHROPIC_MODEL"),
}

TOOL_DESCRIPTION = """\
Submit environment-validation batches to Pyromind platform execution.

Each shard manifest is submitted as its own one-node CustomCommandCPUNode
workflow: the tool freezes the skill's sandbox_runner.py, the chosen
ProcessingProfile, and a minimal pod runtime (openhands namespace shim +
pod_requirements.txt) into a per-run Storage directory, then the node
installs pyromind-sdk and httpx (Python 3.10 pod), injects the platform
auth_token and the coding-agent LLM triple (conversation secrets, DF_*
fallback), and runs `sandbox_runner.py --profile ... --manifest ...` over
every record serially (create sandbox -> install coding agent -> solve ->
verifier -> cleanup), with resume/limit/dedup flags forwarded. Per-record
progress is logged one line per record in the node log.

Two modes (exactly one required):
- `manifest`: one Storage shard manifest (e.g. '<render_root>/batch-001/
  manifest.jsonl'). The smoke path: submit a single shard first and check
  the verdict distribution before committing to the full run.
- `shards` + optional `shard_offset`/`shard_count`: the render step's
  shards.json index. Submit N shards at once, each as an independent
  workflow with its own output_dir and terminal callback. The batch size
  (shard_count) and any full-run submission MUST be confirmed by the user
  via an explicit question — never decide batch capacity yourself.

Execution is asynchronous. Terminal Kafka callbacks resume the
conversation per shard; the agent must check whether every shard of the
batch reached a terminal state before asking about the next batch. After
the callbacks, inspect artefacts exclusively with `preview_dataset` under
the returned output_dirs:
- <output_dir>/run/verdicts.jsonl : per-record usable/error + reward
- <output_dir>/run/traces/<task_id>.pi_trace.jsonl : agent json traces

Pass the same run_id again to resume: each shard's run directory is keyed
by run_id, and its stored verdicts file doubles as the checkpoint.
"""


class EdpSubmitAction(Action):
    """Submit environment-validation shard manifests to Pyromind Studio."""

    manifest: str | None = Field(
        default=None,
        description=(
            "Single Storage shard manifest (e.g. 'edp/batch-001/"
            "manifest.jsonl'), staged by edp_render. The smoke path: one "
            "shard, check verdicts before committing to the full run. "
            "Mutually exclusive with shards."
        ),
    )
    shards: str | None = Field(
        default=None,
        description=(
            "Storage path of the shards.json index written by edp_render. "
            "Submit multiple shard manifests at once, each as an independent "
            "workflow with its own output_dir. Mutually exclusive with "
            "manifest. shard_count must be confirmed by the user."
        ),
    )
    shard_offset: int = Field(
        default=0,
        ge=0,
        description="First shard index to submit from the shards.json list.",
    )
    shard_count: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of shards to submit this call (default: all remaining "
            "from shard_offset). The batch capacity — confirm it with the "
            "user, never decide it yourself."
        ),
    )
    profile_name: str = Field(
        default=DEFAULT_PROFILE_NAME,
        description=(
            "ProcessingProfile basename (without .json) from the skill's "
            "profiles/ directory, e.g. 'tmax-validation'."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "Reuse an existing run id to resume: the runner skips records "
            "already judged in the stored verdicts checkpoint. Omit for a "
            "new full run."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional record cap for smoke runs (progressive sampling); "
            "omit to process the whole manifest."
        ),
    )
    dedup_by_image: bool = Field(
        default=False,
        description=(
            "Reuse verdicts across records sharing an image. Only safe for "
            "environment-only probes, never with agent-solving profiles."
        ),
    )
    cpu: int = Field(
        default=4,
        ge=1,
        le=64,
        description="CPU cores for the pipeline pod (1-64, default 4).",
    )
    memory: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Memory for the pipeline pod in GiB (1-256, default 16).",
    )

    @model_validator(mode="after")
    def _manifest_source_exclusive(self) -> EdpSubmitAction:
        if (self.manifest is None) == (self.shards is None):
            raise ValueError(
                "Set exactly one of manifest (single Storage shard) or "
                "shards (shards.json index)."
            )
        return self

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Submit env-validation batch: ", style="bold blue")
        content.append(self.manifest or self.shards or "")
        return content


class EdpSubmitObservation(Observation):
    """Result of environment-validation platform submissions (per shard)."""

    status: str = Field(default="Unknown")
    task_ids: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    output_dirs: list[str] = Field(default_factory=list)
    resumed: bool = Field(default=False)

    @property
    def visualize(self) -> Text:
        text = Text()
        style = "green" if self.status != "Failed" else "red"
        text.append(f"Env-validation submit: {self.status}\n", style=style)
        for run_id, output_dir in zip(self.run_ids, self.output_dirs, strict=False):
            text.append(f"run_id={run_id} output_dir={output_dir}\n")
        text.append(self.text)
        return text


# ---------------------------------------------------------------------------
# Credential resolution (session secret registry, DF_* fallback)
# ---------------------------------------------------------------------------


def _secret_lookup(conversation: BaseConversation, name: str) -> str | None:
    state = cast("ConversationState", conversation.state)
    value = state.secret_registry.get_secret_value(name)
    if value:
        return value
    return os.environ.get(name) or None


def resolve_llm_env(conversation: BaseConversation) -> dict[str, str]:
    """Resolve the coding-agent LLM triple; raises when incomplete.

    Order: LLM_* in the session secret registry (then process env), falling
    back to DF_API_BASE_URL / DF_API_KEY / DF_MODEL_NAME and legacy
    ANTHROPIC_* secrets. The base URL never keeps a trailing ``/v1`` here;
    install_pi's provider registration re-appends ``/v1`` for OpenAI
    chat-completions endpoints.
    """
    resolved: dict[str, str] = {}
    for key in LLM_ENV_KEYS:
        value = _secret_lookup(conversation, key)
        if not value:
            for fallback in LLM_FALLBACK_KEYS[key]:
                value = _secret_lookup(conversation, fallback)
                if value:
                    break
        if key == "LLM_BASE_URL" and value:
            value = value.rstrip("/")
            if value.endswith("/v1"):
                value = value[: -len("/v1")]
        if not value:
            raise ValueError(
                f"Missing LLM credential {key!r} in conversation secrets "
                f"(fallbacks: {', '.join(LLM_FALLBACK_KEYS[key])}). Configure "
                "it before submitting an env-validation batch."
            )
        resolved[key] = value
    return resolved


# ---------------------------------------------------------------------------
# Command / workflow builders
# ---------------------------------------------------------------------------


def _pod_path(storage_path: str) -> str:
    return f"/target-workspace{storage_path}"


def build_edp_command(
    *,
    output_dir: str,
    profile_name: str,
    platform_env: str,
    cluster: str,
    auth_token: str,
    llm_env: Mapping[str, str],
    limit: int | None = None,
    dedup_by_image: bool = False,
    manifest_pod_path: str | None = None,
) -> str:
    """Assemble the shell command executed inside the CustomCommandCPUNode pod.

    ``manifest_pod_path`` overrides the staged manifest location: pass the
    Storage-mount path (``/target-workspace<storage path>``) when the manifest
    already lives on Storage and is not staged into the run directory.
    """
    pod_output = _pod_path(output_dir)
    pod_runner = f"{pod_output}/{RUNNER_FILENAME}"
    pod_profile = f"{pod_output}/{profile_name}.json"
    pod_manifest = manifest_pod_path or f"{pod_output}/{MANIFEST_FILENAME}"
    pod_run_dir = f"{pod_output}/{OUTPUT_SUBDIR}"
    pod_requirements = f"{pod_output}/{POD_REQUIREMENTS_FILENAME}"
    venv = "/tmp/edp-venv"

    env_items = [
        ("PYROMIND_ENV", platform_env),
        ("PYROMIND_CLUSTER", cluster),
        *[(key, llm_env[key]) for key in LLM_ENV_KEYS if key in llm_env],
        # Auth token deliberately last: the pod executor has been observed to
        # drop the very first `export K=V;` segment of a long command, which
        # previously stripped the platform credential before the runner saw
        # it. It is additionally passed via --auth-token below.
        ("PYROMIND_AUTH_TOKEN", auth_token),
    ]
    # Must be real `export ...;` statements, not one-shot prefix assignments:
    # the `--set KEY="$KEY"` args below expand at argument parsing time, and a
    # prefix assignment (`K=v python ...`) is invisible to same-shell $K
    # expansion, which would strip every credential before the runner sees it.
    env_prefix = (
        "; ".join(f"export {key}={shlex.quote(str(val))}" for key, val in env_items)
        + ";"
    )
    # The pod executor has been observed to drop the very first `export ...;`
    # segment of the pipeline command (observed with auth token, then with the
    # env var exported first). Prefixing a no-op statement absorbs that quirk
    # so every real export reaches the runner. First export (=env var list
    # head, previously the auth token) is also why the credential never made
    # it into the process environment in early smoke runs.
    env_prefix = f"true; {env_prefix}"

    setup_steps = [
        f"python3 -m venv {venv}",
        (
            f"{venv}/bin/pip install --quiet --use-deprecated=legacy-resolver "
            f"-r {shlex.quote(pod_requirements)}"
            " > /tmp/edp-install.log 2>&1"
            " || { echo 'runner dependencies install failed; "
            "last install log lines:'; tail -n 100 /tmp/edp-install.log; "
            "exit 1; }"
        ),
        f"mkdir -p {shlex.quote(pod_run_dir)}",
        f"test -f {shlex.quote(pod_runner)}",
        f"test -f {shlex.quote(pod_profile)}",
        f"test -f {shlex.quote(pod_manifest)}",
    ]

    runner_args = [
        f"--profile {shlex.quote(pod_profile)}",
        f"--manifest {shlex.quote(pod_manifest)}",
        f"--output-dir {shlex.quote(pod_run_dir)}",
        f"--env {shlex.quote(platform_env)}",
        f"--cluster {shlex.quote(cluster)}",
        '--auth-token "$PYROMIND_AUTH_TOKEN"',
        *(
            # Expanded from the pod-env exports above; never inlined as
            # literals so the values stay out of the workflow config on disk.
            # Conditional on the credential being present: an absent key must
            # surface as "secret not provided", never as an empty value.
            f'--set {key}="${key}"'
            for key in LLM_ENV_KEYS
            if key in llm_env
        ),
    ]
    if limit is not None:
        runner_args.append(f"--limit {int(limit)}")
    if dedup_by_image:
        runner_args.append("--dedup-by-image")

    # PYTHONPATH makes the staged pod_runtime openhands namespace visible;
    # the frozen runner then imports processing_profile + the sandbox shim
    # from the Storage mount instead of the (unavailable) openhands wheels.
    pipeline_step = (
        f"{env_prefix} "
        f"PYTHONPATH={shlex.quote(pod_output)} "
        f"{venv}/bin/python {shlex.quote(pod_runner)} " + " ".join(runner_args)
    )
    return " && ".join([*setup_steps, pipeline_step])


def build_edp_workflow(
    *,
    run_id: uuid.UUID,
    command: str,
    cpu: int,
    memory: int,
) -> dict[str, Any]:
    return {
        "id": str(run_id),
        "name": f"agent-edp-{str(run_id)[:8]}",
        "nodes": [
            {
                "id": "1",
                "type": "default",
                "position": {"x": 0, "y": 0},
                "data": {
                    "display_name": "Env Validation Batch",
                    "nodeType": DATAFLOW_NODE_TYPE,
                    "config": {"command": command, "cpu": cpu, "memory": memory},
                },
            }
        ],
        "edges": [],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class EdpSubmitExecutor(ToolExecutor[EdpSubmitAction, EdpSubmitObservation]):
    """Stage runner/profile/manifest and submit one batch task."""

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

    # -- staging -----------------------------------------------------------

    def _storage_headers_for(self, conversation: BaseConversation) -> dict[str, str]:
        headers = {**self._storage_headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(
            _resolve_secret_headers(conversation, self._storage_secret_headers)
        )
        return headers

    def _stage_file(
        self,
        conversation: BaseConversation,
        local_path: Path,
        output_dir: str,
        filename: str,
    ) -> None:
        headers = self._storage_headers_for(conversation)
        local = local_path
        if local.name != filename:
            # The upload API keys the storage file by its local basename, so
            # rename via a staging dir when the target name differs (e.g. an
            # agent-authored manifest not already named manifest.jsonl).
            with tempfile.TemporaryDirectory(prefix="edp-stage-") as tmp:
                target = Path(tmp) / filename
                target.write_bytes(local.read_bytes())
                local = target
        upload_local_file_to_pyromind(
            local_path=local,
            target_dir=output_dir,
            storage_base_url=self._storage_base_url or _default_storage_base_url(),
            headers=headers,
            timeout=float(self._timeout),
        )

    def _stage_bundle(
        self,
        conversation: BaseConversation,
        profile_name: str,
        output_dir: str,
    ) -> None:
        if self._runtime_dir is None:
            raise ValueError(
                "Environment-processing runtime_dir is not configured (agent-server "
                "should point it at the skill's scripts directory)."
            )
        runtime_dir = Path(self._runtime_dir)
        runner_path = runtime_dir / RUNNER_FILENAME
        pod_runtime_dir = runtime_dir / POD_RUNTIME_DIR
        profile_path = runtime_dir.parent / "profiles" / f"{profile_name}.json"
        if not runner_path.is_file():
            raise ValueError(f"Runner not found in runtime_dir: {runner_path}")
        if not profile_path.is_file():
            raise ValueError(
                f"Profile {profile_name!r} not found: {profile_path} "
                f"(choices under {runtime_dir.parent / 'profiles'})"
            )
        if not pod_runtime_dir.is_dir():
            raise ValueError(
                f"Pod runtime not found: {pod_runtime_dir} (missing "
                f"{POD_RUNTIME_DIR} under the skill scripts directory)"
            )
        self._stage_file(conversation, runner_path, output_dir, RUNNER_FILENAME)
        self._stage_file(conversation, profile_path, output_dir, f"{profile_name}.json")
        # pod_runtime is uploaded preserving its relative tree so the pod's
        # PYTHONPATH=<output_dir> resolves the openhands namespace shim.
        for path in sorted(pod_runtime_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(pod_runtime_dir)
            target_dir = (
                output_dir if rel.parent == Path(".") else f"{output_dir}/{rel.parent}"
            )
            self._stage_file(conversation, path, target_dir, rel.name)

    def _resolve_manifest_specs(
        self, action: EdpSubmitAction, conversation: BaseConversation
    ) -> list[str]:
        """Return the Storage manifest paths to submit this call."""
        if action.manifest is not None:
            return [str(PurePosixPath("/" + action.manifest.lstrip("/")))]
        assert action.shards is not None
        index_path = str(PurePosixPath("/" + action.shards.lstrip("/")))
        raw = download_file_from_pyromind(
            storage_path=index_path,
            storage_base_url=self._storage_base_url or _default_storage_base_url(),
            headers=self._storage_headers_for(conversation),
            timeout=float(self._timeout),
            max_bytes=SHARDS_INDEX_MAX_BYTES,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"shards index {index_path} is not valid JSON: {exc}"
            ) from exc
        shards = data.get("shards") if isinstance(data, dict) else None
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"shards index {index_path} has no 'shards' list")
        end = len(shards)
        if action.shard_count is not None:
            end = min(end, action.shard_offset + action.shard_count)
        selected = shards[action.shard_offset : end]
        if not selected:
            raise ValueError(
                f"shard_offset={action.shard_offset}"
                + (
                    f" shard_count={action.shard_count}"
                    if action.shard_count is not None
                    else ""
                )
                + f" selects no shards from {len(shards)} in {index_path}"
            )
        return [str(PurePosixPath("/" + str(s).lstrip("/"))) for s in selected]

    # -- submission --------------------------------------------------------

    def __call__(
        self,
        action: EdpSubmitAction,
        conversation: BaseConversation | None = None,
    ) -> EdpSubmitObservation:
        if conversation is None:
            return EdpSubmitObservation.from_text(
                text="edp_submit requires an active conversation.",
                status="Failed",
                is_error=True,
            )

        resumed = action.run_id is not None
        shared_run_id = uuid.UUID(action.run_id) if action.run_id else None

        try:
            specs = self._resolve_manifest_specs(action, conversation)
            llm_env = resolve_llm_env(conversation)
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
                    "env/cluster not wired into the edp_submit tool; the "
                    "agent-server Pyromind router should inject them."
                )
            client = create_workflow_api_client(
                env=platform_env,
                cluster=cluster,
                auth_token=auth_token,
                headers=self._headers,
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return EdpSubmitObservation.from_text(
                text=f"Failed to submit env-validation batch: {exc}",
                status="Failed",
                is_error=True,
            )

        task_ids: list[str] = []
        run_ids: list[str] = []
        output_dirs: list[str] = []
        last_status = "Unknown"
        failure: str | None = None
        for spec in specs:
            # The run directory is keyed by run_id (shared across shards when
            # resuming) under the shard manifest's own directory, keeping the
            # manifest and its checkpoint together; the workflow id is fresh
            # per shard so repeated submissions never collide on the platform.
            run_dir_id = shared_run_id or uuid.uuid4()
            output_dir = str(PurePosixPath(spec).parent / str(run_dir_id))
            try:
                self._stage_bundle(conversation, action.profile_name, output_dir)
                command = build_edp_command(
                    output_dir=output_dir,
                    profile_name=action.profile_name,
                    platform_env=platform_env,
                    cluster=cluster,
                    auth_token=auth_token,
                    llm_env=llm_env,
                    limit=action.limit,
                    dedup_by_image=action.dedup_by_image,
                    manifest_pod_path=_pod_path(spec),
                )
                workflow = build_edp_workflow(
                    run_id=uuid.uuid4(),
                    command=command,
                    cpu=action.cpu,
                    memory=action.memory,
                )
                response = submit_workflow_task(
                    client=client,
                    workflow=workflow,
                    name=str(workflow["name"]),
                    conversation_id=str(conversation.id),
                )
            except Exception as exc:  # noqa: BLE001
                # Stop the batch at the first failing shard: partially
                # submitted shards keep running, the rest are untouched.
                failure = f"Failed to submit shard {spec}: {exc}"
                break
            conversation.register_active_long_task(
                ActiveLongTask(
                    task_id=response.task_id,
                    kind=EDP_TASK_KIND,
                    status=response.status,
                )
            )
            task_ids.append(response.task_id)
            run_ids.append(str(run_dir_id))
            output_dirs.append(output_dir)
            last_status = response.status

        if not task_ids:
            return EdpSubmitObservation.from_text(
                text=failure or "No shards were submitted.",
                status="Failed",
                is_error=True,
            )

        text = (
            f"Env-validation batches submitted ({len(task_ids)} shard workflow(s)). "
            "While they run, resume by resubmitting with the same run_id. "
            "After the terminal callbacks, inspect each "
            "<output_dir>/run/verdicts.jsonl, then <output_dir>/run/traces/ "
            "for agent traces."
        )
        if failure:
            text += f" WARNING: batch stopped early — {failure}"
        return EdpSubmitObservation.from_text(
            text=text,
            status=last_status,
            task_ids=task_ids,
            run_ids=run_ids,
            output_dirs=output_dirs,
            resumed=resumed,
            is_error=failure is not None,
        )


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class EdpSubmitTool(ToolDefinition[EdpSubmitAction, EdpSubmitObservation]):
    """Submit environment-validation shard manifests as Studio tasks."""

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
            raise ValueError(f"EdpSubmitTool got unknown params: {names}")
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=EdpSubmitAction,
                observation_type=EdpSubmitObservation,
                executor=EdpSubmitExecutor(
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
                    title="edp_submit",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(EdpSubmitTool.name, EdpSubmitTool)
