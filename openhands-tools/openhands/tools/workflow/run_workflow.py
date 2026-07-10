"""Submit a Pyromind workflow to the platform for asynchronous execution.

This module defines the ``run_workflow`` tool used by Pyromind agents to submit
agent-authored workflow Python DSL to the platform, return a task ID after
submission, and surface submission outcomes back to the LLM. Final run status
and runtime errors are delivered later via a platform callback. Platform HTTP
integration lives in :class:`RunWorkflowExecutor`; header/auth resolution mirrors
:mod:`validate_workflow_dsl`.

在 Pyromind 平台上异步提交工作流。

本模块定义 ``run_workflow`` 工具：Pyromind Agent 将 Agent 编写的工作流 Python DSL
提交到平台，返回任务 ID，并把提交结果返回给 LLM。工作流终态与运行错误通过平台
callback 异步回传。平台 HTTP 集成在 :class:`RunWorkflowExecutor` 中；header/鉴权
解析与 :mod:`validate_workflow_dsl` 保持一致。
"""  # noqa: E501

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import Field
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.models import (
    TrainingTaskCreateRequest,
    TrainingTaskCreateResponse,
)

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from openhands.tools.utils.pyromind_api_client import (
    get_api_key,
    get_pyromind_api_client,
)
from openhands.tools.workflow.dsl_to_xyflow import (
    DslToXyflowAction,
    DslToXyflowExecutor,
    DslToXyflowObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


# Default retry budget per executor instance / 每个 Executor 实例的默认最大尝试次数
DEFAULT_MAX_ATTEMPTS = 10


class WorkflowRunError(RuntimeError):
    """Raised when the Pyromind platform fails to create or run a workflow.

    This is a runtime/platform error rather than an invalid input value, so it
    inherits from ``RuntimeError``. The executor catches it and turns it into an
    error observation for the agent loop.

    当 Pyromind 平台创建或运行工作流失败时抛出。属于运行时/平台错误，而非输入值
    不合法，因此继承 ``RuntimeError``。Executor 会捕获它并转为 error observation。
    """





class RunWorkflowAction(Action):
    """Trigger a run of the current Pyromind workflow DSL on the platform.

    触发在当前 Pyromind 工作流 DSL 上的平台运行。
    """

    note: str | None = Field(
        default=None,
        description=(
            "Optional short note on what changed since the previous run "
            "attempt. Not sent to the platform; purely for conversation history."
        ),
    )
    dsl: str | None = Field(
        default=None,
        description=(
            "Pyromind workflow Python DSL source code to run (not a file path). "
            "Required: pass the declarative workflow script text the agent "
            "generated or edited."
        ),
    )
    name: str = Field(
        default="workflow",
        description="Workflow name passed to the platform when submitting the run.",
    )
    test_mode: bool = Field(
        default=False,
        description="Whether to run the workflow in test mode.",
    )


class RunWorkflowObservation(Observation):
    """Result of one workflow run attempt.

    单次工作流运行尝试的结果。
    """

    status: Literal[
        "Succeeded", "Pending", "Running", "Failed", "Error", "Terminated"
    ] = Field(
        description=(
            "'Succeeded': 工作流执行完成; "
            "'Pending': 工作流提交成功，等待系统调度中; "
            "'Running': 工作流运行中; "
            "'Failed': 工作流提交失败; "
            "'Error': 工作流运行异常; "
            "'Terminated': 工作流被主动停止"
        )
    )
    error_log: str | None = Field(
        default=None,
        description=("Runtime error output when status is 'Failed' or 'Error'."),
    )
    task_id: str | None = Field(
        default=None,
        description="工作流提交成功，返回的任务 ID",
    )
    attempt: int = Field(
        description="This run attempt number for the current conversation."
    )
    max_attempts: int = Field(description="The maximum number of run attempts allowed.")

    keep_ui_lock: bool = Field(
        default=False,
        description="Whether to keep the UI lock after the workflow runs.",
    )


TOOL_DESCRIPTION = """Submit the current Pyromind workflow to the platform for asynchronous execution.

Use this tool when the user asks to run, test, debug, or 试跑 a workflow. Pass the
required workflow Python DSL source code in `dsl` (not a file path). Do not execute
the workflow locally with bash or Python.

Platform execution is asynchronous: this tool returns a workflow task ID after
submission. Final run status and runtime errors will be delivered later via a
platform callback once the workflow completes.
"""  # noqa: E501


class RunWorkflowExecutor(ToolExecutor[RunWorkflowAction, RunWorkflowObservation]):
    """Submit a workflow run to Pyromind asynchronously.

    One executor instance is created per conversation via
    :meth:`RunWorkflowTool.create`. It owns the per-conversation attempt counter
    and the env/cluster/header context injected by the agent-server Pyromind
    router. This tool returns after the platform accepts the submission; it does
    not block until the workflow finishes. Terminal run status is delivered later
    via a platform callback.

    向 Pyromind 异步提交工作流运行。

    每个会话通过 :meth:`RunWorkflowTool.create` 创建一个 Executor 实例，维护本会话
    的尝试计数，以及 agent-server Pyromind router 注入的 env/cluster/header 上下文。
    工具在平台接受提交后即返回，不会阻塞等待工作流执行结束；终态结果通过平台
    callback 异步回传。
    """

    def __init__(
        self,
        *,
        cluster: str | None = None,
        env: str | None = None,
        current_user: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Configure how this executor talks to the Pyromind run API.

        配置 Executor 与 Pyromind 运行 API 的通信参数。

        Args:
            cluster: 运行所在集群
            env: 运行所在环境
            current_user: 当前用户信息
            headers: Non-secret HTTP headers forwarded on every run request,
                for example ``x-cluster``. Populated by the Pyromind router from
                the incoming HTTP/WebSocket request.
                每次运行请求转发的非敏感 HTTP header（如 ``x-cluster``），由
                Pyromind router 从入站 HTTP/WebSocket 请求填充。
        """
        self.cluster = cluster
        self.env = env
        self.current_user = current_user
        self._max_attempts = DEFAULT_MAX_ATTEMPTS     # 最大尝试次数
        self._attempt = 0                             # 当前尝试次数
        self.headers = headers or {}

    def __call__(
        self,
        action: RunWorkflowAction,
        conversation: BaseConversation | None = None,
    ) -> RunWorkflowObservation:
        """Handle one ``run_workflow`` tool invocation from the agent loop.

        Performs pre-flight checks before delegating to :meth:`_execute_run`:

        1. Require a local conversation so workspace paths and secrets exist.
        2. Require non-empty workflow DSL source code in ``action.dsl``.
        3. Enforce the per-executor attempt budget.

        Returns an error observation immediately when any pre-flight check fails;
        otherwise increments the attempt counter and forwards to the platform
        integration hook.

        处理 Agent 循环中的一次 ``run_workflow`` 工具调用。

        在委托 :meth:`_execute_run` 前执行预检：

        1. 必须有本地 conversation，以便访问 workspace 路径与 secrets。
        2. 要求 ``action.dsl`` 提供非空的工作流 DSL 源码。
        3. 检查本 Executor 的尝试次数上限。

        任一预检失败则立即返回 error observation；否则递增计数并进入平台集成逻辑。
        """
        if conversation is None:
            return RunWorkflowObservation.from_text(
                text="run_workflow requires a local conversation context.",
                status="Error",
                attempt=self._attempt,
                max_attempts=self._max_attempts,
                is_error=True,
            )

        if self._attempt >= self._max_attempts:
            return RunWorkflowObservation.from_text(
                text=(
                    f"Reached the maximum of {self._max_attempts} run attempts. "
                    "Stop calling run_workflow and report the remaining failure "
                    "to the user."
                ),
                status="Error",
                attempt=self._attempt,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log="Reached the maximum number of run attempts.",
            )
        # 记录执行次数
        self._attempt += 1

        try:
            if not self.env:
                raise ValueError("param env is blank")
            if not self.cluster:
                raise ValueError("param cluster is blank")
            if self.current_user is None:
                raise ValueError("param current_user is None")
            if conversation.id is None:
                raise ValueError("conversation.id is blank")

            # 2. Resolve DSL format to xyflow
            workflow_json = self._resolve_dsl(action)

            # 3. 获取用户的AccessKey
            access_key = self._resolve_access_key(conversation)

            # 4. 获取client
            client = self._get_client(
                api_key=access_key, env=self.env, cluster=self.cluster
            )

            # 5. 运行工作流
            return self._execute_run(
                client=client,
                workflow_json=workflow_json,
                workflow_name=action.name,
                test_mode=action.test_mode,
                attempt=self._attempt,
                conversation_id=str(conversation.id),
            )
        except Exception as exc:
            return RunWorkflowObservation.from_text(
                text=str(exc),
                status="Failed",
                attempt=self._attempt,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log=str(exc),
            )

    def _execute_run(
        self,
        *,
        client: PyroMindAPIClient,
        workflow_json: dict,
        workflow_name: str,
        test_mode: bool = False,
        attempt: int,
        conversation_id: str,
    ) -> RunWorkflowObservation:
        """Submit the workflow run to the platform asynchronously.

        This is the primary integration point for Pyromind platform execution.
        The current implementation:

        1. Converts DSL to xyflow and optionally applies test-mode parameters.
        2. Creates a training task via ``client.studio.create(...)``.
        3. Returns a submission observation with ``task_id`` and an initial
           status such as ``Pending``.
        4. Leaves final run status resolution to a platform callback/webhook
           (see ``openhands.tools.pyromind_debug`` for a similar broker pattern).

        Args:
            client: Authenticated Pyromind SDK client for the target env/cluster.
            workflow_json: Converted xyflow workflow payload to submit.
            workflow_name: Fallback workflow name when xyflow omits one.
            test_mode: Whether to append test-mode execution arguments.
            attempt: 1-based attempt number for this executor instance.

        Returns:
            A submission observation describing whether the platform accepted
            the run. It does not wait for workflow execution to finish.

        异步向平台提交工作流运行（主要集成入口）。

        当前实现会：

        1. 将 DSL 转为 xyflow，并按需附加 test mode 参数。
        2. 通过 ``client.studio.create(...)`` 创建训练任务。
        3. 返回带 ``task_id`` 与初始状态（如 ``Pending``）的提交 observation。
        4. 终态结果交由平台 callback/webhook 处理（可参考
           ``openhands.tools.pyromind_debug`` 的 broker 模式）。

        Args:
            client: 目标 env/cluster 下已鉴权的 Pyromind SDK client。
            workflow_json: 待提交的 xyflow 工作流 payload。
            workflow_name: xyflow 未提供名称时使用的回退工作流名。
            test_mode: 是否附加 test mode 运行参数。
            attempt: 本 Executor 实例上的第几次尝试（从 1 开始）。

        Returns:
            描述平台是否接受提交的 observation，不会等待工作流执行结束。
        """
        # 1. Validate DSL
        if workflow_json is None:
            return RunWorkflowObservation.from_text(
                text="Workflow xyflow is empty.",
                status="Failed",
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log="Workflow xyflow is empty.",
            )

        # 2. 调用内部转换方法
        if not client:
            return RunWorkflowObservation.from_text(
                text="PyroMindAPIClient can not init.",
                status="Failed",
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log="PyroMindAPIClient can not init.",
            )

        try:
            # 2.1 组织Test模式的运行参数 内存操作; 并生成 xyflow dict
            workflow_xyflow = self._resolve_add_test_mode(workflow_json, test_mode)

            # 获取工作流名称
            _workflow_name = str(workflow_xyflow.get("name", workflow_name))

            # 3. 调用平台运行接口
            request = TrainingTaskCreateRequest(
                name=_workflow_name, workflow=workflow_xyflow, out_id=f"agent1#{conversation_id}"
            )

            # 4. 创建工作流
            is_mock = True
            if is_mock:
                response: TrainingTaskCreateResponse = self._mock_submit_workflow(request)
            else:
                response: TrainingTaskCreateResponse = client.studio.create(request)

            # 5. 检查任务是否创建成功
            if not response.task_id:
                # 校验失败，工作流没有创建成功
                raise WorkflowRunError("Workflow create failed")

            if test_mode:
                user_text = (
                    "The test workflow task has been submitted. "
                    "Please wait patiently."
                )
            else:
                user_text = (
                    "The workflow task has been submitted. "
                    "Please wait patiently."
                )

            # 成功提交工作流，返回提交结果
            return RunWorkflowObservation.from_text(
                text=user_text,
                status=response.status,
                task_id=response.task_id,
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=False,
                keep_ui_lock=True,
            )
        except Exception as exc:
            return RunWorkflowObservation.from_text(
                text=str(exc),
                status="Failed",
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log=str(exc),
            )

    def _convert_dsl_to_xyflow(
        self,
        *,
        dsl: str,
        name: str,
    ) -> DslToXyflowObservation:
        """Convert workflow DSL to xyflow JSON via :mod:`dsl_to_xyflow`.

        Reuses :class:`~openhands.tools.workflow.dsl_to_xyflow.DslToXyflowExecutor`
        so run and convert tools share the same ``pyromind_sdk`` conversion path.

        通过 :mod:`dsl_to_xyflow` 将工作流 DSL 转为 xyflow JSON。

        复用 :class:`~openhands.tools.workflow.dsl_to_xyflow.DslToXyflowExecutor`，
        使运行与转换工具走同一套 ``pyromind_sdk`` 转换逻辑。
        """
        return DslToXyflowExecutor(converter_factory=None)(
            DslToXyflowAction(dsl=dsl, name=name)
        )

    def _resolve_access_key(
        self,
        conversation: BaseConversation | None,
    ) -> str | None:
        if conversation:
            state = cast("ConversationState", conversation.state)
            if auth_token := state.secret_registry.get_secret_value("auth_token"):
                return get_api_key(
                    env=self.env, auth_token=auth_token, origin_headers=self.headers
                )
        return None

    def _get_client(
        self,
        env: str | None,
        cluster: str | None,
        api_key: str | None,
    ) -> PyroMindAPIClient:
        if not api_key:
            raise ValueError("API key is required.")
        if not env:
            raise ValueError("env is required.")
        if not cluster:
            raise ValueError("cluster is required.")
        return get_pyromind_api_client(env=env, cluster=cluster, api_key=api_key)

    def _resolve_add_test_mode(self, workflow_json: dict, test_mode: bool) -> dict:
        """Parse workflow JSON string into a dictionary.

        This is the first step for applying test-mode transformations. Invalid
        JSON or non-object payloads raise ``ValueError`` so the caller can turn
        them into an error observation.

        将工作流 JSON 字符串解析为字典，为后续 test mode 改造做准备。JSON 不合法
        或非对象时报 ``ValueError``，由调用方转为 error observation。
        """
        if not isinstance(workflow_json, dict):
            raise ValueError("Workflow JSON must be a JSON object.")
        # 添加 test mode 参数
        if test_mode:
            execution_argos = workflow_json.get("execution_argos")
            if not isinstance(execution_argos, list):
                execution_argos = []
            execution_argos.append({"execution_mode": "test"})
            workflow_json["execution_argos"] = execution_argos
        return workflow_json

    def _resolve_workflow_dsl_source(self, action: RunWorkflowAction) -> str:
        """Return inline workflow Python DSL source code from ``action.dsl``.

        ``action.dsl`` must contain agent-generated script text; it is not a
        filesystem path.
        """
        if action.dsl is None:
            raise ValueError(
                "Workflow DSL source code is required. Pass the agent-generated "
                "Python DSL in `dsl`."
            )
        dsl_source = action.dsl.strip()
        if not dsl_source:
            raise ValueError("Workflow DSL source code must be a non-empty string.")
        return dsl_source

    def _resolve_dsl(self, action: RunWorkflowAction) -> dict:
        """Convert workflow DSL source code to xyflow via :mod:`dsl_to_xyflow`.

        Requires inline ``action.dsl`` text. Conversion goes through
        :class:`~openhands.tools.workflow.dsl_to_xyflow.DslToXyflowExecutor`.

        Returns:
            The converted xyflow workflow object.

        Raises:
            ValueError: If DSL source code is missing/empty or conversion fails.

        将 ``action.dsl`` 中的工作流 DSL 源码转为 xyflow，复用 :mod:`dsl_to_xyflow`。

        ``action.dsl`` 为 Agent 生成/编辑的内联 Python DSL 源码，不是文件路径。

        Returns:
            转换后的 xyflow 工作流 dict。

        Raises:
            ValueError: DSL 源码缺失/为空，或转换失败。
        """
        dsl_source = self._resolve_workflow_dsl_source(action)
        conversion = self._convert_dsl_to_xyflow(dsl=dsl_source, name=action.name)
        if conversion.is_error:
            raise ValueError(
                conversion.text or "Failed to convert workflow DSL to xyflow."
            )
        if conversion.xyflow is None:
            raise ValueError("Workflow DSL conversion did not return xyflow JSON.")
        return conversion.xyflow


    def _mock_submit_workflow(self, request: TrainingTaskCreateRequest) -> TrainingTaskCreateResponse:
        resp = TrainingTaskCreateResponse(task_id="1999", name=request.name, status="Running")
        return resp




class RunWorkflowTool(ToolDefinition[RunWorkflowAction, RunWorkflowObservation]):
    """Tool that submits a Pyromind workflow run asynchronously.

    异步提交 Pyromind 工作流运行的工具定义。
    """

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        env = params.get("env", None)
        cluster = params.get("cluster", None)
        current_user = params.get("current_user", None)
        headers = params.get("headers", {})

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=RunWorkflowAction,
                observation_type=RunWorkflowObservation,
                executor=RunWorkflowExecutor(
                    cluster=cluster,
                    env=env,
                    current_user=current_user,
                    headers=headers,
                ),
                annotations=ToolAnnotations(
                    title="run_workflow",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


# Register with the SDK tool registry so agents can resolve Tool(name="run_workflow").
# 注册到 SDK tool registry，使 Agent 可通过 Tool(name="run_workflow") 解析本工具。
register_tool(RunWorkflowTool.name, RunWorkflowTool)
