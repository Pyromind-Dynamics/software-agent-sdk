# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from wandb-primary skill (wandb_helpers_impl.py).
# https://github.com/coreweave/skills/tree/main/skills/wandb-primary

"""Wandb 数据源适配器。

职责:凭证提取(从平台节点 config / 输出)、run 定位(日志正则)、SDK 连接、
数据拉取,并统一转换为 ``RunData`` 标准格式。数据分析层不感知本模块。

新数据源适配器照此骨架实现同名方法即可。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .base import RunData


RUN_SETUP_RE = re.compile(r"wandb:\s+setting up run\s+(\w+)")
RUN_URL_RE = re.compile(
    r"View run at\s+https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/(\w+)"
)
RUN_PATH_RE = re.compile(r"run-\d{8}_\d{6}-(\w+)")
CONFIG_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")

WANDB_NODE_TYPE = "WandbConfigBuilderNode"


def _ensure_wandb_runtime_dirs(creds: dict[str, str]) -> None:
    """受限沙箱中重定向 wandb 临时/缓存/日志目录到可写位置。

    优先 resolve-target 写入 creds 的 wandb_tmp;已有可用 TMPDIR 时不动它,
    只补缓存/日志目录。创建失败时静默跳过,交给 wandb 原生报错。
    """
    base = creds.get("wandb_tmp") or os.environ.get("WANDB_TMP") or "wandb_tmp"
    path = Path(base).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        for sub in ("cache", "wandb", "config"):
            (path / sub).mkdir(exist_ok=True)
    except OSError:
        return
    if not os.access(str(path), os.W_OK):
        return
    tmpdir = os.environ.get("TMPDIR", "")
    if not tmpdir or not os.access(tmpdir, os.W_OK):
        os.environ["TMPDIR"] = str(path)
    os.environ.setdefault("WANDB_CACHE_DIR", str(path / "cache"))
    os.environ.setdefault("WANDB_DIR", str(path / "wandb"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(path / "config"))


class WandbDataSource:
    """W&B 数据源:平台节点凭证 → wandb SDK → RunData。"""

    name = "wandb"

    def __init__(self) -> None:
        self._api: Any = None

    # ------------------------------------------------------------------
    # 自动探测 / 凭证提取
    # ------------------------------------------------------------------

    def extract_creds(
        self,
        nodes: list[dict[str, Any]],
        client: Any = None,
        task_id: str = "",
    ) -> dict[str, str] | None:
        """从平台节点提取 WANDB_* 凭证;识别不到返回 None(探测时自动跳过)。

        client/task_id 存在时,节点 config 未含凭证则回退查
        WandbConfigBuilderNode 的输出接口。
        """
        creds: dict[str, str] = {}
        for node in nodes:
            data = node.get("data")
            if isinstance(data, dict) and isinstance(data.get("config"), dict):
                creds.update(self._collect_config_env(data["config"]))
            if creds:
                break

        if not creds and client is not None and task_id:
            for node in nodes:
                node_id = self._node_identifier(node)
                if WANDB_NODE_TYPE in self._node_type(node) and node_id:
                    creds = self._collect_output_env(
                        client.node_output(node_id, task_id)
                    )
                    break

        return {k: v for k, v in creds.items() if v} or None

    def _collect_config_env(self, config: dict[str, Any]) -> dict[str, str]:
        """从节点 data.config 提取 wandb 相关环境变量。"""
        return self._collect_wandb_env(config)

    def _collect_wandb_env(self, output: dict[str, Any]) -> dict[str, str]:
        """从节点输出/config 提取 WANDB_* 环境变量,覆盖多种真实形态。"""
        result: dict[str, str] = {}

        def merge_text(text: str) -> None:
            for key, value in self._parse_config_text(text).items():
                if value:
                    result.setdefault(key.upper(), value)

        config_value = output.get("wandb_config")
        if isinstance(config_value, dict):
            for key, value in config_value.items():
                result[str(key).upper()] = str(value)
        elif isinstance(config_value, str):
            merge_text(config_value)

        for key in ("wandb_api_key", "wandb_project", "wandb_name"):
            if output.get(key) is not None:
                result["WANDB_" + key.removeprefix("wandb_").upper()] = str(output[key])

        entries = output.get("entries")
        if isinstance(entries, list):
            merge_text(
                "".join(
                    e.get("m", "")
                    for e in entries
                    if isinstance(e, dict) and isinstance(e.get("m"), str)
                )
            )

        if not result:
            for value in output.values():
                if isinstance(value, str) and value.strip():
                    merge_text(value)
                    break

        # 只保留 wandb 相关键,避免混入其他节点配置
        return {k: v for k, v in result.items() if k.startswith("WANDB_")}

    @staticmethod
    def _parse_config_text(text: str) -> dict[str, str]:
        """解析 wandb_config YAML 文本(仅 key: value 行,嵌套不做展开)。"""
        result: dict[str, str] = {}
        for line in text.splitlines():
            match = CONFIG_LINE_RE.match(line)
            if match:
                result[match.group(1)] = match.group(2)
        return result

    @staticmethod
    def _node_identifier(node: dict[str, Any]) -> str | None:
        for key in ("node_code", "node_id", "id", "nodeId", "code"):
            value = node.get(key)
            if value is not None and str(value) != "":
                return str(value)
        return None

    @staticmethod
    def _node_type(node: dict[str, Any]) -> str:
        data = node.get("data")
        if isinstance(data, dict):
            node_type = data.get("nodeType")
            if isinstance(node_type, str) and node_type:
                return node_type
        for key in ("node_type", "type", "nodeType", "node_class"):
            value = node.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _collect_output_env(self, output: dict[str, Any]) -> dict[str, str]:
        """从 node_output 接口响应提取凭证。"""
        if isinstance(output, dict):
            return self._collect_wandb_env(output)
        return {}

    # ------------------------------------------------------------------
    # run 定位
    # ------------------------------------------------------------------

    def extract_run_ref(self, log_text: str) -> tuple[str, str, str] | None:
        """从日志中提取 (entity, project, run_id);URL 行优先,其次 setting up run。"""
        for match in RUN_URL_RE.finditer(log_text):
            return match.group(1), match.group(2), match.group(3)
        match = RUN_SETUP_RE.search(log_text)
        if match:
            return "", "", match.group(1)
        match = RUN_PATH_RE.search(log_text)
        if match:
            return "", "", match.group(1)
        return None

    # ------------------------------------------------------------------
    # SDK 连接
    # ------------------------------------------------------------------

    def connect(self, creds: dict[str, str]) -> None:
        """用凭证初始化 wandb.Api(WANDB_SILENT 抑制日志噪音)。"""
        os.environ.setdefault("WANDB_SILENT", "true")
        _ensure_wandb_runtime_dirs(creds)
        from wandb import Api

        api_key = creds.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            raise ValueError(
                "WANDB_API_KEY missing: 先运行 resolve-target --creds-out,"
                "或传入 --api-key / 设置环境变量 WANDB_API_KEY"
            )
        timeout = int(os.environ.get("WANDB_API_TIMEOUT", "60"))
        self._api = Api(api_key=api_key, timeout=timeout)

    def resolve_entity(self, project: str, fallback: str = "") -> tuple[str, str]:
        """通过 wandb API 确认 entity(优先 api.viewer(),其次显式 fallback)。"""
        if fallback and project:
            return fallback, project
        if self._api is None:
            raise ValueError("connect() 未初始化 wandb API")
        viewer = self._api.viewer()
        username = getattr(viewer, "username", "") or ""
        if not username:
            raise ValueError("无法从 wandb API 推断 entity,请用 --entity 显式指定")
        return username, project

    # ------------------------------------------------------------------
    # 数据拉取 → RunData
    # ------------------------------------------------------------------

    def probe(self, project_path: str, run_id: str | None = None) -> dict[str, Any]:
        """探查项目规模/指标键/config 键;指定 run_id 时输出该 run 的键。"""
        if self._api is None:
            raise ValueError("connect() 未初始化 wandb API")
        result: dict[str, Any] = {"path": project_path, "warnings": []}

        runs = self._api.runs(
            project_path,
            filters={"state": "finished"},
            order="-created_at",
            per_page=3,
        )
        sample = runs[:3]
        if not sample:
            result["run_count_estimate"] = 0
            result["warnings"].append("No finished runs found")
            return result

        all_metric_keys: set[str] = set()
        all_config_keys: set[str] = set()
        has_history = False

        for run in sample:
            metric_keys = {
                k for k in run.summary_metrics.keys() if not k.startswith("_")
            }
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
                "Runs have "
                f"{n_metrics} metrics — ALWAYS pass keys= to history/scan_history"
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

        if run_id:
            run = self._api.run(f"{project_path}/{run_id}")
            result["run_summary_keys"] = sorted((run.summary or {}).keys())
            result["run_config_keys"] = sorted((run.config or {}).keys())
        return result

    def get_run_data(
        self,
        project_path: str,
        run_id: str,
        keys: list[str],
    ) -> RunData:
        """拉取单个 run 并转换为 RunData。

        keys 为空时只返回元数据(config/summary),history 为空列表;
        keys 非空时按显式键拉取历史(大项目必须限制 keys,防 502)。
        """
        if self._api is None:
            raise ValueError("connect() 未初始化 wandb API")
        run = self._api.run(f"{project_path}/{run_id}")
        data: RunData = {
            "run_id": run_id,
            "display_name": str(run.display_name or ""),
            "state": str(run.state or ""),
            "config": self._json_safe(dict(run.config or {})),
            "summary": self._json_safe(dict(run.summary or {})),
            "history": [],
        }
        if keys:
            data["history"] = self.scan_history(run, keys=keys)
        return data

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def scan_history(
        self,
        run: Any,
        keys: list[str],
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        """按显式 metric keys 读取 history 行(大项目必须限制 keys)。"""
        if not keys:
            raise ValueError("keys is required — never scan without explicit keys")
        rows: list[dict[str, Any]] = []
        scanner = run.scan_history(keys=keys, page_size=min(max_rows or 10_000, 10_000))
        for row in scanner:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= max_rows:
                break
        return rows

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """递归转原生 JSON 类型(wandb SummarySubDict/Config 等 dict 子类)。"""
        if isinstance(value, dict):
            return {
                str(key): WandbDataSource._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [WandbDataSource._json_safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
