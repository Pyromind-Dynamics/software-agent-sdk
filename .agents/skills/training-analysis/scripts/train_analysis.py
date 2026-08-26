"""训练数据分析 CLI — Pyromind skill 辅助工具。

子命令:
- resolve-target <task_id>: 自动探测数据源并通过平台 API 定位
  run(entity/project/run_id)与凭证
- probe <entity/project> [--run-id]: 探查 project/run 的指标与 config 键
- analyze-run <entity/project> <run_id>: 单 run 稳定性诊断
- report <entity/project> <run_id> --out <md>: 四阶段分析报告

数据源抽象: resolve-target 根据平台节点 config 自动探测数据源(wandb 等),
分析命令从 creds 文件读 data_source 字段选择适配器,数据分析层只消费
标准 RunData 格式,与具体数据源解耦(见 data_sources/)。

平台 API 认证与 validate_workflow 工具对齐: cookie / x-cluster /
authorization 三个 header 小写透传,分别从环境变量 PYROMIND_COOKIE /
X_CLUSTER / PYROMIND_AUTHORIZATION 读取,或通过 --cookie / --cluster /
--authorization 参数传入;base URL 按 APP_ENV(prod/production/online 走
正式环境,否则 pre 环境)推断,可用 --api-base 或 PYROMIND_API_BASE 覆盖。
凭证优先从 --api-key 或 --creds-file(resolve-target 的 --creds-out
产物)读取,否则回退环境变量。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis_helpers import diagnose_run
from data_sources import create_data_source, detect_data_source


PRE_API_BASE = "https://pre-api-portal.pyromind.ai/std2/studio_api/"
PROD_API_BASE = "https://api-portal.pyromind.ai/std2/studio_api/"
_PROD_APP_ENVS = {"prod", "production", "online"}
_CREDENTIAL_KEY_MARKERS = (
    "API_KEY",
    "APIKEY",
    "AUTHORIZATION",
    "COOKIE",
    "PASSWORD",
    "SECRET",
    "TOKEN",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "ACCESS_KEY",
    "REFRESH_TOKEN",
    "CLUSTER",
)
_INLINE_CREDENTIAL = re.compile(
    r"((?:api[_-]?key|authorization|cookie|password|secret|token|cluster)"
    r"\s*[:=]\s*)"
    r"([^\s,;\"']+)",
    re.IGNORECASE,
)


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# 训练节点类型关键词(平台通用概念,与具体数据源无关)
TRAIN_NODE_KEYWORDS = ("Train", "train")


class WandbAnalysisError(Exception):
    """脚本内可预期错误,以友好消息退出。"""


def _ssl_context() -> ssl.SSLContext:
    """创建 SSL context:优先 certifi,兜底系统默认路径。"""
    try:
        import certifi  # noqa: F401

        return ssl.create_default_context(cafile=certifi.where())
    except (ImportError, FileNotFoundError):
        pass
    for path in (
        "/etc/ssl/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/usr/local/etc/openssl@3/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ):
        if os.path.exists(path):
            return ssl.create_default_context(cafile=path)
    return ssl.create_default_context()


@dataclass
class PlatformClient:
    """Pyromind studio_api 客户端(task_workflow_result / 节点日志 / 节点输出)。"""

    base_url: str = PRE_API_BASE
    cookie: str = ""
    cluster: str = ""
    authorization: str = ""

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.cookie:
            headers["cookie"] = self.cookie
        if self.cluster:
            headers["x-cluster"] = self.cluster
        if self.authorization:
            headers["authorization"] = self.authorization
        return headers

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = urllib.parse.urljoin(self.base_url, path)
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}" if query else url
        request = urllib.request.Request(full_url, headers=self._headers())
        try:
            with urllib.request.urlopen(
                request, timeout=60, context=_ssl_context()
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise WandbAnalysisError(
                f"platform API {path} failed: HTTP {exc.code} {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise WandbAnalysisError(
                f"platform API {path} unreachable: {exc.reason}"
            ) from exc

    def task_workflow_result(self, task_id: str) -> dict[str, Any]:
        """返回任务工作流信息(节点列表来源)。"""
        return self._get("api/task_workflow_result", {"task_id": task_id})

    def node_log(self, node_id: str, task_id: str) -> str:
        """返回节点 stdout 日志文本。"""
        data = self._get(
            "internal/logs/node/raw", {"nodeId": node_id, "taskId": task_id}
        )
        return _extract_log_text(data)

    def node_output(self, node_code: str, task_id: str) -> dict[str, Any]:
        """返回节点输出结构。"""
        data = self._get(
            "internal/output/node/raw", {"node_code": node_code, "task_id": task_id}
        )
        if isinstance(data, dict):
            return data
        return {"raw": data}


def _extract_log_text(data: Any) -> str:
    """从节点日志 API 响应中提取纯文本;真实响应为 entries[].m 数组。"""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        entries = data.get("entries")
        if isinstance(entries, list):
            parts = [
                e.get("m", "")
                for e in entries
                if isinstance(e, dict) and isinstance(e.get("m"), str)
            ]
            if parts:
                return "".join(parts)
        for key in ("log", "logs", "content", "data", "stdout", "text", "raw"):
            if isinstance(data.get(key), str):
                return data[key]
        # 深层遍历找第一个字符串值
        for value in data.values():
            if isinstance(value, str):
                return value
    return ""


def _find_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从 task_workflow_result 响应中容错提取节点列表。

    平台响应结构以实施校准为准;此处覆盖常见形态:顶层 nodes 数组、
    workflow 内嵌、或带 data/data 包层的数组。
    """
    candidates: list[Any] = []
    for key in ("nodes", "node_list", "node_infos"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.append(value)
    workflow = payload.get("workflow")
    if isinstance(workflow, dict):
        for key in ("nodes", "node_list"):
            value = workflow.get(key)
            if isinstance(value, list):
                candidates.append(value)
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("nodes", "node_list"):
            value = data.get(key)
            if isinstance(value, list):
                candidates.append(value)
    if isinstance(data, list):
        candidates.append(data)
    for candidate in candidates:
        if candidate and all(isinstance(item, dict) for item in candidate):
            return candidate
    return []


def _node_identifier(node: dict[str, Any]) -> str | None:
    """节点标识:优先 node_code,其次 node_id/id;按实施校准调整优先级。"""
    for key in ("node_code", "node_id", "id", "nodeId", "code"):
        value = node.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


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


def _is_train_node(node: dict[str, Any]) -> bool:
    return any(keyword in _node_type(node) for keyword in TRAIN_NODE_KEYWORDS)


def _credential_file() -> Path:
    return Path(tempfile.gettempdir()) / "train_analysis_creds.json"


def _load_creds(path: Path | None = None) -> dict[str, str]:
    target = Path(path) if path else _credential_file()
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _source_from_args(args: argparse.Namespace) -> tuple[Any, dict[str, str]]:
    """从 creds 文件创建数据源;creds 缺 data_source 时默认 wandb(向后兼容)。"""
    creds = _load_creds(getattr(args, "creds_file", None))
    name = getattr(args, "data_source", "") or creds.get("data_source") or "wandb"
    return create_data_source(name), creds


def _history_float_values(history: list[dict[str, Any]], metric: str) -> list[float]:
    """从 history 行中提取数值型指标值(跳过非数值/None)。"""
    values: list[float] = []
    for row in history:
        value = row.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _history_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    nans = sum(1 for v in values if v != v)
    clean = [v for v in values if v == v]
    if not clean:
        return {"count": len(values), "nan": nans}
    min_v = min(clean)
    max_v = max(clean)
    final_v = clean[-1]
    spikes = 0
    for prev, curr in zip(clean, clean[1:]):
        if prev != 0 and abs(curr - prev) / abs(prev) > 10:
            spikes += 1
    return {
        "count": len(values),
        "nan": nans,
        "min": min_v,
        "max": max_v,
        "final": final_v,
        "spikes": spikes,
    }


def _loss_metric_key(summary: dict[str, Any]) -> str:
    """从 summary 选默认 loss 键。"""
    return next((k for k in summary if "loss" in k), "")


def _probe_suggestions(
    config: dict[str, Any], stats: dict[str, Any]
) -> list[dict[str, str]]:
    """基于单 run 统计生成单变量探针实验建议(保守启发式)。"""
    suggestions: list[dict[str, str]] = []
    lr = config.get("learning_rate")
    if isinstance(lr, (int, float)) and lr > 0:
        suggestions.append(
            {
                "parameter": "learning_rate",
                "change": f"{lr} → {lr / 2}",
                "rationale": "loss 抖动/发散时降低学习率是首选单变量探针",
            }
        )
    if stats.get("nan", 0) > 0:
        suggestions.append(
            {
                "parameter": "batch_size",
                "change": "减半并同步增大 grad_accumulation_steps 保持有效 batch",
                "rationale": "NaN 常与过大梯度相关,减小 batch 可降低单步方差",
            }
        )
    epochs = config.get("num_epochs")
    if isinstance(epochs, (int, float)) and stats.get("count", 0) > 0:
        suggestions.append(
            {
                "parameter": "num_epochs",
                "change": f"{epochs} → {int(epochs) + 1}",
                "rationale": "loss 仍在下降(未平台期)时延长训练",
            }
        )
    return suggestions


def _redact_report(output: str, creds: dict[str, str]) -> str:
    """Remove credential-like values before a report reaches a caller."""
    credential_items = [
        *(
            (key, value)
            for key, value in os.environ.items()
            if any(marker in key.upper() for marker in _CREDENTIAL_KEY_MARKERS)
        ),
        *creds.items(),
    ]
    secrets = [
        value
        for key, value in credential_items
        if value and any(marker in key.upper() for marker in _CREDENTIAL_KEY_MARKERS)
    ]
    for secret in sorted(set(secrets), key=len, reverse=True):
        output = output.replace(secret, "[REDACTED]")
    return _INLINE_CREDENTIAL.sub(r"\1[REDACTED]", output)


def _redact_value_for_error(value: Any, key: str = "", depth: int = 0) -> Any:
    """Keep platform diagnostics useful without copying credential fields."""
    if any(marker in key.upper() for marker in _CREDENTIAL_KEY_MARKERS):
        return "[REDACTED]"
    if depth >= 8:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            str(name): _redact_value_for_error(child, str(name), depth + 1)
            for name, child in list(value.items())[:100]
        }
    if isinstance(value, list):
        return [
            _redact_value_for_error(child, key, depth + 1) for child in value[:200]
        ]
    if isinstance(value, str):
        return _INLINE_CREDENTIAL.sub(r"\1[REDACTED]", value[:100_000])
    return value


def _parse_run_url(run_url: str) -> tuple[str, str, str]:
    """Extract entity/project/run id from a W&B run URL hint."""
    if not run_url.strip():
        return "", "", ""
    parsed = urllib.parse.urlparse(run_url.strip())
    path = parsed.path or run_url.strip()
    segments = [segment for segment in path.rstrip("/").split("/") if segment]
    if len(segments) < 4 or segments[-2] != "runs":
        return "", "", ""
    return segments[-4], segments[-3], segments[-1]


class TrainingAnalysisService:
    """Reusable training-analysis business service shared by CLI and worker."""

    def _source(
        self,
        creds_file: str | Path | None,
        data_source: str | None,
    ) -> tuple[Any, dict[str, str]]:
        creds = _load_creds(Path(creds_file) if creds_file else None)
        name = data_source or creds.get("data_source") or "wandb"
        return create_data_source(name), creds

    def resolve_target(
        self,
        *,
        api_base: str | None = None,
        cookie: str = "",
        cluster: str = "",
        authorization: str = "",
        api_key: str = "",
        creds_file: str = "",
        entity: str = "",
        data_source: str = "",
        task_id: str,
        run_url: str = "",
        creds_out: str = "",
    ) -> dict[str, Any]:
        """Resolve a task to a data-source run and optionally write credentials."""
        _ = creds_file  # retained for CLI compatibility
        if not task_id.strip():
            raise ValueError("task_id is required")
        client = PlatformClient(
            base_url=api_base or _default_api_base(),
            cookie=cookie,
            cluster=cluster,
            authorization=authorization,
        )
        payload = client.task_workflow_result(task_id)
        nodes = _find_nodes(payload)
        if not nodes:
            safe_payload = json.dumps(
                _redact_value_for_error(payload), ensure_ascii=False
            )
            raise WandbAnalysisError(
                "task_workflow_result 响应中未找到节点列表,响应结构待校准:"
                f"{safe_payload[:800]}"
            )

        source_name = data_source or detect_data_source(nodes)
        if not source_name:
            raise WandbAnalysisError(
                "未能从平台节点自动识别数据源类型,请用 --data-source 指定"
                "(如 --data-source wandb)"
            )
        source = create_data_source(source_name)
        creds = source.extract_creds(nodes, client=client, task_id=task_id) or {}
        extracted_creds = bool(creds)
        effective_api_key = (
            api_key
            or creds.get("WANDB_API_KEY", "")
            or os.environ.get("WANDB_API_KEY", "")
        )
        if effective_api_key:
            creds["WANDB_API_KEY"] = effective_api_key

        run_id = ""
        run_entity = ""
        run_project = ""
        for node in nodes:
            node_id = _node_identifier(node)
            if not node_id or not _is_train_node(node):
                continue
            if run_id:
                break
            info = source.extract_run_ref(client.node_log(node_id, task_id))
            if info and info[2]:
                run_entity, run_project, run_id = info

        if not run_id and not extracted_creds:
            raise WandbAnalysisError(
                "未能从平台 API 定位 run:训练节点日志无数据源 run 记录,"
                "且未在节点 config 找到凭证。请提供 --run-url 或检查 task_id。"
            )

        run_url_entity, run_url_project, run_url_id = _parse_run_url(run_url)
        if run_url_id and not run_id:
            run_id = run_url_id
        if run_url_entity and not run_entity:
            run_entity = run_url_entity
        project = creds.get("WANDB_PROJECT") or run_project or run_url_project
        resolved_entity = creds.get("WANDB_ENTITY") or run_entity or entity
        if effective_api_key and project:
            try:
                source.connect(creds)
                resolved_entity, project = source.resolve_entity(
                    project, fallback=resolved_entity
                )
            except Exception:
                pass

        result: dict[str, Any] = {
            "task_id": task_id,
            "data_source": source_name,
            "entity": resolved_entity,
            "project": project,
            "run_id": run_id,
            "env": {
                key: value
                for key, value in creds.items()
                if not any(marker in key.upper() for marker in _CREDENTIAL_KEY_MARKERS)
            },
        }
        if creds_out:
            creds_path = Path(creds_out)
            creds_path.parent.mkdir(parents=True, exist_ok=True)
            wandb_tmp = creds_path.parent / "wandb_tmp"
            wandb_tmp.mkdir(parents=True, exist_ok=True)
            creds_path.write_text(
                json.dumps(
                    {
                        **creds,
                        "data_source": source_name,
                        "WANDB_API_KEY": creds.get("WANDB_API_KEY", ""),
                        "api_key": creds.get("WANDB_API_KEY", ""),
                        "wandb_tmp": str(wandb_tmp),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.chmod(creds_path, 0o600)
            result["creds_file"] = str(creds_path)
            result["wandb_tmp"] = str(wandb_tmp)
        return result

    def resolve(self, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias for callers that use the shorter operation name."""
        return self.resolve_target(**kwargs)

    def probe(
        self,
        *,
        creds_file: str | Path | None = None,
        data_source: str | None = None,
        entity_project: str,
        run_id: str = "",
    ) -> dict[str, Any]:
        if not entity_project.strip():
            raise ValueError("entity_project is required")
        source, creds = self._source(creds_file, data_source)
        source.connect(creds)
        return source.probe(entity_project, run_id=run_id or None)

    def analyze_run(
        self,
        *,
        creds_file: str | Path | None = None,
        data_source: str | None = None,
        entity_project: str,
        run_id: str,
        metric: str = "",
        keys: str | list[str] | None = None,
    ) -> dict[str, Any]:
        if not entity_project.strip() or not run_id.strip():
            raise ValueError("entity_project and run_id are required")
        source, creds = self._source(creds_file, data_source)
        source.connect(creds)
        data = source.get_run_data(entity_project, run_id, keys=[])
        selected_metric = metric or _loss_metric_key(data["summary"])
        if isinstance(keys, str):
            history_keys = [key.strip() for key in keys.split(",") if key.strip()]
        else:
            history_keys = [
                str(key).strip() for key in (keys or []) if str(key).strip()
            ]
        if len(history_keys) > 20:
            raise ValueError("keys must contain at most 20 metrics")
        if not history_keys:
            history_keys = [selected_metric] if selected_metric else []
        if history_keys:
            data = source.get_run_data(entity_project, run_id, keys=history_keys)

        report: dict[str, Any] = {
            "run": run_id,
            "display_name": data["display_name"],
            "state": data["state"],
            "config": data["config"],
            "summary": data["summary"],
            "metric": selected_metric or None,
        }
        history = data["history"]
        if history:
            if selected_metric in history_keys:
                values = _history_float_values(history, selected_metric)
                report["metric_stats"] = _history_stats(values)
                report["diagnostics"] = diagnose_run(data, train_key=selected_metric)
            if len(history_keys) > 1:
                multi: dict[str, list[float]] = {}
                for key in history_keys:
                    values = _history_float_values(history, key)
                    if values:
                        multi[key] = values
                if multi:
                    report["multi_metrics"] = multi
            first_step = history[0].get("_step")
            last_step = history[-1].get("_step")
            if first_step is not None:
                report["steps"] = [first_step, last_step]
        return report

    def analyze(
        self,
        *,
        creds_file: str | Path | None = None,
        data_source: str | None = None,
        entity_project: str,
        run_id: str,
        metric: str = "",
        keys: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Public service spelling used by the business tool."""
        return self.analyze_run(
            creds_file=creds_file,
            data_source=data_source,
            entity_project=entity_project,
            run_id=run_id,
            metric=metric,
            keys=keys,
        )

    def report(
        self,
        *,
        creds_file: str | Path | None = None,
        data_source: str | None = None,
        entity_project: str,
        run_id: str,
        metric: str = "",
        output_path: str | Path | None = None,
    ) -> str:
        creds = _load_creds(Path(creds_file) if creds_file else None)
        source_name = data_source or creds.get("data_source") or "wandb"
        analysis = self.analyze_run(
            creds_file=creds_file,
            data_source=data_source,
            entity_project=entity_project,
            run_id=run_id,
            metric=metric,
            keys=[metric] if metric else [],
        )
        selected_metric = str(analysis.get("metric") or "")
        diagnostics = analysis.get("diagnostics") or {}
        config = analysis.get("config") or {}
        stats = {
            "total_steps": diagnostics.get("total_steps"),
            "final_value": diagnostics.get("final_value"),
            "min_value": diagnostics.get("min_value"),
            "has_nan": diagnostics.get("has_nan"),
            "converged": diagnostics.get("converged"),
            "likely_overfit": diagnostics.get("likely_overfit"),
            "train_val_gap": diagnostics.get("train_val_gap"),
            "error": diagnostics.get("error"),
        }
        suggestions = _probe_suggestions(config, stats)
        lines = [
            f"# 训练分析报告:{entity_project}/{run_id}",
            "",
            "## 1. 先验假设",
            "- 训练类型:由 config 推断(SFT/DPO/GRPO,见 config 摘要)",
            f"- 关注指标: {selected_metric or '未知'}",
            "- 基线预期: loss 单调下降或收敛至平台期",
            "",
            "## 2. 数据惊奇",
            f"- run state: {analysis.get('state', '')}, display_name: "
            f"{analysis.get('display_name', '')}",
            f"- 诊断: {json.dumps(stats, ensure_ascii=False)}",
            "",
            "## 3. 候选机制",
        ]
        if diagnostics.get("has_nan"):
            lines.append("- NaN 出现 → 数值不稳定(梯度爆炸/学习率过大/数据含异常样本)")
            lines.append("  - 同时建议检查训练数据集质量(重复样本/异常值/格式问题)")
        if diagnostics.get("likely_overfit"):
            lines.append("- 疑似过拟合 → train/val gap 扩大,考虑早停/正则化/增大数据量")
            lines.append(
                "  - 数据集质量检查:重复样本/标签噪声可能导致过拟合, "
                "建议检查并清洗训练数据"
            )
        if diagnostics.get("converged") is False:
            lines.append("- 未收敛 → 训练不足,考虑延长 epoch 或调整学习率")
        if not diagnostics.get("has_nan") and not diagnostics.get("likely_overfit"):
            lines.append("- 未发现明显异常,可能与基线/相邻 run 的指标对比确认")
            lines.append(
                "  - 数据集质量基线检查:若后续调优效果不达预期,"
                " 建议检查训练数据集质量(重复样本/标签噪声/分布)"
            )
        lines += ["", "## 4. 可证伪探针实验"]
        if suggestions:
            for suggestion in suggestions:
                lines.append(
                    f"- **{suggestion['parameter']}**: {suggestion['change']}"
                    f" — {suggestion['rationale']}"
                )
        else:
            lines.append("- 无自动建议,请人工检查 config 或对比其他 run")
        lines += [
            "",
            "## 审计线索",
            f"- entity/project: `{entity_project}`",
            f"- run id: `{run_id}`",
            f"- 指标键: `{selected_metric or 'N/A'}`",
            f"- 数据来源: {source_name} 数据源,"
            "凭证来自平台节点 config/输出",
        ]
        output = _redact_report("\n".join(lines) + "\n", creds)
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(output, encoding="utf-8")
        return output


def cmd_resolve_target(args: argparse.Namespace) -> int:
    result = TrainingAnalysisService().resolve_target(
        api_base=args.api_base,
        cookie=args.cookie,
        cluster=args.cluster,
        authorization=args.authorization,
        api_key=args.api_key,
        creds_file=args.creds_file,
        entity=args.entity,
        data_source=args.data_source,
        task_id=args.task_id,
        run_url=args.run_url,
        creds_out=args.creds_out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    result = TrainingAnalysisService().probe(
        creds_file=args.creds_file,
        data_source=args.data_source,
        entity_project=args.entity_project,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_analyze_run(args: argparse.Namespace) -> int:
    result = TrainingAnalysisService().analyze(
        creds_file=args.creds_file,
        data_source=args.data_source,
        entity_project=args.entity_project,
        run_id=args.run_id,
        metric=args.metric,
        keys=args.keys,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    output = TrainingAnalysisService().report(
        creds_file=args.creds_file,
        data_source=args.data_source,
        entity_project=args.entity_project,
        run_id=args.run_id,
        metric=args.metric,
        output_path=args.out,
    )
    if args.out:
        print(f"report written to {args.out}")
    else:
        print(output)
    return 0


def _default_api_base() -> str:
    configured = os.environ.get("PYROMIND_API_BASE", "").strip()
    if configured:
        return configured
    app_env = os.environ.get("APP_ENV", "").lower()
    return PROD_API_BASE if app_env in _PROD_APP_ENVS else PRE_API_BASE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base", default=os.environ.get("PYROMIND_API_BASE", _default_api_base())
    )
    parser.add_argument("--cookie", default=os.environ.get("PYROMIND_COOKIE", ""))
    parser.add_argument("--cluster", default=os.environ.get("X_CLUSTER", ""))
    parser.add_argument(
        "--authorization", default=os.environ.get("PYROMIND_AUTHORIZATION", "")
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--creds-file", default="")
    parser.add_argument("--entity", default="")
    parser.add_argument(
        "--data-source",
        default="",
        help="数据源类型(如 wandb);留空则自动探测或读 creds",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_resolve = subparsers.add_parser("resolve-target", help="定位 wandb run 与凭证")
    p_resolve.add_argument("task_id")
    p_resolve.add_argument("--run-url", default="")
    p_resolve.add_argument("--creds-out", default="")
    p_resolve.set_defaults(func=cmd_resolve_target)

    p_probe = subparsers.add_parser("probe", help="探查 project/run 数据契约")
    p_probe.add_argument("entity_project")
    p_probe.add_argument("--run-id", default="")
    p_probe.set_defaults(func=cmd_probe)

    p_analyze = subparsers.add_parser("analyze-run", help="单 run 稳定性诊断")
    p_analyze.add_argument("entity_project")
    p_analyze.add_argument("run_id")
    p_analyze.add_argument("--metric", default="")
    p_analyze.add_argument(
        "--keys",
        default="",
        help="逗号分隔的多指标键,如 train/loss,train/entropy,train/learning_rate",
    )
    p_analyze.set_defaults(func=cmd_analyze_run)

    p_report = subparsers.add_parser("report", help="四阶段分析报告")
    p_report.add_argument("entity_project")
    p_report.add_argument("run_id")
    p_report.add_argument("--metric", default="")
    p_report.add_argument("--out", default="")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WandbAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(
            f"error: 缺少依赖,请先安装 wandb SDK: pip install wandb ({exc})",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # 兜底:wandb 网络/认证异常等转友好消息
        exc_name = type(exc).__name__
        print(f"error: {exc_name}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
