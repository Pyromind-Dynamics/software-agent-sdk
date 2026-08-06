"""wandb 训练数据分析 CLI — Pyromind skill 辅助工具。

子命令:
- resolve-target <task_id>: 通过平台 API 定位 wandb run(entity/project/run_id)并提取凭证
- probe <entity/project> [--run-id]: 探查 project/run 的指标与 config 键
- analyze-run <entity/project> <run_id>: 单 run 稳定性诊断
- compare-runs <entity/project> <run_a> <run_b>: 两 run config 与指标对比
- project-summary <entity/project>: 按 config 轴分桶统计
- report <entity/project> <run_id> --out <md>: 四阶段分析报告

平台 API 认证与 validate_workflow 工具对齐: cookie / x-cluster /
authorization 三个 header 小写透传,分别从环境变量 PYROMIND_COOKIE /
X_CLUSTER / PYROMIND_AUTHORIZATION 读取,或通过 --cookie / --cluster /
--authorization 参数传入;base URL 按 APP_ENV(prod/production/online 走
正式环境,否则 pre 环境)推断,可用 --api-base 或 PYROMIND_API_BASE 覆盖。
wandb 凭证优先从 --api-key 或 --creds-file(resolve-target 的 --creds-out
产物)读取,否则回退环境变量 WANDB_API_KEY。
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wandb_helpers import (
    compare_configs,
    diagnose_run,
    probe_project,
    scan_history,
)


PRE_API_BASE = "https://pre-api-portal.pyromind.ai/std2/studio_api/"
PROD_API_BASE = "https://api-portal.pyromind.ai/std2/studio_api/"
_PROD_APP_ENVS = {"prod", "production", "online"}


def _default_api_base() -> str:
    """按 APP_ENV 推断平台 API base,与 validate_workflow 的默认端点选择一致。"""
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env in _PROD_APP_ENVS:
        return PROD_API_BASE
    return PRE_API_BASE


RUN_SETUP_RE = re.compile(r"wandb:\s+setting up run\s+(\w+)")
RUN_URL_RE = re.compile(
    r"View run at\s+https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/(\w+)"
)
CONFIG_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")

# Cloudflare 会拦截 urllib 默认 UA(browser_signature_banned),需浏览器风格 UA
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# 训练节点类型关键词,用于从任务节点列表中识别 wandb 输出源
TRAIN_NODE_KEYWORDS = ("Train", "train")
WANDB_NODE_TYPE = "WandbConfigBuilderNode"


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


def _parse_config_text(text: str) -> dict[str, str]:
    """解析 wandb_config YAML 文本(仅 key: value 行,嵌套不做展开)。"""
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = CONFIG_LINE_RE.match(line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def _collect_wandb_env(output: dict[str, Any]) -> dict[str, str]:
    """从节点输出/config 提取 WANDB_* 环境变量,覆盖三种真实形态:

    - wandb_config 文本块(训练节点 config / node_output 输出)
    - wandb_config 为 dict
    - 结构化字段 wandb_api_key / wandb_project / wandb_name(Builder 节点 config)
    - entries[].m 文本数组(node_output 接口响应)
    """
    result: dict[str, str] = {}

    def merge_text(text: str) -> None:
        for key, value in _parse_config_text(text).items():
            if value:
                result.setdefault(key, value)

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
    return result


def _collect_wandb_env_from_node(node: dict[str, Any]) -> dict[str, str]:
    """从 task_workflow_result 节点的 data.config 提取 wandb 凭证。"""
    data = node.get("data")
    if not isinstance(data, dict):
        return {}
    config = data.get("config")
    if not isinstance(config, dict):
        return {}
    return _collect_wandb_env(config)


def _extract_run_info(log_text: str) -> tuple[str, str, str] | None:
    """从日志中提取 (entity, project, run_id);URL 行优先,其次 setting up run。"""
    for match in RUN_URL_RE.finditer(log_text):
        return match.group(1), match.group(2), match.group(3)
    match = RUN_SETUP_RE.search(log_text)
    if match:
        return "", "", match.group(1)
    return None


def _run_id_from_path(text: str) -> str | None:
    """从 run-<ts>-<run_id> 本地路径行提取 run id(兜底)。"""
    match = re.search(r"run-\d{8}_\d{6}-(\w+)", text)
    return match.group(1) if match else None


def _credential_file() -> Path:
    return Path(tempfile.gettempdir()) / "wandb_analysis_creds.json"


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


def _resolve_api_key(args: argparse.Namespace) -> str:
    creds = _load_creds(getattr(args, "creds_file", None))
    api_key = (
        getattr(args, "api_key", None)
        or creds.get("WANDB_API_KEY")
        or os.environ.get("WANDB_API_KEY", "")
    )
    if not api_key:
        raise WandbAnalysisError(
            "WANDB_API_KEY missing: 先运行 resolve-target --creds-out,"
            "或传入 --api-key / 设置环境变量 WANDB_API_KEY"
        )
    return api_key


def _resolve_entity(
    api_key: str, project: str, fallback_entity: str = ""
) -> tuple[str, str]:
    """通过 wandb API 确认 entity(优先 api.viewer(),其次显式 fallback)。"""
    entity = fallback_entity or os.environ.get("WANDB_ENTITY", "")
    if entity and project:
        return entity, project
    from wandb import Api  # 延迟导入,仅需凭证解析时避免强制依赖

    viewer = Api(api_key=api_key, timeout=120).viewer()
    username = getattr(viewer, "username", "") or ""
    if not username:
        raise WandbAnalysisError("无法从 wandb API 推断 entity,请用 --entity 显式指定")
    return username, project


def cmd_resolve_target(args: argparse.Namespace) -> int:
    """定位 wandb run:平台节点列表 → 训练节点日志(run id)+ Builder 输出(凭证)。"""
    client = PlatformClient(
        base_url=args.api_base,
        cookie=args.cookie,
        cluster=args.cluster,
        authorization=args.authorization,
    )
    payload = client.task_workflow_result(args.task_id)
    nodes = _find_nodes(payload)
    if not nodes:
        raise WandbAnalysisError(
            "task_workflow_result 响应中未找到节点列表,响应结构待校准:"
            f"{json.dumps(payload, ensure_ascii=False)[:800]}"
        )

    run_id = ""
    run_entity = ""
    run_project = ""
    wandb_env: dict[str, str] = {}

    for node in nodes:
        node_type = _node_type(node)
        node_id = _node_identifier(node)
        if not node_id:
            continue
        # 凭证优先从节点 config 直接提取(task_workflow_result 已含全部 config)
        if not wandb_env:
            wandb_env = _collect_wandb_env_from_node(node)
        if _is_train_node(node) and not run_id:
            log_text = client.node_log(node_id, args.task_id)
            info = _extract_run_info(log_text)
            if info and info[2]:
                run_entity, run_project, run_id = info
            elif not run_id:
                run_id = _run_id_from_path(log_text) or ""

    # 回退:config 未含凭证时查 WandbConfigBuilderNode 输出接口
    if not wandb_env:
        for node in nodes:
            node_type = _node_type(node)
            node_id = _node_identifier(node)
            if WANDB_NODE_TYPE in node_type and node_id:
                wandb_env = _collect_wandb_env(
                    client.node_output(node_id, args.task_id)
                )
                break

    if not run_id and not wandb_env:
        raise WandbAnalysisError(
            "未能从平台 API 定位 run:训练节点日志无 wandb run 记录,"
            "且未找到 WandbConfigBuilderNode 输出。请提供 --run-url 或检查 task_id。"
        )

    api_key = wandb_env.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY", "")
    project = (
        wandb_env.get("WANDB_PROJECT")
        or run_project
        or (args.run_url_project if args.run_url else "")
    )
    entity = wandb_env.get("WANDB_ENTITY") or run_entity or args.entity
    if api_key and project and (entity or True):
        try:
            entity, project = _resolve_entity(api_key, project, fallback_entity=entity)
        except Exception:
            pass  # 平台网络受限时保留原始值,由后续命令报错

    result: dict[str, Any] = {
        "task_id": args.task_id,
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "wandb_env": {k: v for k, v in wandb_env.items() if k != "WANDB_API_KEY"},
    }
    if args.creds_out:
        creds_path = Path(args.creds_out)
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        creds_path.write_text(
            json.dumps(
                {
                    "WANDB_API_KEY": api_key,
                    "api_key": api_key,
                    **wandb_env,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.chmod(creds_path, 0o600)
        # 创建 wandb_tmp 工作目录,避免后续命令因缺少临时目录失败
        wandb_tmp = creds_path.parent / "wandb_tmp"
        wandb_tmp.mkdir(parents=True, exist_ok=True)
        result["creds_file"] = str(creds_path)
        result["wandb_tmp"] = str(wandb_tmp)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _make_api(args: argparse.Namespace):
    os.environ.setdefault("WANDB_SILENT", "true")
    from wandb import Api

    return Api(api_key=_resolve_api_key(args), timeout=120)


def _run_keys(api: Any, entity: str, project: str, run_id: str) -> list[str]:
    run = api.run(f"{entity}/{project}/{run_id}")
    return sorted({str(k) for k in (run.summary or {}).keys()})


def cmd_probe(args: argparse.Namespace) -> int:
    api = _make_api(args)
    result = probe_project(api, args.entity_project)
    if args.run_id:
        run = api.run(f"{args.entity_project}/{args.run_id}")
        result["run_summary_keys"] = sorted((run.summary or {}).keys())
        result["run_config_keys"] = sorted((run.config or {}).keys())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


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


def _loss_metric_key(run: Any) -> str:
    """从 summary 选默认 loss 键;SummarySubDict 迭代会 KeyError,需先转 dict。"""
    summary = dict(run.summary or {})
    return next((k for k in summary if "loss" in k), "")


def _json_safe(value: Any) -> Any:
    """递归转原生 JSON 类型(wandb SummarySubDict/Config 等 dict 子类不可直接序列化)。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def cmd_analyze_run(args: argparse.Namespace) -> int:
    api = _make_api(args)
    run = api.run(f"{args.entity_project}/{args.run_id}")
    metric = args.metric or _loss_metric_key(run)

    # 多指标支持:--keys 指定逗号分隔的指标键列表
    history_keys = [k.strip() for k in args.keys.split(",")] if args.keys else []
    if not history_keys:
        history_keys = [metric] if metric else []

    report: dict[str, Any] = {
        "run": args.run_id,
        "display_name": run.display_name,
        "state": run.state,
        "config": _json_safe(dict(run.config or {})),
        "summary": _json_safe(dict(run.summary or {})),
        "metric": metric or None,
    }
    if history_keys:
        history = scan_history(run, keys=history_keys)
        if history:
            # 主指标统计
            if metric in history_keys:
                vals = _history_float_values(history, metric)
                report["metric_stats"] = _history_stats(vals)
                report["diagnostics"] = diagnose_run(run, train_key=metric)
            # 多指标原始数据
            if len(history_keys) > 1:
                multi: dict[str, list[float]] = {}
                for k in history_keys:
                    vals = _history_float_values(history, k)
                    if vals:
                        multi[k] = vals
                if multi:
                    report["multi_metrics"] = multi
            first_step = history[0].get("_step")
            last_step = history[-1].get("_step")
            if first_step is not None:
                report["steps"] = [first_step, last_step]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_compare_runs(args: argparse.Namespace) -> int:
    api = _make_api(args)
    run_a = api.run(f"{args.entity_project}/{args.run_a}")
    run_b = api.run(f"{args.entity_project}/{args.run_b}")
    config_diff = compare_configs(run_a, run_b)
    metric_a = _loss_metric_key(run_a)
    metric_b = _loss_metric_key(run_b)
    summary_a = dict(run_a.summary or {})
    summary_b = dict(run_b.summary or {})
    summary_keys = sorted(set(summary_a) | set(summary_b))
    summary_diff = {
        key: {"a": summary_a.get(key), "b": summary_b.get(key)}
        for key in summary_keys
        if summary_a.get(key) != summary_b.get(key)
    }
    config_diff_map = {
        d["key"]: {run_a.name: d[run_a.name], run_b.name: d[run_b.name]}
        for d in config_diff
    }
    print(
        json.dumps(
            {
                "config_diff": _json_safe(config_diff_map),
                "summary_diff": _json_safe(summary_diff),
                "run_a_metric": metric_a,
                "run_b_metric": metric_b,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_project_summary(args: argparse.Namespace) -> int:
    api = _make_api(args)
    runs = list(api.runs(f"{args.entity_project}", per_page=200))
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "losses": []})
    for run in runs:
        config = dict(run.config or {})
        axis = str(config.get(args.axis, "default"))
        bucket = buckets[axis]
        bucket["count"] += 1
        loss = next((v for k, v in (run.summary or {}).items() if "loss" in k), None)
        if loss is not None:
            bucket["losses"].append(float(loss))
    for bucket in buckets.values():
        losses = bucket.pop("losses")
        bucket["mean_loss"] = sum(losses) / len(losses) if losses else None
        bucket["min_loss"] = min(losses) if losses else None
    print(
        json.dumps(
            {
                "total_runs": len(runs),
                "axis": args.axis,
                "buckets": dict(sorted(buckets.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


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


def cmd_report(args: argparse.Namespace) -> int:
    api = _make_api(args)
    run = api.run(f"{args.entity_project}/{args.run_id}")
    metric = args.metric or _loss_metric_key(run)
    diagnostics: dict[str, Any] = {}
    if metric:
        diagnostics = diagnose_run(run, train_key=metric) or {}
    config = dict(run.config or {})
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
        f"# wandb 训练分析报告:{args.entity_project}/{args.run_id}",
        "",
        "## 1. 先验假设",
        "- 训练类型:由 config 推断(SFT/DPO/GRPO,见 config 摘要)",
        f"- 关注指标: {metric or '未知'}",
        "- 基线预期: loss 单调下降或收敛至平台期",
        "",
        "## 2. 数据惊奇",
        f"- run state: {run.state}, display_name: {run.display_name}",
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
            "  - 数据集质量检查:重复样本/标签噪声可能导致过拟合, 建议检查并清洗训练数据"
        )
    if diagnostics.get("converged") is False:
        lines.append("- 未收敛 → 训练不足,考虑延长 epoch 或调整学习率")
    if not diagnostics.get("has_nan") and not diagnostics.get("likely_overfit"):
        lines.append("- 未发现明显异常,可能需要与基线/相邻 run 对比(compare-runs)")
        lines.append(
            "  - 数据集质量基线检查:若后续调优效果不达预期,"
            " 建议检查训练数据集质量(重复样本/标签噪声/分布)"
        )
    lines += [
        "",
        "## 4. 可证伪探针实验",
    ]
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
        f"- entity/project: `{args.entity_project}`",
        f"- run id: `{args.run_id}`",
        f"- 指标键: `{metric or 'N/A'}`",
        "- 数据来源: wandb SDK(wandb.Api),凭证来自平台 WandbConfigBuilderNode 输出",
    ]
    output = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(output)
    return 0


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

    p_compare = subparsers.add_parser("compare-runs", help="两 run 对比")
    p_compare.add_argument("entity_project")
    p_compare.add_argument("run_a")
    p_compare.add_argument("run_b")
    p_compare.set_defaults(func=cmd_compare_runs)

    p_summary = subparsers.add_parser("project-summary", help="project 分桶统计")
    p_summary.add_argument("entity_project")
    p_summary.add_argument("--axis", default="job_type")
    p_summary.set_defaults(func=cmd_project_summary)

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
