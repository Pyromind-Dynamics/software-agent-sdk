# SDK 停止平台任务

系统消息启用 SDK 时，在 Main terminal 直接使用 `client.studio.stop(task_id)`。
未启用 SDK 或已回退的会话继续使用 `df_stop_task`。执行前核对用户要停止的具体任务。

```sh
: "${PYROMIND_SDK_PYTHON:?Platform SDK execution context is unavailable}"
"$PYROMIND_SDK_PYTHON" - "task_id" "actual-task-id" <<'PY'
import json
import os
import sys
from pathlib import PurePosixPath

from pyromind_sdk import PyroMindAPIClient, PyroMindAPIError
from requests import RequestException

kind, target = sys.argv[1:3]
task_ids = [target]
if kind in {"run_id", "output_dir"}:
    associations = json.loads(os.environ["PYROMIND_TASK_ASSOCIATIONS"])
    if kind == "output_dir":
        def normalize(path, *, user_path=False):
            parsed = PurePosixPath(path.strip())
            if ".." in parsed.parts:
                raise ValueError("Parent path components are not allowed")
            parts = parsed.parts[1:] if parsed.is_absolute() else parsed.parts
            if user_path and parts and parts[0] == "workspace":
                parts = parts[1:]
            return "/" + "/".join(parts)
        target = normalize(target, user_path=True)
        matches = [
            item for item in associations
            if isinstance(item.get("output_dir"), str)
            and item["output_dir"].strip()
            and normalize(item["output_dir"]) == target
        ]
        runs = {item.get("run_id") or item["task_id"] for item in matches}
        if len(runs) > 1:
            raise ValueError("Output directory matches multiple executions; use run_id")
    else:
        matches = [item for item in associations if item.get("run_id") == target]
    if not matches:
        raise ValueError("No task association in the current conversation")
    task_ids = list(dict.fromkeys(item["task_id"] for item in matches))
elif kind != "task_id":
    raise ValueError("Expected task_id, run_id or output_dir")

results = []
accepted = terminated = 0
terminal_states = {"terminated", "stopped", "cancelled", "canceled", "succeeded", "failed", "error"}
with PyroMindAPIClient(timeout=30, max_retries=0) as client:
    for task_id in task_ids:
        try:
            result = client.studio.stop(task_id)
            confirmed = result.status.strip().lower() in terminal_states
            accepted += 1
            terminated += int(confirmed)
            item = {"task_id": task_id, "stop_request": "accepted", "status": result.status,
                    "terminal_confirmed": confirmed}
        except (PyroMindAPIError, RequestException, TypeError, ValueError) as error:
            item = {"task_id": task_id, "stop_request": "failed", "error": str(error)[:300]}
        if len(results) < 20:
            results.append(item)
print(json.dumps({"requested": len(task_ids), "accepted": accepted,
                  "terminal_confirmed": terminated, "results": results,
                  "omitted_results": max(0, len(task_ids) - len(results))}))
raise SystemExit(0 if accepted == len(task_ids) else 1)
PY
```

已有实际平台 ID 时用 `task_id`；只有业务 run 或输出目录时，把第一个参数改为
`run_id` 或 `output_dir`。映射是当前会话已有运行记录的只读快照，包括兼容的历史关联，
优先当前 Product run；同一执行可包含多个分片的真实平台 ID，按 `run_id` 停止时逐个处理。
没有输出目录的记录会跳过；输出目录对应多个不同执行时先报告歧义，改用明确的 `run_id`。
映射不创建任务，也不决定平台 API 的调用入口。返回总数和最多 20 条结果，部分失败保留已处理结果。

请求受理后任务可能仍是 Running/Stopping，不能报告“已停止”。用同一 SDK 的
`client.studio.get_task(task_id)` 或平台回调确认终态，再调整和重提。
SDK 异常、超时或返回状态不明时先查状态，不盲目重试停止请求。
业务记录、输出目录与 Project 阶段仍由对应场景维护，不删除 Runtime 恢复信息。
