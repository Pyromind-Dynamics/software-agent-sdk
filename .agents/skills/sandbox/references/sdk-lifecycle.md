# SDK 创建与释放沙箱

系统消息启用 SDK 时，在 Main terminal 使用 `$PYROMIND_SDK_PYTHON`。
SDK 身份和集群由会话提供，创建沙箱不登记 Project 或工作流任务。

## 创建与等待

```sh
: "${PYROMIND_SDK_PYTHON:?Platform SDK execution context is unavailable}"
"$PYROMIND_SDK_PYTHON" - "experiment-name" <<'PY'
import os
import sys

from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.models import ResourceConfig, SandboxRequest, SandboxType

wait_timeout = 600
cpu = 4
with PyroMindAPIClient(timeout=30, max_retries=0) as client:
    created = client.sandboxes.create(
        SandboxRequest(
            name=sys.argv[1],
            sandbox_type=SandboxType.CUSTOM,
            image=os.environ["PYROMIND_SANDBOX_DEFAULT_IMAGE"],
            resources=ResourceConfig(cpu=str(cpu), memory=f"{cpu * 2}Gi"),
        )
    )
    print(created.model_dump_json(exclude_none=True), flush=True)
    if wait_timeout > 0:
        ready = client.sandboxes.wait_for_sandbox_status(
            created.id, target_status="running", timeout=wait_timeout
        )
        latest = client.sandboxes.get_sandbox(created.id)
        print(latest.model_dump_json(exclude_none=True))
        if not ready:
            raise RuntimeError(f"Sandbox {created.id} not ready: {latest.status}")
PY
```

默认镜像已按部署配置、集群默认值解析。用户指定镜像时替换 `image`；CPU 默认 4，
未指定内存时使用 CPU×2Gi。需要挂载/端口时给 `SandboxRequest` 显式传
`volume_mounts=[VolumeMount(...)]`、`port_mappings=[PortMapping(...)]`，类型来自
`pyromind_sdk.client.models`。容器内路径用挂载的 `mount_path`。

默认等待 600 秒，terminal 的超时需留出余量（例如 660 秒）；不等待时设
`wait_timeout=0`。超时或失败后保留已打印的 ID，查询该实例，不重复创建。
SDK 创建请求没有已确认的幂等保证；响应丢失时先核对平台实例列表。

## 暂停与删除

```sh
: "${PYROMIND_SDK_PYTHON:?Platform SDK execution context is unavailable}"
"$PYROMIND_SDK_PYTHON" - "sandbox-id" <<'PY'
import sys
import time

from pyromind_sdk import PyroMindAPIClient

sandbox_id = sys.argv[1]
with PyroMindAPIClient(timeout=30, max_retries=0) as client:
    current = client.sandboxes.get_sandbox(sandbox_id)
    if current.status.lower() == "running":
        client.sandboxes.pause(sandbox_id)
        deadline = time.monotonic() + 60
        while current.status.lower() not in {"paused", "stopped"}:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Sandbox {sandbox_id} has not stopped")
            time.sleep(1)
            current = client.sandboxes.get_sandbox(sandbox_id)
    client.sandboxes.delete(sandbox_id)
print(f"Sandbox {sandbox_id} deleted")
PY
```

平台拒绝删除运行实例时，先 pause 并确认状态，再删除；失败不报告资源已释放。
任务结束后释放自己创建的实例，用户要求保留时记录 ID 和用途。
SDK 上下文不可用时报告具体错误；未启用 SDK 或已回退的会话使用现有
`sandbox_create` / `sandbox_delete`。
