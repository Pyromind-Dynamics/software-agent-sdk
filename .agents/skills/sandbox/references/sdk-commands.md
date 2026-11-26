# 沙箱命令执行入口

当前仍列出 `sandbox_terminal` 时使用它，尤其是 `os-sandbox`：SDK 0.1.17 的
WebSocket 客户端不读取该执行环境必须使用的代理配置。不要绕开 OS 网络隔离。
只有系统消息启用 SDK 且没有列出该 Tool 时，使用下面的 SDK 流式执行方式。

```sh
: "${PYROMIND_SDK_PYTHON:?Platform SDK execution context is unavailable}"
"$PYROMIND_SDK_PYTHON" - "sandbox-id" "pwd; ls" "/workspace" <<'PY'
import sys

from pyromind_sdk import PyroMindAPIClient

returncode = None
with PyroMindAPIClient(timeout=30, max_retries=0) as client:
    for event in client.sandboxes.exec_command_stream(
        sys.argv[1], ["bash", "-lc", sys.argv[2]], cwd=sys.argv[3], timeout=60, tty=False
    ):
        if event.type in {"stdout", "stderr"}:
            data = event.data
            if data is not None:
                raw = data if isinstance(data, bytes) else data.encode("utf-8")
                stream = sys.stdout if event.type == "stdout" else sys.stderr
                stream.buffer.write(raw)
                stream.buffer.flush()
        elif event.type == "exit":
            returncode = event.returncode
if returncode is None:
    raise RuntimeError("Command completion is unconfirmed")
raise SystemExit(returncode)
PY
```

shell 按镜像实际提供的程序选择；需要 shell 操作符时显式用 `bash -lc`，
单个可执行文件可以直接用参数列表。`cwd` 是容器内路径，每次命令不保留 cd 或环境变量。
默认沿用 60 秒上限；长任务显式设定合适超时，terminal 自身超时应稍长。

非零退出码是命令失败；SDK 连接拒绝、超时或断流没有 exit 时会抛
`SandboxExecStreamError`，结果未确认。已有部分输出也不能推断成功，先核对副作用再决定
是否重试。大输出由通用 terminal 截断并提供完整输出位置，不为日志创建业务任务。
这不是交互式 stdin 会话；需要持续交互时仍使用平台已支持的终端入口。
