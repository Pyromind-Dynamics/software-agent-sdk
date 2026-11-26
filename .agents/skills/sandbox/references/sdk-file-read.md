# 通过 SDK 读取沙箱文件

仅在系统消息明确启用平台 SDK 上下文时使用。该 Workbench 主会话已停止注入旧读工具，
保留实现用于回退。
其他会话继续使用其已启用的工具，不自行查找或复制服务端凭据。

使用 terminal 和已配置的 SDK 解释器。将两个参数替换为实际沙箱 ID 和容器内绝对路径：

```sh
: "${PYROMIND_SDK_PYTHON:?Platform SDK execution context is unavailable}"
"$PYROMIND_SDK_PYTHON" - "sandbox-id" "/absolute/container/path" <<'PY'
import base64
import sys

from pyromind_sdk import PyroMindAPIClient

with PyroMindAPIClient(timeout=30, max_retries=0) as client:
    data = client.sandboxes.read_file(sys.argv[1], sys.argv[2])

try:
    text = data.decode("utf-8")
except UnicodeDecodeError:
    text = base64.b64encode(data).decode("ascii")
    print("encoding: base64")
else:
    print("encoding: utf-8")

print(text[:20000])
if len(text) > 20000:
    print(f"[truncated; file size: {len(data)} bytes]")
PY
```

SDK 从执行环境取得当前用户的 API Key、目标集群和 API 地址，不在脚本中重复设置，
不打印这些值。解释器缺失或上下文不可用时，若当前仍列出 `sandbox_read_file`，使用该
兼容入口并报告 SDK 尚未就绪；否则报告执行错误，不改用其他账户的环境。

`read_file` 返回 bytes；文本按 UTF-8 读取，二进制使用 base64，并限制展示长度。
容器路径与当前会话的本地路径不同，不为读一个文件重复创建沙箱或 Project。
读取操作直接调用 SDK，不创建业务执行记录，也不经过产品工作流提交 API。
