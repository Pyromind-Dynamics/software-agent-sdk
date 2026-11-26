# 通过 SDK 删除沙箱文件

系统消息明确启用平台 SDK 上下文时，使用当前 Main terminal。该 Workbench 主会话
已停止注入 `sandbox_delete_file`；其他上下文仍使用其已启用入口。

把参数替换为已确认要删除的沙箱 ID 和容器内绝对路径：

```sh
: "${PYROMIND_SDK_PYTHON:?Platform SDK execution context is unavailable}"
"$PYROMIND_SDK_PYTHON" - "sandbox-id" "/absolute/container/file" <<'PY'
import sys

from pyromind_sdk import PyroMindAPIClient

with PyroMindAPIClient(timeout=30, max_retries=0) as client:
    result = client.sandboxes.delete_file(sys.argv[1], sys.argv[2], recursive=False)
print(result)
PY
```

普通文件默认 `recursive=False`。确认删除整个目录时才改为 `recursive=True`；
不为解决权限或路径错误扩大删除范围。容器路径不等于当前 terminal 的本地路径。
SDK 抛异常表示删除未确认，不打印成功或盲目重试；超时后先核对目标是否仍存在。

身份、集群和 API 地址由当前会话注入。解释器或 SDK 上下文不可用时，若仍列出
`sandbox_delete_file` 就使用该入口，否则报告执行错误，不寻找其他账户凭据。
此操作直接调用 SDK，不创建 Project 或业务运行记录。
