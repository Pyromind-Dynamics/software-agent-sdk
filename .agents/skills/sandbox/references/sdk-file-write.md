# SDK 写入沙箱文件

系统消息启用 SDK 时，通过 Main terminal 的 `$PYROMIND_SDK_PYTHON` 执行。
参数是实际沙箱 ID、容器内绝对路径；下面直接写 UTF-8 正文：

```sh
: "${PYROMIND_SDK_PYTHON:?Platform SDK execution context is unavailable}"
"$PYROMIND_SDK_PYTHON" - "sandbox-id" "/tmp/example.txt" <<'PY'
import sys

from pyromind_sdk import PyroMindAPIClient

payload = "hello\n".encode("utf-8")
with PyroMindAPIClient(timeout=60, max_retries=0) as client:
    receipt = client.sandboxes.write_file(sys.argv[1], sys.argv[2], payload)
    actual = client.sandboxes.read_file(sys.argv[1], sys.argv[2])
    if receipt["size"] != len(payload) or actual != payload:
        raise RuntimeError("Sandbox file verification failed")
print(f"Verified {len(payload)} bytes")
PY
```

`source` 传字符串或 `Path` 时代表调用端本地文件路径，不能把正文字符串直接传入。
正文用 bytes；空文件传 `b""`。大文件可以传 `Path("public_data/file.bin")`，
SDK 自动分片；核对返回大小，必要时读回比较内容或摘要，不把“上传请求成功”当作核验。
容器内路径与当前 terminal 本地路径不同。目标父目录不存在且平台返回路径错误时，
先按 [命令参考](sdk-commands.md) 创建该目录，再写入已确认的目标路径。

不输出文件中的凭据，不在脚本中手填平台认证。超时或分片失败先核对已写内容，
不直接重放整段业务命令。未启用 SDK 或回退的会话继续使用 `sandbox_write_file`。
