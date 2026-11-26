# 从 Storage 挂载读取进度与产物

远端 Main 的 Storage 已挂载为 `/workspace`。SDK 模式不注入 `df_check_progress`、
`upload_file_to_pyromind`、`sandbox_upload`、`sandbox_download`；普通文件直接读写挂载目录，
进度使用下面的普通 Python 函数。旧实现仍保留，本机没有挂载或关闭 SDK 时，以实际清单为准。

## 按已知运行目录读取一次

使用提交时保存的真实 `output_dir`，不扫描 Storage 查找运行。执行解释器是
`"$PYROMIND_SDK_PYTHON"`，其中已安装 `pyromind-platform-helpers`：

```python
import sys

from pyromind_common.platform.progress import read_progress

snapshot = read_progress(sys.argv[1])
print(snapshot.model_dump_json())
```

函数读取 `progress.json`、`report.json` 和 `processed.jsonl` 尾部，返回计数、百分比、
速率、ETA、最近记录以及读取问题。`progress_found` / `report_found` 表示相应文件是否
成功读取；`errors` 说明读取、超限或格式问题。进度未写出时如实说明，不能据此判断任务
失败或完成。平台任务状态仍使用[SDK 任务检查](../../workflow-execution/references/sdk-inspection.md)。

默认最多读取 1 MiB 的单个 JSON，JSONL 仅读取末尾 64 KiB、展示最后 5 条完整记录。
`tail_lines` 可设 0–50；`tail_truncated` 表示发生字节窗口、记录数或无效记录截断，不把
半行或尚未写完的记录当成完整结果。超大文件先查看大小，再按有界窗口读取，不整份输出。

EDP 的 verdicts 在分片输出的 `run` 子目录，使用实际路径和文件名：

```python
snapshot = read_progress(
    sys.argv[1],
    tail_filename="verdicts.jsonl",
    tail_lines=5,
)
print(snapshot.model_dump_json())
```

此处参数应为 `<分片 output_dir>/run`，不是 manifest 父目录。`tail_filename` 只接受
文件名，不通过它跳到其他目录。按用户请求或分片决策需要读取一次；提交后仍结束当前轮次
等待终态回调，不用 sleep 或循环读取文件等待任务完成。

## 报告链接与人工复核

`artifact_paths` 只列出该运行目录内实际存在的文件路径。将其中需要交付的 HTML 或其他
文件交给保留的 `get_storage_url`，原样使用其返回 URL；挂载路径不是浏览器链接。

已有报告入口只有在产品记录已登记、调用方已确认该链接与当前授权运行匹配时，才可通过
`stable_report_url` 传给 `read_progress`，并从 `evaluation_report_url` 交付。函数不会创建
报告登记或验证浏览器凭据，也不会根据 UUID、目录名或 HTML 存在就拼接一个稳定 URL。

已登记的 evaluation/review 报告入口仍是首选，可保留其已有的页内复核保存能力。
没有已确认入口时，使用 `get_storage_url` 返回的真实 HTML 预览，并明确这是只读交付，
不承诺标注自动保存。用户需要人工标注时使用保留的 `label_studio_project`；不能用预览
链接冒充可编辑复核入口。链接尚不可用时，交付真实 Storage 位置和指标汇总。

## 文件直接落位

在 Main 中将 `public_data/...` 文件写入或复制到已确认的 `/workspace/...` 目录，
复制后核对大小和 SHA-256，再把实际 Storage 路径交给平台任务。失败则停止，不猜目标路径。
SDK 创建的其他沙箱需另行核实挂载；未共享挂载时按 sandbox Skill 用 SDK 文件读写，
不能把 Main 的路径当成目标容器已能读取的文件。本机无 Storage 挂载时也不能用本机 `cp`
冒充发布，使用当前仍列出的实际传输入口。
