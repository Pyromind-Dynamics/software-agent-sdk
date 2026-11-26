---
name: sandbox
description: >-
  当用户要求创建/查看/管理/操作 Pyromind 沙箱（sandbox/沙箱/容器/云环境）时使用。
  沙箱是平台提供的一次性 CUSTOM 无头容器（Jupyter-lab 镜像，支持挂载卷、端口映射
  与文件读取），用于实验、调试与可复现演示。本 Skill 按当前会话使用 SDK 或
  sandbox 工具提供容器生命周期管理、文件操作与终端命令执行
  （sandbox_terminal 与 pyromind terminal 共用同一个
  WebSocket TTY 桥）。数据集可用性验证、逐条起镜像跑任务验题等编排式数据处理
  任务请用 data-processing skill 的 environment-processing 处理范式（声明式
  profile + 冻结 runner），不要用本 skill 手工逐条编排数据验证。
---

# 创建与管理 Pyromind 沙箱

本 Skill 处理单个 CUSTOM 容器的实验、调试与演示。数据集逐条验题、环境依赖验证或
批量编排使用 data-processing 的 environment-processing 范式，不手工逐条操作容器。

## SDK 执行入口

系统消息明确启用 SDK 时，使用 Main terminal 的 `$PYROMIND_SDK_PYTHON`。
当前会话提供身份、API 地址、集群和默认镜像；不在脚本中写凭据。
按当前操作读取一份参考，不一次加载所有 API：

| 操作 | SDK / 当前入口 | 按需参考 |
|---|---|---|
| 创建、等待、暂停、删除 | `client.sandboxes.create/get_sandbox/pause/delete` | [生命周期](references/sdk-lifecycle.md) |
| 读取文本或二进制文件 | `client.sandboxes.read_file`，UTF-8 或 base64，有界输出 | [读取](references/sdk-file-read.md) |
| 写入正文、本地文件或空文件 | `client.sandboxes.write_file`，写后核验 | [写入](references/sdk-file-write.md) |
| 删除文件或目录 | `client.sandboxes.delete_file`，目录显式递归 | [删除](references/sdk-file-delete.md) |
| 执行命令、列目录、查询进程 | 仍列出 `sandbox_terminal` 时用该 Tool，否则 SDK stream | [命令](references/sdk-commands.md) |

已启用 SDK 的标准单 Main 按原授权清单移除上述原子 Tool；OS 模式仍保留
`sandbox_terminal`，因为 SDK 0.1.17 的 WebSocket 代理能力不满足 OS 隔离要求。
Tool 实现与注册仍保留。SDK 环境不可用时报告具体错误，不绕开隔离或换账户。

创建保持 CUSTOM、CPU 默认 4、内存默认 CPU×2Gi。用户指定镜像优先，否则用
`PYROMIND_SANDBOX_DEFAULT_IMAGE` 的部署/集群有效默认值。挂载后使用容器内
`mount_path`，它与当前 terminal 的本地路径不同。创建后保留返回 ID，默认等
600 秒；未 running 不重复创建。运行实例先 pause 并确认状态，再删除。

命令显式设置 cwd 和超时，默认沿用 60 秒；非零退出码是失败，无 exit 的断流或超时
是结果未确认。长任务保存 PID，用后续命令查询状态和日志，不重复启动。
任务成功或失败都释放自己创建的容器，除非用户要求保留；失败时报告未释放的 ID。

## Storage 文件直接落位

SDK 模式的远端 Main 中，用户 Storage 已挂载为 `/workspace`，使用 `read` / `write` / `edit` 或
terminal 直接读取、写入和复制真实文件；复制后核对大小及 SHA-256。远端不再注入
`upload_file_to_pyromind`、`sandbox_upload`、`sandbox_download`，不需要经过中转上传。
进度和报告也从已知运行目录读取，按需查看
[挂载目录进度读取](../data-processing/references/storage-progress.md)。

SDK 创建的其他沙箱要单独确认其挂载；只有同一 Storage 的相同文件在目标挂载下可读，
才能直接使用目标容器内路径。未共享挂载时，用 SDK `read_file` / `write_file` 在实际
执行环境与指定沙箱之间传递所需文件，分别按上表读取对应参考，并核对目标内容。
不要把 Main 的工作区路径当成其他沙箱或平台节点已经可读的路径。

## 未启用 SDK 或已回退的会话

只使用实际列出的工具。受限权限、自定义角色、未登录和子 Agent 保持原权限范围，
不自行获取或更换 SDK 身份。

| 工具 | 用法 |
|---|---|
| `sandbox_create` | `name`、可选 `image/cpu/memory/volume_mounts/port_mappings`；默认等待 600 秒，0 为不等待 |
| `sandbox_delete` | 指定 `sandbox_id`；运行实例会先 pause 再删除 |
| `sandbox_read_file` | 指定 `sandbox_id/path`，文本或 base64，约 2 万字符展示上限 |
| `sandbox_write_file` | 指定 `sandbox_id/path/content`，正文按 UTF-8 写入 |
| `sandbox_delete_file` | 指定 `sandbox_id/path`，非空目录需要 `recursive=True` |
| `sandbox_terminal` | 指定 `sandbox_id/command`，可选 `cwd/timeout_seconds`；每次独立 shell |

原终端 Tool 使用平台 TTY WebSocket；`timed_out=True` 表示超时，普通失败看
`returncode`。目录、挂载与资源配置使用平台支持的值；配额、权限和镜像错误如实报告。
