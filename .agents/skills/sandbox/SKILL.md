---
name: sandbox
description: >-
  当用户要求创建/查看/管理/操作 Pyromind 沙箱（sandbox/沙箱/容器/云环境）时使用。
  沙箱是平台提供的一次性 CUSTOM 无头容器（Jupyter-lab 镜像，支持挂载卷、端口映射
  与文件读取），用于实验、调试与可复现演示。本 Skill 通过 sandbox_create /
  sandbox_delete / sandbox_read_file / sandbox_terminal 提供容器生命周期管理、
  文件读取与终端命令执行（sandbox_terminal 与 pyromind terminal 共用同一个
  WebSocket TTY 桥）。数据集可用性验证、逐条起镜像跑任务验题等环境依赖数据处理
  任务请用 environment-data-processing（声明式 profile + 冻结 runner），
  不要用本 skill 手工逐条编排数据验证。
triggers:
- sandbox
- 沙箱
- 容器
- 创建沙箱
- jupyter
- sandbox environment
---

# 创建与管理 Pyromind 沙箱

Pyromind Sandbox 服务提供一次性 CUSTOM 无头容器（默认 Jupyter-lab 镜像，含 SSH），
用于实验、调试与可复现演示。工具封装平台 Sandbox API（pyromind-sdk `SandboxClient`
≥ 0.1.9）。沙箱内命令通过 `sandbox_terminal` 执行，它走 `/sandboxes/{id}/terminal`
WebSocket 桥接（与 `pyromind terminal <sandbox-id> --cluster <集群>` CLI 同一通道），
不使用 exec_command API。

## 工具一览

| 工具 | 作用 |
|------|------|
| `sandbox_create` | 创建 custom 容器沙箱并等待 `running`；未指定 `image` 时用集群默认镜像 |
| `sandbox_delete` | 删除沙箱、释放资源；Running 状态会先自动 pause 再删除 |
| `sandbox_read_file` | 读取沙箱内文件（文本 utf-8；二进制 base64；大文件截断） |
| `sandbox_terminal` | 通过终端 WebSocket 在沙箱内执行一条命令（返回输出与退出码；可配 cwd/timeout） |

## 典型流程

1. **创建**：`sandbox_create(name=..., wait_timeout=600)`；`image` 不传时按集群选择
   默认镜像：
   - us-west-2 → `pyrominddynamics/jupyter-lab-with-ssh:v0.9-aws`
   - 其他集群（如 us-west-1）→ `pyrominddynamics/jupyter-lab-with-ssh:v0.9`
   - 可选：`cpu`（默认 4）、`memory`（默认 cpu×2Gi）、`volume_mounts`
     （如 `[{"host_path": "/workspace", "mount_path": "/data"}]`，之后
     `sandbox_read_file` 用容器内路径 `/data/...`）、`port_mappings`
     （如 `[{"container_port": 8080}]`）
2. **等待就绪**：`wait_timeout=0` 立即返回；否则工具会等待 `running`。若超时未
   running，以返回的状态为准再决定，不要重复创建
3. **执行命令**：`sandbox_terminal(sandbox_id=..., command="pip list",
   cwd="/workspace")`；每条命令在全新 shell 中执行（`cwd` 会被拼成
   `cd <cwd> && <command>`），返回合并输出与 `returncode`；可配
   `timeout_seconds`（默认 60、上限 600）
4. **读取文件**：`sandbox_read_file(sandbox_id=..., path="/data/x.log")`；
   二进制返回 base64，超大文件截断至约 2 万字符并在结果中标注
5. **收尾**：任务结束后 `sandbox_delete`（除非用户要求保留）

## 注意事项

- **边界**：本 skill 只提供容器原语。任务是"数据集可用性验证/逐条起镜像跑任务
  验题/环境依赖数据处理"时，改用 environment-data-processing skill（profile
  匹配走 sandbox_runner.py 冻结运行时），不要在本 skill 下手工逐条编排沙箱做
  数据验证
- 创建是异步的：`wait_timeout` 内未 `running` 时以返回状态为准，勿重复创建
- 平台拒绝删除 Running 状态的沙箱（`status is Running, can not delete!`）；
  `sandbox_delete` 遇此错误会自动先 `pause` 再重试删除，无需手工暂停
- `sandbox_terminal` 每条命令一个独立 shell（不保留 cd/环境变量）；需要持续会话
  时用 `pyromind terminal` 交互式桥接
- `sandbox_terminal` 的 `timed_out=True` 表示命令超时（输出可能不完整）；
  普通命令失败看 `returncode`
- `sandbox_read_file` 仅读文件；目录列表/写入等操作用 `sandbox_terminal`
  （如 `ls`、`cat`、`vi`）
- 资源/配额/权限类报错请如实告知用户，不要自行规避重试
