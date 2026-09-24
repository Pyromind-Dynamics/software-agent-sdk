# Pi 会话级沙箱执行方案（v2，评审稿）

## 背景与问题

复杂数据任务里，Pi 需要反复探查 Storage、preview dataset、写脚本、跑脚本。
当前 Pi 的执行面在 agent-server 本地：`read/write/edit/terminal` 走
`WorkspaceAccessPolicy` + `os-sandbox`，业务工具（`validate_workflow_dsl`、
`workflow_debug`、`run_workflow` 等）通过 `LocalWorkspace` 读本地文件。

结果是链路很长：每一步探查都要经过 agent 工具调用与 Storage HTTP API 往返，
本地又拿不到 Storage 的数据，只能靠业务工具间接读。

平台 Sandbox 已经把用户 Storage 读写挂载进容器，容器内可以直接 `ls`/`python`
/`du` 等原生命令操作数据。本方案把 Pi 的**执行面**整体切到会话级 Sandbox，
控制面（事件、权限、业务工具编排、持久化、快照恢复）留在 agent-server。

## 目标与非目标

目标：同一会话内的文件读写与命令执行都发生在沙箱里，模型可以直接用
`read`/`write`/`edit`/`terminal` 完成"探查—写脚本—跑脚本—看结果"，
业务工具也读取同一份工作区。

非目标：不引入 SSH 作为主通道；不使用 `exec_command`（一次性调用，超时即
失联）；不改变对外协议、事件模型与前端语义；`openhands-sdk` 保持平台无关。

## 已确认的决策

| # | 决策 |
|---|---|
| 1 | Pi 控制面留在 agent-server，只有执行面切到 Sandbox |
| 2 | 用户 Storage 挂载（create 时声明 host `/workspace` → 容器 `/target-workspace`） |
| 3 | 执行工作区固定为 `/target-workspace/.pyromind-agent/<conversation_id>` |
| 4 | 写入限制是软约束，不作为安全边界 |
| 5 | 不用 `exec_command`；命令走 Sandbox 终端 WebSocket |
| 6 | 沙箱按 `conversation_id` 隔离；资源按会话配置 |
| 7 | 终端 idle 超时 30 分钟，由"有输出"重置；命令自带心跳 |
| 8 | 回收策略：会话删除删容器、空闲 pause、TTL 兜底 |
| 9 | 会话删除后 Storage 侧数据保留，不做清理 |
| 10 | 业务工具与 Pi 文件工具共用同一份沙箱工作区（远程化） |

## 架构

```text
agent-server（Pi 控制面）
  ├─ <conversation_dir>/pi/*        控制面状态（session/inflight/business-state）
  └─ PiAdapter ── stdin JSONL ──▶ Node Pi runner
                                   ├─ Pi 会话 / 模型 / 权限 / 业务工具 RPC
                                   └─ read/write/edit/terminal → Sandbox
Sandbox（CUSTOM，create 时挂载用户 Storage rw）
  └─ /target-workspace/.pyromind-agent/<conversation_id>   执行工作区
```

两个工作区必须显式区分：**控制工作区**（本地，只放控制面状态）与**执行工作区**
（沙箱内，模型与业务工具读写文件的地方）。当前 `workspace_root` 同时承担两种
语义，是本方案最主要的解耦点。

## 一、路径与资源

### 路径

- 挂载契约沿用平台既有约定：节点 `/workspace` 读写挂载进容器
  `/target-workspace`。CUSTOM 沙箱不会自动挂载 Storage，必须在创建请求里显式
  传 `volume_mounts=[VolumeMount(host_path="/workspace", mount_path="/target-workspace")]`；
  挂载在容器创建时固定，配置变更后必须重建容器（`sandbox.json` 记录
  `mount_path` 与 `storage_host_path`，不一致即重建）。
- 执行工作区固定 `/target-workspace/.pyromind-agent/<conversation_id>`，与
  Storage 侧 `PYROMIND_AGENT_STORAGE_ROOT = "/.pyromind-agent"` +
  `conversation.id` 的既有约定对齐。
- 执行工作区根目录创建 `storage` 软链指向挂载根 `/target-workspace`。模型统一
  用工作区相对路径 `storage/...` 读取用户 Storage：`read` 工具把该别名解析到
  挂载根，`terminal` 与脚本通过同名软链走同一位置，因此 `preview_dataset` 不再是
  数据探查的主路径。写路径仍限制在 `public_data/`（与本地执行面一致，避免模型
  随手改坏用户 Storage）；产物落 Storage 仍走 `upload_file_to_pyromind` 与平台
  节点，system prompt 明确要求模型把 `storage/` 当只读。
- 容器内挂载点做成可配置常量（默认 `/target-workspace`）。PoC 第一步用
  `ls -d /target-workspace` 验证挂载存在，失败即 fail fast。
- 模型可见路径词汇保持不变：相对路径基于会话根、`public_data/` 可写、
  `.agents/skills/` 与 `knowledge/` 是资源别名。映射只发生在执行层边界；
  技能文件内容不用改，但 system prompt 里技能 `<location>` 的宿主绝对路径
  由执行层改写成别名，避免模型拿到沙箱里不存在的路径。

### 资源

会话级配置放在 `SessionSpec.extra["sandbox"]`（结构化 dict），由 PiAdapter
校验后映射为 `SandboxRequest(resources=ResourceConfig(...))`：

```json
{"sandbox": {"image": "...", "cpu": "4", "memory": "8Gi",
             "gpu": "1", "gpu_card": "L40S", "mount_path": "/target-workspace"}}
```

缺省回落顺序：会话配置 → `PYROMIND_SANDBOX_*` 环境变量 → 平台缺省
（`cpu=4`、`memory=8Gi`，与 `SandboxTool` 的 `cpu x 2Gi` 口径一致；镜像回落
集群默认镜像）。平台拒绝不带 memory 的创建请求，所以这里必须有最终缺省，
不能把 `None` 直接透传给 `ResourceConfig`。
现有 `SessionSpec.resource_limits`（memory/nproc）继续只服务本地 `os-sandbox`
后端，两种后端的资源语义不混用。

选 `extra` 而不是新增 `SessionSpec` 字段：`extra` 是既有通用逃生口，不引入
跨层 harness 类型，也不改 ports 契约；若后续多个 harness 都需要执行环境配置，
再提升为显式字段。

## 二、工作区唯一权威在沙箱

业务工具远程化意味着不能再有"本地一份、远端一份"。本地 conversation 目录
只保留控制面状态：

```text
<conversation_dir>/pi/{session.json,session.jsonl,inflight.json,
                       business-state.json,fork-index.json,run-completions.json}
```

`public_data/` 整体迁移到沙箱执行工作区，`_prepare_workspace` 不再强制创建本地
`public_data`。`apply_workspace_quota` 对远程工作区不适用，由沙箱资源限额兜底。

## 三、业务工具远程化

`BaseWorkspace` 已有 `execute_command` / `file_download` / `file_upload` 端口
（`openhands-sdk/openhands/sdk/workspace/base.py`），业务工具本该走端口而不是
直读本地 FS。本方案把绕过端口的地方修正过来，不新增 SDK 抽象。

### 1. 新增 `SandboxWorkspace(BaseWorkspace)`（harness-adapter 内）

| 端口 | 实现 |
|---|---|
| `working_dir` | `/target-workspace/.pyromind-agent/<cid>` |
| `file_download(src, dst)` | 沙箱文件 API 读到本地临时路径 |
| `file_upload(src, dst)` | 沙箱文件 API 写入（自动建父目录） |
| `execute_command` | 沙箱终端 WebSocket |
| `pause()` / `resume()` | 映射沙箱 pause/resume |

### 2. 三个工具改为走 workspace 端口

- `openhands-tools/openhands/tools/workflow/validate_workflow_dsl.py`
  （`Path(workspace.working_dir).read_text()` 一处）
- `openhands-tools/openhands/tools/workflow_debug/definition.py`（同上）
- `openhands-tools/openhands/tools/workflow/run_workflow.py`（`dsl_path` 读取）

保留各自现有的"路径必须留在 workspace 内"的越界校验。`LocalWorkspace` 行为
完全不变，OpenHands adapter 不受影响，SDK 无破坏性变更。

### 3. adapter.py 路径读写点迁移

统一走 `self._workspace(session)`：本地后端返回 `LocalWorkspace`，沙箱后端返回
`SandboxWorkspace`。

| 位置 | 现状 | 迁移后 |
|---|---|---|
| `_stage_xyflow` | `_atomic_text(root/_WORKFLOW_PATH, dsl)` | `workspace.file_upload` |
| `_workflow_snapshot` | `read_text(root/_WORKFLOW_PATH)` | `workspace.file_download` |
| `restore_workflow` | 写 `root/_WORKFLOW_PATH` | `workspace.file_upload` |
| `_is_workflow_mutation` | 本地路径比较 | 工作区相对路径比较 |
| `_copy_public_data_for_fork` | `shutil.copytree` | 沙箱内 `cp -a`（见下） |
| `_prepare_workspace` | 强制建本地 `public_data` | 只建控制面目录 |
| `_ToolConversationFacade` | `LocalWorkspace(working_dir=...)` | 注入 `SandboxWorkspace` |
| `execute_validation_tool` | 同上 | 同上 |

### 4. fork / rollback

fork 的现状：`_copy_public_data_for_fork` 把源会话 `public_data` 立即 `copytree`
到目标会话目录，然后在 checkpoint leaf 上分支 Pi 会话，并写入 checkpoint 的
工作流 DSL。

远程化后源与目标各有独立执行工作区（按 `conversation_id` 区分），所以必须复制
工作区内容，否则新会话看不到分叉点之前的数据。复制发生在同一挂载内
（`cp -a /target-workspace/.pyromind-agent/<src> .../<dst>`），不走网络。

采用**延迟复制**：fork 时只在目标会话写一个 pending 标记并立即返回，目标会话
首次访问工作区时再执行复制。理由是 fork 接口不应该为 O(数据量) 的复制阻塞，
而 fork 后不立即使用的会话不应付出复制成本。复制完成前，目标会话的工作区
访问会等待复制结束（而不是读到半份数据）。

rollback（`restore_workflow`）只是写单个 DSL 文件，直接走 `file_upload`。

### 5. 需要宿主进程的工具：host staging

`df_run_pipeline`（DataFlow 子进程）、`df_submit_pipeline`（脚本冻结与校验）、
`upload_file_to_pyromind`（打包上传）都在 agent-server 宿主上起进程或读文件，
它们拿不到容器内的路径，也不该把 DataFlow 搬进容器。这些工具保持执行位置不变，
改为按次 staging：

1. 把本次要读的工作区文件（含 `storage/...` 指向的挂载目标）拉到宿主临时目录；
2. 在临时目录里沿用原有逻辑运行；
3. 把产物（输出、日志、状态目录）打回沙箱工作区；
4. 删除临时目录。

由此 `df_run_pipeline` 直接接受 `storage/...` 输入，不再需要先
`preview_dataset(mode="sample"|"materialize")` 把数据物化进会话工作区。沙箱会话
里这两种模式仍可用：样本先写到宿主临时目录，再整体发布到沙箱工作区的
`public_data/data-preparation/previews/<key>/`，返回值仍是工作区相对路径，图片
pipeline 直接把该目录交给 `df_run_pipeline`（vision 输入会连同级图片一起
staging）。`mode="inspect"` 仍用于未挂载的 shared-space 数据。写路径仍只允许
`public_data/`；模型看到的始终是工作区相对路径，宿主临时路径不出现在工具返回值里。

## 四、沙箱连接与生命周期

token 不进 runner 的 start 帧，也不落盘。改为**懒 ensure + 按需下发**：

1. Node 侧第一次需要执行远端操作时，通过既有 JSONL RPC 发 `sandbox.ensure`。
2. Python 侧 ensure 生命周期后返回
   `{base_url, ws_base_url, sandbox_id, api_key, workspace_path, storage_path}`。
3. runner 重启后 Node 重新 `ensure` 即可拿回端点。
4. runner 是长生命周期进程，端点 bundle 可能被平台侧重建容器或轮换 access key
   变成死句柄。Node 侧遇到传输层失败时把 bundle 标记失效，下一次 resolve 带
   `{"refresh": true}` 重新 `ensure`；Python 侧收到 refresh 就丢掉进程内缓存，
   重读平台状态并用新的 client（新的 access key）组 bundle。文件 API 在刷新后
   重试一次；terminal 不重试命令本身（shell 命令不幂等），只刷新供后续调用使用。

好处：只聊天不碰文件的会话不创建沙箱；token 只在内存中传递，不进
`sandbox.json`、不进 start 帧、不进日志与错误信息。

生命周期状态机：

| 当前状态 | 动作 |
|---|---|
| 无 `sandbox_id` | create（等待 `running`） |
| `running` | 复用 |
| `paused` / `stopped` | resume |
| `error` / 平台侧不存在 | 重建，并重建执行工作区目录 |
| 会话删除 | delete 容器（Storage 侧数据保留） |
| 空闲淘汰 | pause |
| 兜底 | TTL 清理 |

持久化新增 `<conversation_dir>/pi/sandbox.json`，只存
`{sandbox_id, workspace_path, mount_path, workspace_version, created_at}`。

凭据链路复用现有实现：`RequestContext.cookie` →
`parse_auth_token_from_cookie_header`；`env`/`cluster` 复用
`business_tool_host._execution_target`。`ws_base_url` 必须用集群直连域
（`resolve_base_url_from_cluster`），portal 域不代理 WebSocket。

## 五、Node 执行层

1. `pi-session.ts` / `tools.ts`：`terminal_backend` 接受 `"sandbox"`；沙箱模式下
   `read`/`write`/`edit`/`terminal` 四件套**全部**换成远端 operations。只换
   terminal 会造成"文件工具写本地、terminal 写远端"的读写分裂。
2. 新增 `sandbox-operations.ts`：实现 `ReadOperations` / `WriteOperations` /
   `EditOperations`，走沙箱文件 HTTP API。

| Pi operation | 实现 |
|---|---|
| `readFile` | 文件 API 取 bytes |
| `access` | 读探测，404/403 映射为不可读 |
| `detectImageMimeType` | 对已取回 buffer 做魔数判断，不额外往返 |
| `writeFile` | 文件 API 写入（自动建父目录） |
| `mkdir` | `mkdir -p`（终端） |

3. 新增 `sandbox-terminal.ts`：实现 `BashOperations.exec`，per-sandbox 串行化，
   避免 TTY 输出交织。
4. 新增 `sandbox-paths.ts`：POSIX 路径解析 + 写入软限制。不复用
   `WorkspaceAccessPolicy`（它是本地 realpath 语义，在沙箱模式下不适用）。
5. skills / knowledge：会话启动时上传到沙箱，避免资源别名出现读写分裂。
   整棵树打成单个 tar 包上传后在容器内解开（251 个文件逐个走 HTTP 往返要
   30 秒左右），空文件也由 tar 正常携带。

## 六、终端执行协议

终端是裸 TTY 字节流，且 30 分钟 idle 超时由"有输出"重置，所以协议自带心跳
与结束标记。命令统一走 supervised 模式，消除"短命令前台、长命令后台"的分支：

```text
<workspace>/.pyromind-agent-runs/<call_id>/{out.log, pid, rc}
setsid bash -lc '<command>' > out.log 2>&1 &   # 沙箱内启动
tail -F out.log                                # WS 只做转发
```

- 心跳：supervisor 每 `heartbeat_seconds`（默认 60）写 `__PM_HB__<token>`，
  Node 侧剔除，仅用于保活与"仍在运行"提示，不进模型上下文。
- 结束：写 `__PM_END__<token>:<rc>`，Node 解析出 `exitCode`。
- 取消：先发 Ctrl-C，超时后 `kill -TERM -<pgid>`，再 `kill -KILL`，并清理
  心跳进程。
- 断线：用同一 `call_id` 重新 `tail` 即可恢复，命令本身不受 WS 断开影响。
- 每次调用显式 `cd <workspace_path>`，不依赖持久 `cd`（与现有 system prompt
  语义一致）。

## 七、依赖与分层约束

- 所有改动落在 `harness-adapter` 及其下层实现，符合 `rule.md`：harness 差异
  不出现在 Adapter 之上，`openhands-sdk` 保持平台无关。
- 建议把沙箱终端桥 / 文件客户端从
  `openhands-tools/openhands/tools/sandbox/definition.py` 的私有函数
  （`_terminal_websocket_url` / `_run_terminal_command`）抽成公共原语，供
  OpenHands 工具与 Pi 的 `SandboxWorkspace` 共用，避免重复实现 WS + TLS +
  certifi 处理。harness-adapter 已经在 import `openhands.tools.*`，不新增依赖
  方向。
- 会话数据、工作区与执行状态按 `conversation_id` 隔离；Cookie / Token /
  Authorization 不落盘。

## 八、改造清单

Python：

- 新增 `harness_adapter/pi_adapter/sandbox_runtime.py`（生命周期 + ensure RPC）
- 新增 `harness_adapter/pi_adapter/sandbox_workspace.py`（`SandboxWorkspace`）
- 改 `harness_adapter/pi_adapter/adapter.py`（工作区访问器、fork、RPC）
- 改 `harness_adapter/pi_adapter/terminal_backend.py`（`sandbox` 后端 + 配置校验）
- 改 `harness_adapter/pi_adapter/persistence.py`（`sandbox.json`）
- 改 `harness_adapter/pi_adapter/business_tool_host.py`、
  `business_tools.py`（注入 `SandboxWorkspace`）
- 改 `openhands-tools` 三个工具（走 workspace 端口）
- 改 `pyromind-agent-server/bootstrap.py`（注入 env/cluster 提供者）

Node：

- 改 `pi-runtime/src/pi-session.ts`、`tools.ts`
- 新增 `pi-runtime/src/sandbox-operations.ts`、`sandbox-terminal.ts`、
  `sandbox-paths.ts`

## 九、分阶段与提交序列

Phase 1（本方案主体）

1. Node 执行层：远端四件套 + supervised 终端协议（含心跳/取消/重连）
2. Python 生命周期层：`sandbox` 后端 + 懒 ensure RPC + `sandbox.json`
3. 工作区端口迁移：`SandboxWorkspace` + 三个工具 + adapter 路径读写点 + fork

Phase 2

4. 空闲 pause 在 Phase 1 已随 `close()` 落地（容器保留、按需 resume）；
   剩余 TTL 清理与孤儿容器回收
5. 指标与观测（生命周期事件、命令时长、重连次数）

Phase 3

6. staging 实测与压测、断线/重启恢复演练

## 十、测试与验收

Node 单测：marker/心跳解析与过滤、断线重连、取消与进程组清理、POSIX 路径映射
与写入软限制、文件 API 读写行为。

Python 单测：ensure 四态（create/attach/resume/recreate）、`sandbox.json` 不含
token、endpoint RPC 协议、`SandboxWorkspace` 的 upload/download/execute 映射、
fork 延迟复制与等待语义、`_is_workflow_mutation` 远端语义。

集成（staging）：`ls -d /target-workspace` 验证挂载；静默长命令跨过 idle 间隔
仍存活；流式输出、取消、断线后 reattach；workflow staging → 校验 → 调试 →
回滚全链路在远程工作区跑通；现有 Pi 数据/工作流业务工具回归。

## 十一、风险

- **业务工具远程化是本方案的主要复杂度来源**：它把本地工作区整体搬进沙箱，
  涉及 `openhands-tools` 三个工具与 adapter 多处路径读写点，需要按提交序列
  分批落地并保持 OpenHands adapter 行为不变。
- **fork 复制成本**：同一挂载内 `cp -a` 仍是 O(数据量)；延迟复制只解决接口
  阻塞，不解决复制本身。
- **Storage 一致性**：沙箱内写入经挂载落 Storage，其他消费者（如
  `preview_dataset`）看到变更可能有秒级延迟；业务工具统一走沙箱文件 API
  读取以获得强一致。
- **沙箱资源**：每个活跃会话一个沙箱，空闲 pause + TTL 兜底是控制成本的关键，
  需要确认平台侧的并发配额。
