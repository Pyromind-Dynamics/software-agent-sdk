# AgentGenome v1 接入

AgentGenome 管共享资产、版本和图内状态。SDK Runtime 管会话、权限、排队和产品事件；HarnessAdapter 将执行请求接到当前 Pi 会话的终端后端。固定脚本直接执行，不经过第二次 LLM 调用。

AgentGenome 0.2.0 在原生 Pi 与 SDK 中使用同一个默认插件入口。SDK 通过 Pi 的 `additionalExtensionPaths` 从已安装包读取插件，不再通过 `extensionFactories` 创建 AgentGenome。插件通过 Pi 事件总线发现 SDK 的宿主能力；工具定义、请求封装及 Python 操作分发都在 AgentGenome，SDK 只保留沙箱、权限、会话和通知适配。启动时校验插件已连到 SDK，失败不回退为本机执行。

原生 Pi 的安装、`/genome setup` 和共享本地服务由 AgentGenome 自带，不需要修改 Pi。SDK 仍使用独立的 `workspace/agentgenome` 目录，不与正在运行的原生服务争用同一执行锁。对外产品协议和已有持久化模型不变。

## 名称与选用顺序

面向 Agent 和用户，AgentGenome graph 统一称为“历史经验”；“工作流”指 Pyromind 平台 DSL。历史经验（graph + scripts）和 Skill 都可复用：执行任务先查 `genome_list`，用 `genome_get` 确认适用性，合适则复用，否则再选 Skill。闲聊无需查询。工具名 `genome_*` 和内部工作流接口保持兼容。

Storage 输入须保留 `storage/` 前缀，例如 `storage/agentTest/input.csv`。`agentTest/input.csv` 指当前会话下的相对路径，并非同一文件；平台的 `/workspace/...` 也不能直接当作当前沙箱路径。使用 `read`/`terminal` 已验证成功的原路径。

## 快速测试与启动

在 SDK 目录执行：

```bash
# 先停止使用同一资产目录的 SDK 服务
./start_inference.sh --test-agentgenome
# 验证通过后启动服务；沿用原来的模型 Key、DataFlow Key 和登录配置
./start_inference.sh --agentgenome
```

也可以在 AgentGenome 目录执行 `./start_sdk.sh test`、`./start_sdk.sh start`。两个入口调用同一套 SDK，不启动独立的 AgentGenome 服务。

测试命令安装锁定的 Python/npm 依赖、编译 Pi runtime，然后通过 SDK 的实际 Pi OS 沙箱执行两批 CSV。不调用模型或业务平台，不需要 API Key。预期两次输出 `succeeded`，清洗后分别为 2 行和 1 行，最后输出 `published`。已安装依赖时，可用 `AGENTGENOME_SKIP_INSTALL=1 ./start_inference.sh --test-agentgenome` 跳过安装。

这两个启动入口现在都走 Pi 标准插件加载链路。`pi-runtime` 的 `genome-plugin.integration.test.ts` 另外通过真实 Pi 会话和本地模拟模型验证 `genome_list/genome_run → SDK RPC`，覆盖包加载、宿主发现、重载、关闭开关和禁止本地回退。CSV 验证负责真实沙箱执行，不需要真实模型调用。

模板会复制到 `workspace/agentgenome/assets/data-cleaning/1.0.0/`，验收后发布；测试输入和产物在 `workspace/genome-validation/`。重复测试创建新运行，不能用不同内容覆盖已发布版本。测试固定使用本地 `os-sandbox`；真实平台沙箱需在聊天中另测。

资产目录可通过 `AGENTGENOME_HOME` 配置，默认取 `OH_CONVERSATIONS_PATH` 的同级 `agentgenome` 目录；一份目录只能由一个 SDK 执行进程持有。非相邻仓库可配置 `AGENTGENOME_DIR`（SDK 入口）或 `SOFTWARE_AGENT_SDK_DIR`（AgentGenome 入口）。

启动服务时，执行后端仍由 `PYROMIND_PI_TERMINAL_BACKEND` 决定。无启动选项时插件默认关闭，也可设置 `PYROMIND_AGENTGENOME_ENABLED=1` 启用。Python wheel 和 Pi 扩展 npm tarball 随 SDK 锁定；修改 AgentGenome 源码后需按下文重新打包，启动脚本不会跨仓库加载源码。

## 在聊天里验证

启动服务后新建会话，发送：

> 查看 genome_list，再读取 data-cleaning 1.0.0 的说明。在当前会话的 public_data/input.csv 创建 CSV，内容为 name,age 表头，以及 Ada,36、Ada,36、Bob,、Cy,12 四行。调用 genome_run，参数 data_file 为 public_data/input.csv，复用资产中的脚本完成清洗。提交后结束本轮，等待运行结果。

应看到节点日志和成功结果，报告为 2 行。再让 Agent 创建另一份 CSV，仅更换 `data_file` 复用同一版本；用不存在的文件测试失败状态。可用 `genome_status` 查询运行详情。自动测试目录不属于新会话，请使用该会话内的文件。

在实际会话中，Pi 可通过以下工具使用资产：

- `genome_list`：所有 latest 已发布包的简介。
- `genome_get`：指定包版本的参数和产物说明。
- `genome_run`：传入 `asset_id`、`version`、`params`，立即返回运行 ID。
- `genome_status` / `genome_cancel`：查询或取消本会话运行。

路径参数使用当前会话可访问的路径，如 `public_data/input.csv`；平台 sandbox 后端也支持 `storage/...`。不能传服务端任意路径。图启动前会冻结版本并校验参数，Pi 本轮结束后才开始运行。

## 行为与限制

- 一个会话最多运行一个图；执行期间普通消息沿用原有会话机制，取消仍可用，执行环境不会被空闲回收。
- 复用现有 `operation` 和 `status.changed` 事件，前端无需增加协议。运行中展示节点、命令、验收和日志尾部；日志不反复唤醒模型。完成后 Runtime 直接通知 Agent 读取实际产物并继续回复，不经过 Kafka。取消只更新上下文，不自动续聊；完整日志保存在 AgentGenome 运行目录。
- 脚本和输出位于会话执行环境的 `public_data/workflow-runs/<run_id>/`。服务端只获取摘要、日志和最多 64 KiB 的验收结果，不回传整批数据。
- 资产脚本和验收辅助脚本通过 SDK 文件接口写入运行目录：平台沙箱复用 HTTP 文件上传，本地后端执行路径权限校验后写入。终端只执行脚本路径，不携带编码后的脚本内容。
- 命令沿用 SDK 终端后端和权限策略；默认单条命令超时 300 秒。大体积验收应写成脚本，仅输出小型 JSON 结论供 metric 判断。
- 原有 YAML 重试规则继续有效；失败不会自动让 Pi 改脚本。取消无法确认、连接异常或服务重启时记录为中断，不自动续跑或重试副作用。
- 本版未实测连接真实 Pyromind 远端沙箱，需要在有平台凭据的环境补充验证；实际 OS 沙箱和远端传输模拟均有测试。

## 两个仓库的依赖更新

修改 AgentGenome 后重新构建发行包，再更新锁文件；不要向 SDK 复制 Python/插件源码：

```bash
uv build --wheel --out-dir vendor ../AgentGenome
cd harness-adapter/pi-runtime
npm pack ../../../AgentGenome/pi-extension --pack-destination vendor
npm install --ignore-scripts ./vendor/agentgenome-pi-extension-0.2.0.tgz
npm run build
cd ../..
uv lock --upgrade-package agentgenome --refresh-package agentgenome
uv sync
```

升级版本时同步更新根 `pyproject.toml` 的 wheel 路径和 `harness-adapter/pyproject.toml` 的版本约束，再生成锁文件。npm 包包含同版 Python wheel；SDK Python 环境仍通过正常 wheel 依赖安装，不使用插件的本地环境初始化流程。

对外 API 和已有 Product Store 模型不变。新增通用工作流端口；AgentGenome 的 SQLite 独立管理，不复用原画布工作流记录。

验证命令：`uv run pytest tests/pyromind_runtime`、Pi runtime 的 `npm test` 和 `npm run check`，以及 AgentGenome 的 `pytest tests` 与插件测试。

## 异步完成通知

历史经验以 `historical_experience` 接入通用异步任务记录。结果引用、是否自动续聊和待通知状态保存在 Product Store；Pi 将已接收的通知写入会话日志，按通知 ID 去重，正在回复时排队。重新打开会话会补送未完成交接的通知、查询已中断 graph 的状态，不自动重跑脚本。旧任务数据缺少新增字段时使用兼容默认值。

验证聊天：提交历史经验后应立即得到运行 ID，卡片持续显示节点；无需追问，成功后 Agent 自动读取报告并回复。失败只解释原因，取消不发起新轮次。执行期间刷新页面或发送新消息，卡片不重复、结果不打断正在进行的回复。

## 消息关联与文件交付

用户消息事件和快照增加可选 `command_id`，用于确认前端发送气泡；`run_id` 只标识实际执行轮次。旧消息未提供命令 ID 时沿用运行 ID 匹配。快照新增可选 `active_run_id`，用于防止旧轮次结束事件覆盖正在执行的新轮次。新增字段均有兼容默认值。

通知恢复识别 Pi 的原生 `custom_message` 和完成回执；每次续聊使用独立执行 ID，通知 ID 保持稳定以便去重。已有会话历史不自动重写。

产物交付默认调用 `get_storage_url`，以 `[清洗结果](url)`、`[清洗报告](url)` 等 Markdown 链接回复。工作区产物可直接解析，无需先询问复制到 Storage；解析失败则说明原因并保留真实路径，不伪造 URL 或擅自移动文件。

## 远端 Terminal 超时与停止确认

共享 Terminal 通过沙箱文件接口上传 `cmd.sh`、`watch.sh` 和 `launch.sh`，PTY 只发送短控制命令。环境变量留在内存，关闭回显、历史记录和历史展开后按 512 字符分段确认。普通 terminal 与历史经验使用同一实现，能力留在 HarnessAdapter 下层。

取得执行锁后计时：端点查询、连接、准备和启动共用 30 秒期限，计入默认 300 秒总期限。断线只恢复观察，不重发启动；取消或超时最多等待 10 秒确认进程组停止。无法确认时记录 `interrupted` 并阻止下一条命令，直到确认原进程停止。

执行适配将执行 ID、阶段、启动/停止确认和远端目录转换为通用执行证据。AgentGenome 保存证据并保留中断状态；现有 API 和历史数据兼容，不增加自动重跑或 revision 入口。

测试包括真实 POSIX PTY 的长脚本、大环境变量、分段确认丢失、无输出和进程组取消。可用 SDK 安装包在临时目录验证真实 `1.0.4` 资产：

```bash
uv run python scripts/validate_agentgenome.py \
  --template workspace/agentgenome/assets/data-cleaning/1.0.4 \
  --home /tmp/genome-terminal-check/assets \
  --workspace /tmp/genome-terminal-check/conversation
```

真实远端验收沿用同一资产和执行入口，由 SDK 会话提供 `sandbox.ensure` 端点。使用专用测试沙箱，不将凭据写入脚本或命令历史；验证只操作指定临时资产目录和独立运行目录，不改公共 latest 或重放历史会话。
