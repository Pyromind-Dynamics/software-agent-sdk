# 历史经验 v2 接入与验证

包依赖升级为 AgentGenome Python/npm 0.6.3，通过 vendor 安装，继续由 Pi 标准加载器加载默认入口。工具定义、节点回执、修订规则、冻结和修订记录属于 AgentGenome；Runtime 使用通用 `execute_stage` 能力关联当前会话。上层没有 Pi 类型，公开 API 与原有 SSE 事件类型不变。SQLite 仅增加表，原资产及运行记录保留。

0.6.5 把「llm 节点必须保留原始执行方式」变成默认蒸馏行为（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.10）：

- llm 节点的提示词除原样包含声明路径外，还必须点名这一步原本由哪个 skill、以什么命令形态执行（脚本、`--pipeline`、模型 profile 等）。此前提示词只承载任务目标，回放的 agent 会绕开 skill 现场硬做，链路看似跑通但没沿原路径；
- 约束写在插件 prompt 的三处（系统提示词、`genome_distill` 工具描述、`steps` schema），沉淀时不需要用户额外交代；
- 只影响 llm 节点的提示词文本，`run` 节点的命令与宿主 `agent_task` 端口不变。

0.6.3 把「自由裁量的 LLM 步骤」变成图节点，而不是丢进 gaps（设计见 AgentGenome 仓库 `docs/v2-capture.md` §6）：

- `genome_distill` 的 `steps` 新增 llm 形态 `{llm, output, input?, verify?, from?}`。提示词里原样写出的输入/产出路径被替换成 `{input.*}` / `{artifact}`，节点编译成 `do: {llm}` 叶子，`verify` 默认 `test -s {output.<key>}`；
- llm 节点由宿主的 `agent_task` 端口执行：宿主提示 agent 产出声明的 artifact，agent 写文件后回报 `genome_step_result`；`GenomeService._requires_agent` 在回放前自动给 manifest 补 `agent_task` 能力要求，SDK 桥接侧无需新增分支；
- 硬边界由 `_check_llm_promotions` 把住：一条真实跑过的固定 shell 命令（哪怕脚本内部调用 LLM）写成 llm 节点、或把人工门禁/编辑工具产物写成 llm 节点，给了 `source_traces` 时会被直接拒绝——固定命令必须用 `run`，没有固定调用形式的步骤才进 `gaps`。

0.6.0 给「具体 case 资产」补上输入证据与迭代稳定性（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.9）：

- capture 冻结包时对**根输入**做有界采样（`agentgenome/capture/sample.py`）：CSV/TSV 取表头 + 前 20 行、JSONL 取前 20 条、文本取头部、二进制只留元数据，单文件 ≤ 32 KiB、总量 ≤ 512 KiB。采样走宿主读端口，落 `captures/<id>/samples/`，**不进包**（包字节是回放证据，事后改写会让发布失效）。`genome_get` 返回 `input_samples` 有界预览，`genome_list` 返回 `has_samples`；
- 复用已发布资产时先用采样的 `format` / `required_fields` 机械核对本次根输入，不符直接拒绝并说明差异；`genome_run` 的 `allow_mismatch=true` 表示明知不符仍强行复用，差异记进 run 结果而不是中断；
- 回放 A 成功后把采样输入 + 产物 `sha256` 固化成一条 `baseline-<capture_id>` 迭代基线（`agentgenome/capture/baseline.py`）。采样不完整（截断/二进制/读取失败）、参数与 manifest 不符或没有产物摘要时基线标记 `ready=false` 并写明原因；
- 改脚本的修订自动继承该基线：`revisions.prepare` 在无已冻结 suite、也无未发布 pending 时用 `captures.baseline_cases(...)` 填充 `baseline_cases`，因此「改脚本 + 新用例」不再需要手工重造旧场景用例。回归执行新增 `kind=hash`，直接比对 checkpoint 里的产物摘要，天然适用大文件；
- SDK 宿主适配新增 `ExecutionHost.head(path, limit)`：本地导出文件直接取头部，远端文件在执行面用 `head -c ... | base64` 截断后回传，避免把整份数据集拉过桥接。

0.5.1 修掉了沉淀链路上两个「链路看起来沉淀成功、复用却跑不动」的洞（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.8）：

- 脚本参数物化：`{参数名}` 指向 agent 写进工作区的脚本（`.py`/`.sh` 等）时，插件把脚本随包搬运、把命令改写成具体路径，并把它从 `params` 去掉（报告写「脚本参数已物化」）。此前这类脚本只能作为根输入留给复用方自带，包不自洽，复用必然要重写一遍；真实会话 `0e360d94d6704bbaab15437788e271b0` 沉淀出的 `question-filter-answer-sft@0.1.0` 就是这种窄口径资产；
- 复用侧执行边界写进插件 prompt：graph 覆盖的确定性步骤只能交给 `genome_run` 执行，agent 不得用文件工具重写 graph 里的脚本、也不得手工跑等价命令替代，脚本确实不适用时走 `revision_id` 通道；此前复用会话会把 graph 里的转换脚本重写一遍（2358 字符 vs 原 1731 字符）而不是调用 `genome_run`；
- 沉淀带根输入的链路时要求给 `new_params`（换一组真实输入或按 `input_contract` 造的小样本）跑回放 B，缺省时经验只能发布成未泛化的窄口径版本。

0.5.0 的沉淀链路补齐了四件事（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.6 / §5.7）：

- `genome_distill` 支持 `{参数名}` 占位符 + `params` 绑定，回放 A 用声明的真实值执行；
- 传 `source_traces` 时校验链路完整性：计划读取的产物如果由会话里更早的成功步骤产出而该步未提交，直接报错并指出缺哪一步；占位符输入作为显式边界写进报告；
- `references` 新增具体 case 资产指针 `{asset_id, version?, when?, input_mapping?}`：沉淀时解析到已发布版本并校验参数映射，不复制文件；
- 复用按两层匹配：先命中具体 case 资产直接 `genome_run`，都不命中才退到泛化资产并按 `gaps` / 文件 reference 补做 LLM、人工步骤。`genome_list` 的 `reference_assets` 直接带 `when` / `input_mapping`。

0.4.0 新增 `genome_distill`：会话里的步骤是内联代码、只读探测或脚本内部写出产物时，机械固化不会通过，agent 先把脚本落盘、把输入产出写到命令行上，再提交规范化链路；插件校验契约、编译 graph，并和 `genome_capture` 一样排队回放验证。`distill` 与 `capture` 共用同一道 origin turn 闸门和异步完成通知，SDK 侧不需要专属分支。

本版使用 `os-sandbox` 的受保护目录保存验收代码，文件工具与终端均不能修改。执行前校验摘要，模型请求通过独立文件 API 交接，不占用终端。业务数据仍留在执行环境，产物链接继续通过 SDK Storage 工具获取。

远程 `sandbox` 支持固定脚本及连续修订；验收约束由 AgentGenome 的工具说明和修订返回结果提示，不承诺运行目录只读。提交仅回收允许修改的业务脚本，验收文件继续使用冻结版本。Agent 节点（`do.llm`）由 Pi 会话执行，不依赖只读验收目录，因此 sandbox 下同样可用（`execution.capabilities` 的 `agent_task` 不再跟随 `protectedRoot`）；模型验收仍要求只读验收目录，sandbox 下保持不可用。

修订不设次数限制：Pi 使用最近的运行 ID，基于上次冻结副本继续修订，或自行判断交还 Skill。重复提交仍去重，旧版回归标准保持冻结；未通过验证的草稿用例可纠正后重新验证。失败运行可直接申请修订；成功但结果不符合需求的运行须提供 `reason`。修订副本返回 `verification_protection`（`prompt` 或 `read_only`）。原因持久化到 AgentGenome 的修订记录，旧记录兼容；Pi 能力发现仍在 Adapter 下层，Runtime 和产品协议不增加专属分支。

## 自动验证

在 SDK 根目录执行：

```sh
uv sync --frozen
npm --prefix harness-adapter/pi-runtime install
npm --prefix harness-adapter/pi-runtime run build
uv run pytest tests/pyromind_runtime
npm --prefix harness-adapter/pi-runtime test
```

`test_genome_v2_integration.py` 使用安装的插件包、真实 Pi 会话与 OS 沙箱，通过本地模型替身验证两批 CSV、阶段回执、独立模型验收、参数修订、脚本修订和失败接手。它不访问业务平台或真实模型服务，不证明模型在真实业务数据上的判断质量。

测试需要允许监听本地 HTTP 端口，并允许 OS 沙箱启动。AgentGenome 的原生服务测试还需要 Unix socket。测试只创建临时资产，不自动发布到共享业务资产目录。

## 人工测试准备

平台验证使用默认的远端执行面启动（不设置 `PYROMIND_PI_TERMINAL_BACKEND` 时即为 `sandbox`）：

```sh
./start_inference.sh
```

本地无平台凭据时改用支持的 OS 沙箱环境：

```sh
PYROMIND_PI_TERMINAL_BACKEND=os-sandbox ./start_inference.sh
```

任务包作者可参考 AgentGenome 的 `templates/data-cleaning-v2`。新包需通过宿主验证后显式发布，无策略的旧纯脚本包默认允许修订，由 Pi 声明业务脚本与验收资源；显式禁止的包仍拒绝。`genome_run` 返回后让出当前轮次；阶段任务和最终结果回到同一会话，不需要手动轮询。

本版不恢复重启前的节点任务、不自动修复或重放，不启动独立 Pi worker。取消节点只针对该请求对应的轮次；完成通知仍由 Runtime 发送一次。

## 固定脚本 revision 快速验证

在 AgentGenome 根目录使用临时资产与工作区验证，不影响正在使用的共享资产：

```sh
test_root=$(mktemp -d /tmp/genome-revision.XXXXXX)
AGENTGENOME_HOME="$test_root/registry" \
workspace_dir="$test_root/workspace" \
OH_CONVERSATIONS_PATH="$test_root/workspace/conversations" \
./start_sdk.sh test-revision
```

若有意在共享目录验证，须先停止使用同一资产目录的 SDK。

命令使用 SDK 的真实 Pi OS 沙箱，不调用模型；从旧 `data-cleaning@1.0.0` 验证两批 CSV、脚本修订、逗号与分号回归和重复提交。全部通过才自动发布候选并切换 latest。建议设置临时 `AGENTGENOME_HOME` 和 `workspace_dir` 测试。普通 `start` 不导入、发布或重放资产。

远程沙箱的提示约束修订由 Runtime 测试中的无只读能力执行宿主覆盖；真实远程平台仍需在连接的平台上做一次人工验证。测试用脚本识别逗号及分号，保持旧 `data_file` 调用可用，旧包内容不变。参数值调整不发布新版本；脚本或参数接口变更先冻结候选，再执行本次任务与全部回归。已有样例不可删除或降低断言；并发 latest 变化时保留候选，不覆盖新版本。

## terminal 数据处理

新会话不再注册 `df_run_pipeline`，历史消息类型继续兼容。普通 Python 直接通过 terminal 执行；DataFlow 使用技能的 `sandbox_sample.py`，优先 `--pipeline/--input/--output/--model-profile/--output-schema/--timeout` 直参，只有 flags 覆盖不到的字段才用 `--config` 读配置，用法见技能 `references/terminal-cli.md`。模型配置由宿主在内存中注入，远程命令脚本和日志不保存密钥；重连只恢复观察。平台提交工具不变。

远程终端按三段各自计时：端点解析（`sandbox.ensure`，含平台创建与工作区准备，默认 15 分钟）、准备（脚本落盘、TTY 握手、启动回执，默认 3 分钟）、命令执行（工具传入的 `timeout`，默认 300 秒）。落盘分三层：会话静态的 `watch.sh`、`launch.sh`、`env_runner.py` 和会话环境 `env.b64` + `env.sig` 每个会话只下一次，握手回执顺带报告它们在沙箱里是否齐全、环境签名是否匹配，命中时零额外往返，环境变化才重新落盘；每条命令只搬 `cmd.sh`，staging 行按字节预算打包（单行默认 3072 字符，落在 Linux PTY 4096 字节规范队列内，可用 `PYROMIND_PI_TERMINAL_SCRIPT_CHUNK_CHARS` 调整），每批一条确认，整包（脚本 + 启动 + 字节校验）放得下时合成一跳完成 stage 与 launch，放不下才分批或走文件 API，文件 API 缺失、失败或超出准备预算时自动降级回 TTY，校验失败的批次退回逐行确认（`batch-fallback`）。启动回执本身就证明启动行执行过，正常路径不再轮询 `pid`/`rc`，只有重连后的观察路径才先确认启动。执行证据的 `staging`、`round_trips`、`timings_ms` 字段记录本次实际走的路径、往返次数和分段耗时；`PYROMIND_PI_TERMINAL_TRACE=1` 把每次确认的耗时（每条命令的首跳含建连）和每条命令的分段耗时各写一行 JSON 到 stderr，用来把平台侧延迟与命令自身耗时分开，stderr 不在协议通道上，行为与未开启时一致。前两段预算可用 `PYROMIND_PI_TERMINAL_RESOLVE_TIMEOUT_MS`、`PYROMIND_PI_TERMINAL_STARTUP_TIMEOUT_MS` 覆盖（毫秒），`PYROMIND_PI_TERMINAL_FAST_PATH=0` 可整体退回逐行确认的慢路径。
