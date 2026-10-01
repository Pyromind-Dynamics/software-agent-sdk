# 历史经验 v2 接入与验证

包依赖升级为 AgentGenome Python/npm 0.3.0，通过 vendor 安装，继续由 Pi 标准加载器加载默认入口。工具定义、节点回执、修订规则、冻结和修订记录属于 AgentGenome；Runtime 使用通用 `execute_stage` 能力关联当前会话。上层没有 Pi 类型，公开 API 与原有 SSE 事件类型不变。SQLite 仅增加表，原资产及运行记录保留。

本版使用 `os-sandbox` 的受保护目录保存验收代码，文件工具与终端均不能修改。执行前校验摘要，模型请求通过独立文件 API 交接，不占用终端。业务数据仍留在执行环境，产物链接继续通过 SDK Storage 工具获取。

远程 `sandbox` 支持固定脚本及连续修订；验收约束由 AgentGenome 的工具说明和修订返回结果提示，不承诺运行目录只读。提交仅回收允许修改的业务脚本，验收文件继续使用冻结版本。Agent 节点和模型验收仍保持原有能力限制。

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

使用支持的 OS 沙箱环境启动：

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

新会话不再注册 `df_run_pipeline`，历史消息类型继续兼容。普通 Python 直接通过 terminal 执行；DataFlow 使用技能的 `sandbox_sample.py --config`，配置格式见技能 `references/terminal-cli.md`。模型配置由宿主在内存中注入，远程命令脚本和日志不保存密钥；重连只恢复观察。平台提交工具不变。
