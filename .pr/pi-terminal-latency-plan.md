# Pi 终端每命令固定耗时优化方案

范围：`harness-adapter/pi-runtime/src/sandbox-terminal.ts` 及其调用方
（`terminal-channel.ts` / `tools.ts` / `pi-session.ts`）。
目标：把每条命令的**固定控制开销**从 ~10-12 次串行 ack 往返压到 1-2 次，
同时保留 V2 已建立的正确性属性。

## 1. 背景：固定耗时是怎么来的

真机数据（会话 `08215dbf`，workspace_version 3）：21 次 terminal 调用，
中位数 43s、最长 273s，首条 233s；同期模型耗时仅 4.5 分钟。
耗时与命令内容无关（`ls`/`wc -l` 也要上百秒），说明成本在控制握手，
不在命令执行。

稳态下每条命令的串行 ack 往返：

| 阶段 | ack 次数 |
| --- | --- |
| `ready`（顺带检查 harness 是否存在） | 1 |
| `cmd.sh` TTY staging（mkdir/truncate/chunk/decode/staged） | 5 |
| env staging（`PYROMIND_DATAFLOW_PROFILES` base64 分块） | 2-4 |
| `launch` ack | 1 |
| `started` resume 轮询 | 1-2 |
| **合计** | **10-13** |

首条命令还要额外 stage 整个 harness（~16 行），所以最贵。
每次 ack 都是一次完整往返（send → 平台 → PTY → shell → printf → 回传），
单跳约 3-4s 时正好是 40s 量级。

对照：旧版（`af0b1f89`）把整段启动脚本拼成**一次 `socket.send()`**，
0 次 ack，所以快。V2 用逐行 ack 换来了背压安全（PTY 输入队列溢出会静默
丢字节），代价是串行往返。

## 2. 设计不变量（任何改动都不能破坏）

1. **只启动一次**：`launch.sh` 的 `mkdir launched` 原子 guard 必须保留；
   ack 丢失后重发不得二次执行。
2. **断线可恢复**：命令启动后掉线只恢复观察，绝不重放；
   `pid`/`rc`/`out.log`/`offset` 语义不变。
3. **密钥不进日志**：env 不得进入 `cmd.sh`、`ps`、`out.log`；
   `stty -echo` 未确认前不得发送密钥。
4. **PTY 不溢出**：任何一次写入的字节数必须显著低于规范输入队列上限；
   单行仍 ≤512 字符。
5. **命令语义不变**：超时、取消、退出码、重连行为与现在一致。

## 3. 方案总览

| 阶段 | 内容 | 预期收益 |
| --- | --- | --- |
| P0 | 分阶段计时 + 往返计数 | 量化定位，决定后续取舍 |
| P1 | 会话级 env 落盘，命令不再重传 | -2~4 跳/命令 |
| P2 | 批量 staging + 单跳快速路径 + 字节数校验 + 降级 | 10-13 跳 → 1 跳 |
| P3 | happy path 去掉 resume 轮询 | -1~2 跳/命令 |
| P4 | WebSocket 会话级复用（已落地） | 省每命令建连 |
| P5 | watch 轮询节流调优（可选） | ≤1s/命令 |
| P6 | 平台侧 exec API（长期，需平台配合） | 根治 |

## 3.1 实施状态（本轮）

P0-P3 已落地在 `harness-adapter/pi-runtime/src/sandbox-terminal.ts`：

- 环境按会话落盘 `harness/env.b64` + `harness/env.sig`，签名变化才重写；
  `ready` 回执同时报告 harness 与环境是否可复用。
- staging 行按字节预算（默认 `TERMINAL_BATCH_BUDGET_BYTES = 3584`）打包，
  每批一条确认；整包放得下时用一条 burst 完成 stage + launch + 校验。
- 最后一批按 `wc -c` 校验每个文件的解码字节数；校验失败或 ack 超时退回
  逐行确认（evidence 记 `staging: "batch-fallback"`）。
- 启动 ack 即证明启动行执行过，正常路径不再轮询 `pid`/`rc`；只有重连
  换 socket 后才先确认启动。
- evidence 记录 `staging`、`round_trips`、`timings_ms`（ready /
  stage_harness / stage_env / upload_script / stage_script / launch /
  started / watch / total）。
- 回滚开关：`PYROMIND_PI_TERMINAL_FAST_PATH=0` 退回逐行确认的慢路径；
  该变量已加入 `pi_adapter.runner._runner_environment` 白名单。
- PTY 输入队列比 Linux 沙箱小的环境（例如 macOS 本机 PTY 测试）可通过
  `SandboxTerminalOptions.batchBudgetBytes` 调小 burst 预算。
- 诊断开关：`PYROMIND_PI_TERMINAL_TRACE=1` 让每次确认（首跳含建连）和
  每条命令各写一行 JSON 到 stderr，经 `Pi runner:` 落到产品日志；stderr
  不在 JSONL 协议通道上，开启后行为与关闭时一致。该变量同样已进白名单。

### 3.1.1 第二轮：资源缓存 + 预热 + 会话级 WS 复用（已落地）

四项改动（2026-10-12 实施）：

| # | 改动 | 位置 | 效果 |
| --- | --- | --- | --- |
| 1 | skills/knowledge 资源包按内容 hash 缓存到用户 Storage 挂载 | `pi_adapter/sandbox_runtime.py` | 命中缓存不再打包上传，只解包；首次会话才付上传成本 |
| 2 | 会话级 WS 复用，命令间不再新建连接 | `pi-runtime/src/sandbox-terminal.ts` | 省每命令建连；暖连接探测预算 30s |
| 3 | 会话创建时后台 prewarm sandbox | `pi_adapter/adapter.py`、`sandbox_runtime.py` | 首条工具调用直接拿到 ready endpoint |
| 4 | burst 预算 2048 → 3584（TS 与 Python 两侧对齐） | `sandbox-terminal.ts`、`tools/sandbox/definition.py` | staging 批数约减 1/3 |

资源缓存细节：digest 取「workspace 相对路径 + 文件字节」的 sha256 前 32 hex
（不能用 tar 字节，gzip 带打包时间戳，同内容每次不同）。缓存路径
`{storage_mount}/.pyromind-agent-resources/{digest}/pi-session-resources.tgz`，
上传先写 `...partial-<token>` 再 `mv` 到最终名，避免并发会话读到半写文件。

未做：折叠 `ready` 检查（本轮决策暂缓）。

## 3.2 本机协议地板与真机采样

本机真实 PTY（`test/fixtures/terminal_pty.py`，零网络）连跑 5 条 `echo`：

| 指标 | 数值 |
| --- | --- |
| 每命令总耗时 | 1.10-1.15s |
| watch（1s 轮询粒度） | 1.04-1.08s |
| ready / stage_script / launch | 48-76ms / 9-16ms / 5-12ms |
| ack 次数（暖会话，1024 预算） | 5（ready + 2 批 staging + 校验 + launch） |

结论：协议自身的地板是 ~1.1s，且几乎全部来自 watch 的 1s 轮询；ack 本身
在本机是毫秒级。真机上同样的轻命令却要 8-19s（见下表），差额只能在平台侧
（每次 WS 建连、平台 TTY 桥每跳延迟、容器调度）。

真机 `bdedd6962d194693b9c0db0e09dc2097`（2026-10-10 14:34，已含 P0-P3）：

| # | 命令 | 耗时 | 备注 |
| --- | --- | --- | --- |
| 0 | `read .agents/skills/.../SKILL.md` | 135.1s | 首次触达：`sandbox.ensure` 创建 + 工作区准备 + 文件读 |
| 1 | `ls -la /workspace/datasets/...` | 43.6s | 首条 terminal：首建连 + harness/env 落盘；命令本身零输出 |
| 2 | 多个 `ls -la` | 16.0s | 轻命令地板 |
| 3 | 3 × `find /target-workspace` | 25.2s | Storage 挂载遍历 |
| 4 | `ls -la; ls \| wc -l; find . -maxdepth 2` | 118.8s | 同一目录后续再访问只要 12-19s，冷/热差距明显 |
| 5 | `head -100 _report.json; ls -la` | 32.6s | |
| 6 | 3 × `ls -la <case>/` | 18.9s | |
| 7 | 6 × `find . -name ...` | 18.8s | |
| 8 | `for f in $(find ...)` | 11.8s | |

同日 `6804bab7`（P0-P3 之前）轻命令 7.7-19.5s，与 P0-P3 之后的地板同量级，
说明**地板不是 staging 往返**；BAC 省掉的是 HTTP 落盘与 resume 轮询，而不是
平台每跳延迟。

待办：跑一条带 `PYROMIND_PI_TERMINAL_TRACE=1` 的真机会话，用
`round_trip=1`（含建连）与 `round_trip≥2`（纯确认）的差值判断 P4（会话级
WS 复用）值不值得做；watch 段耗时决定 P5（轮询粒度）是否要调。

## 4. 详细设计

### P0 分阶段计时（先做，0.5d）

- `TerminalExecutionEvidence` 增加 `timings_ms`（resolve / socket_open /
  ready / stage_harness / stage_script / stage_env / launch / started /
  first_output / watch_end / total）与 `round_trips`。
- 计数点：每次 `channel.acknowledge()` +1；watch 的 `receive()` 不计往返。
- 成功路径输出一条 `console.debug("[pi-terminal] ...")`，失败路径随
  evidence 抛出；两者都进产品日志，便于真机取样。
- 目的：确认单跳延迟分布，判断 P4/P5 是否值得做。

### P1 会话级 env 落盘（1d）

现状：`options.env`（`PYROMIND_DATAFLOW_PROFILES`，约 1-2KB base64）
每条命令重传，占 2-4 跳。

- 客户端对 env JSON 取 sha256 作为签名；
- `ready` 行扩展为「harness 文件齐 + `harness/env.b64` 存在 +
  `harness/env.sig` 匹配」才回 `ok`，否则回 `missing`；
- 缺失时用 `(umask 077)` 写 `env.b64` + `env.sig`，整个会话只写一次；
- `launch` 行改为
  `PYROMIND_TERMINAL_ENV="$(base64 -d < harness/env.b64)" sh launch.sh <runDir>`；
- 无 env 时保持现有行为。

安全权衡：文件 600 权限，位于用户自己的会话沙箱内，内容是该用户自己的
密钥；`out.log` 脱敏逻辑不变。若要求更严，可选「shell 内存缓存」变体，
但依赖 shell 会话跨 WS 持久，需要先验证。

### P2 批量 staging + 单跳快速路径（1.5d）

现状：`buildStageLines` 每行自带 ack，`stageOverTerminal` 逐行 await。

- 新增 `buildStageBatches(dir, token, files, budgetBytes)`：把 staging 行按
  字节预算切批，每批末尾一条 ack 行；
- **快速路径**：整包（cmd.sh + launch + 校验）≤
  `TERMINAL_BATCH_BUDGET_BYTES`（最终值 3584）时，用一个 `receive()` 完成
  stage + launch + ack，即 **1 跳**；
- **长载荷**：每批 ≤ budget，一批一个 ack；首次 harness ~5KB →
  5-6 跳（原来 ~16）；
- **校验升级**：最后一行对每个文件比对
  `wc -c` 与客户端预期解码字节数（现有检查只验非空），任何丢字节都会
  改变长度并暴露；
- **降级**：校验失败或快速路径超时 → 回退现有逐行 ack staging，
  evidence 记 `staging: "batch-fallback"`；重发由 launch guard 保证不二次执行。

预算依据：Linux 规范输入队列 4096 字节。3584 是刻意贴着上限取值（收益是
批数最少），因此**每批字节校验 + 失败回退逐行 ack 是强制项**，不是可选
优化；行内仍 ≤512 字符，两个约束同时满足。PTY 队列更小的环境（macOS
本机测试）通过 `batchBudgetBytes` 调小。

### P3 happy path 去掉 resume 轮询（0.5d）

现状：launch ack 后还要轮询 `pid`/`rc` 确认 started，1-2 跳。

- 同一连接上 launch ack 成功即直接发 watch 行进入观察；
- `resume` 只保留给重连路径（attempt > 0 或 dispatched 后换 socket）；
- 不变量：BEGIN marker 仍证明观察开始，`pid`/`rc` 仍由 `cmd.sh` 写入。

### P4 WebSocket 会话级复用（已落地）

原状：每条命令 `this.channel()` 新建、`finally` 关闭。实测 connect 占比
显著（首条命令 43.6s 中相当部分在建连与 harness 落盘），故实施。

- `SandboxTerminalOperations` 持有会话级 channel（`private session`），
  exec 时复用（`sessionExchanges` 计数判断暖连接）；
  仅在连接丢失 / quarantine / session dispose 时关闭重建；
- 残留输出隔离：per-command 18-hex token，ack 只匹配自己 token；
  无 pending 时到达的旧输出直接丢弃；
- 暖连接探测预算 `DEFAULT_SESSION_PROBE_MS = 30_000`，超时才重连；
- session dispose 钩子已接（`LazySandboxTerminalOperations` 清空
  operations 前调 `dispose()`；`terminal-channel.ts` 暴露 `dead` getter）。

### P5 watch 轮询调优（可选）

`sleep 1` → `sleep 0.2`（POSIX 支持小数），心跳 ticks 相应换算。
收益 ≤1s/命令，P0 数据显示首字节/收尾延迟占比高才做。

### P6 平台侧 exec API（长期）

建议平台提供 `POST /sandboxes/{id}/exec`（流式返回）：一次请求完成命令
执行，不再用 PTY 做控制通道，`out.log`/`watch.sh`/轮询全部退场。
这是根治方案，需要平台侧排期，本方案不阻塞。

## 5. 行为对比

| 指标 | 原状 | 现在 |
| --- | --- | --- |
| 稳态 ack 往返/命令 | 10-13 | 1（快速路径）/ 2-4（长载荷） |
| 首命令（harness+env） | ~16-25 跳，实测 233s | 6-8 跳，目标 ≤30s |
| 单命令中位数（真机） | 43s | 目标 ≤8s |
| 每命令 HTTP | 0（短）/ 3（长） | 不变 |
| 每命令 WS 建连 | 1 | 0（会话级复用，断线才重建） |
| 资源 staging（非首次会话） | 每次上传 ~MB 级 tar | 命中缓存 0 上传 |
| 首条工具调用前 | 现场建容器 | prewarm 已就绪 |
| 正确性属性 | — | 五项不变量全部保留 |

## 6. 验证方案

- **单测**（`test/sandbox-terminal.test.ts`）：
  - batch 切分边界（0 / 1 / 1024 / 1025 字节）；
  - 校验行含预期字节数，长度不符触发 fallback；
  - env 签名命中/失配（失配重写）；
  - resume 仅跳过 happy path，重连路径仍走 resume；
  - launch guard 重发不二次执行（扩展现有测试）。
- **PTY 集成**（`test/fixtures/terminal_pty.py`）：
  - 真实 PTY 下批量 ≤3584B 全部执行；
  - 注入丢字节 → 校验失败 → fallback 成功；
  - 统计每命令 ack 次数，断言 ≤2。
- **Python 侧**（`tests/pyromind_runtime/`）：
  - `test_ensure_reuses_the_cached_resource_archive`：第二次会话只有 1 次
    上传；
  - `test_prewarm_starts_the_sandbox_before_the_first_touch`：prewarm 提前
    建沙箱且并发去重；
  - `test_prewarm_can_be_switched_off`：`PYROMIND_SANDBOX_PREWARM=0` 关闭。
- **真机**：跑一条完整预打标链路，对比 P0 计时前后；
  断言中位数 ≤8s、首命令 ≤30s；重连/中断回归不劣化。
- **回归**：现有 `sandbox-terminal*.test.ts` 全绿；`uv run pytest
  tests/pyromind_runtime` 相关用例不变。

## 7. 风险与回滚

- 全部改动在客户端（pi-runtime），无平台协议变更，插件开源适配不受影响；
- 提供 `PYROMIND_PI_TERMINAL_FAST_PATH=0` 一键回退到逐行 ack 现状；
- 3584 预算贴着 PTY 4096 上限，正确性依赖每批字节校验与 fallback；
  要更保守可只调小 `batchBudgetBytes`，不改协议；
- 会话级 WS 复用引入「长连接残留输出」风险，靠 per-command token 隔离 +
  `retire()`/`dispose()` 收口；出问题可回退到每命令新建连接；
- prewarm 会在会话创建时提前占用沙箱资源，可用
  `PYROMIND_SANDBOX_PREWARM=0` 关闭；
- 资源缓存落在用户 Storage 挂载内，天然按用户隔离；digest 命中失败只会
  退化为重新上传，不会用错版本（digest 变了就是另一个目录）。
- env 落盘是本方案唯一的语义新增，若不接受，可退化为
  「shell 内存缓存」或保持每命令重传（只损失 2-4 跳）。

## 8. 实施顺序与工作量

| 阶段 | 工作量 | 依赖 |
| --- | --- | --- |
| P0 计时 | 0.5d | 无 |
| P1 env 落盘 | 1d | P0 |
| P2 批量 + 快速路径 | 1.5d | P1 |
| P3 去 resume 轮询 | 0.5d | P2 |
| 真机验证 | 0.5d | P1-P3 |
| P4 WS 复用 | 已完成 | P0 数据 |
| 资源缓存 + prewarm + 3584 预算 | 已完成 | P2/P4 |
| P5 轮询调优 | 0.5d | P0 数据 |

合计约 4-5 人日（P0-P3 + 验证）；P4 与资源缓存/prewarm/预算四项已完成，
剩余 P5 视 P0 数据决定。
