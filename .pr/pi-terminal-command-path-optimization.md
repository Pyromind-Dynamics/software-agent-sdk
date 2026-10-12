# Pi 终端指令下发链路：优化方案与实际改动

范围：`harness-adapter/pi-runtime/src/sandbox-terminal.ts` 及其调用方
（`sandbox-operations.ts` / `tools.ts` / `pi-session.ts` / `pi_adapter/runner.py`）。
本文说明 V2 之后的指令下发路径为什么慢，以及这次解耦怎么改、改完是什么行为。

## 1. 背景：V2 的指令路径

每次 `bash` 调用由四段组成：

1. 文件 API 写三个脚本到 `${runsRoot}/<callId>/`：`cmd.sh`、`watch.sh`、`launch.sh`；
2. 打开 TTY，`stty -echo && unset HISTFILE ...` 握手，再分块把环境变量 base64
   塞进 `PYROMIND_ENV_BUFFER`；
3. `sh ${runDir}/launch.sh`：`mkdir launched` 原子 guard + `setsid sh cmd.sh`，写 `launch_pid`；
4. `sh ${runDir}/watch.sh`：轮询 `out.log` 增量输出、发心跳、读 `rc` 收尾。

三个脚本的职责（这部分设计是对的，本次保留）：

| 文件 | 职责 |
| --- | --- |
| `cmd.sh` | 单次命令：写 `pid`、`trap` 写 `rc`、`cd` 工作目录、执行命令 |
| `launch.sh` | 原子 guard + `setsid` 启动，保证「只启动一次」，断线后不重放 |
| `watch.sh` | 观察 `out.log` 增量、心跳、退出码，跨重连继续收尾 |
| `env_runner.py` | 解码 base64 环境变量、按 key 脱敏后 exec，密钥不进 `cmd.sh` / `ps` |

## 2. 问题

1. **每次执行都要写 3 个文件**。文件 API 每个文件 3 次 HTTP（init / part / complete），
   一条短命令也要 9 次请求；而 TTY 此时已经开着，指令本身很短。
2. **`writeFile` 是硬依赖**。文件 API 不可用直接
   `throw new Error("sandbox terminal requires a file upload interface")`，
   而终端通道本来是可用的。
3. **30 秒预算是共享的**。endpoint resolve 和启动握手共用 `STARTUP_TIMEOUT_MS = 30_000`，
   但控制面 resolve 自身预算是 600s 创建 + 120s prepare；创建稍慢就报 `startup_timeout`。
4. **重连没有新预算**。第一次握手花掉的 startup 预算不会重置，重连后的轮询可能直接超时。

## 3. 方案：三层解耦

### 3.1 会话级 harness 与单次载荷分离

`watch.sh` / `launch.sh` / `env_runner.py` 不含任何每次执行的状态，属于会话级基建；
只有 `cmd.sh` 是每次命令的载荷。

- harness 写到 `${runsRoot}/harness/`（`SANDBOX_HARNESS_DIRNAME`），整个会话只落一次；
- ready 握手顺带 `[ -s harness/<name> ]` 检查，健康会话零额外成本，缺失才补；
- `launch.sh` 改为从 `$1` 取 run 目录，`buildStartLine(harnessDir, runDir, token, ...)`；
- `cmd.sh` 里的环境执行器从内联 `python3 -u -c <runner>` 改为引用
  `${harnessDir}/env_runner.py`，`cmd.sh` 因此更小、可读。

### 3.2 传输通道按成本选，不按能力选

- 短载荷（staging 行 ≤ `TERMINAL_STAGING_MAX_LINES = 12`）直接走已打开的 TTY：
  base64 分块（`TERMINAL_SCRIPT_CHUNK_CHARS = 512`，低于 macOS 1024 / Linux 4096 的
  规范输入上限）、逐行 ack、最后 `base64 -d` 解码并校验，全程 0 次 HTTP；
- 长载荷仍走文件 API（3 请求/文件，行数多时才划算）；
- 文件 API 缺失、失败、超时 → **自动降级回 TTY staging**，命令不再失败；
- 上传超时后 `abort()`，避免迟到的分片覆盖 TTY 刚写入的内容；
- `TerminalExecutionEvidence` 增加 `staging: "file" | "terminal"`，报告里能看出走了哪条路。

### 3.3 预算分离

- resolve 独立预算：`DEFAULT_RESOLVE_TIMEOUT_MS = 900_000`，
  环境变量 `PYROMIND_PI_TERMINAL_RESOLVE_TIMEOUT_MS`；
- startup 默认 30s → `DEFAULT_STARTUP_TIMEOUT_MS = 180_000`，
  环境变量 `PYROMIND_PI_TERMINAL_STARTUP_TIMEOUT_MS`；
- `prepareDeadline()` 让每次握手/轮询重新取一份 startup 预算，命令 deadline 兜底；
- 命令默认超时 300s 收敛为 `DEFAULT_COMMAND_TIMEOUT_SECONDS`。

### 3.4 重连与失败语义收紧

- 启动前掉线：在替换 endpoint 上重放握手（脚本已在盘上），最多
  `MAX_ENDPOINT_REFRESHES = 1` 次，命令本身不重放；
- 启动后掉线：只观察，不重放；
- 失败路径：只有 `dispatched`（launch 行已发出）才走 `stop()` 确认进程组；
  未 dispatch 只清 `PYROMIND_ENV_BUFFER`（`clearEnvironment`），
  不再为没有进程组的失败付一次换 socket 的代价。

## 4. 实际改动（按文件）

### `src/sandbox-terminal.ts`（+376 / -75）

- 新增常量与类型：`SANDBOX_HARNESS_DIRNAME`(16)、`MAX_ENDPOINT_REFRESHES`(35)、
  `DEFAULT_RESOLVE_TIMEOUT_MS`(41)、`DEFAULT_STARTUP_TIMEOUT_MS`(42)、
  `DEFAULT_COMMAND_TIMEOUT_SECONDS`(43)、`TERMINAL_STAGING_MAX_LINES`(54)、
  `SandboxTerminalTimeouts`、`sandboxTerminalTimeouts()`(115)、
  `RunScripts`、`HarnessFiles`、`TerminalStageLine`；
- `writeFile` 由必需改为可选且带 `signal`；
- `buildCommandScript(..., environmentRunner?)`(300)：环境执行器改为文件路径引用；
- `buildLaunchScript()`(361)：从 `$1` 取 run 目录，不再把 runDir 编进脚本；
- 新增 `buildStageLines()`(396)、`buildStageCheckLine()`(426)；
- `buildStartLine(harnessDir, runDir, token, ...)`(442)；
- `stageScripts()`(585)：选通道（TTY / 文件 API），失败降级；
- `stageOverTerminal()`(620)：逐行 ack 的 TTY staging；
- `clearEnvironment()`(556)、`prepareDeadline()`(572)；
- `execute()`：harness 构造(667)、ready ack 顺带检查 harness、按 staging 结果补写、
  启动前掉线重放、轮询独立预算；
- `LazySandboxTerminalOperations.resolveOperations()`(834)：resolve 独立预算，
  启动前掉线才在替换 endpoint 上重试一次(822)；
- 新增 `isConnectionLost` / `isStaleTerminal` / `isPreLaunchDrop`(893)；
- `ENVIRONMENT_RUNNER`(269) 导出，供 harness 目录复用。

### `src/sandbox-operations.ts`

- `write(path, content, signal?)`(100)：signal 贯穿 upload 全链路；
- `withEndpoint`(161)：重试升级为「同一 endpoint 先重试一次 → 仍失败才
  `invalidate()` 换 bundle」，只有未到达沙箱的传输失败才重试（`isTransportFailure`(272)）；
- `fetch` 超时与外部 signal 用 `AbortSignal.any` 合并(226)。

### `src/tools.ts` / `src/pi-session.ts` / `pi_adapter/runner.py`

- `CreateToolsOptions.terminalTimeouts`(67) 透传到 `LazySandboxTerminalOperations`(236)；
- `pi-session.ts` 注入 `terminalTimeouts: sandboxTerminalTimeouts()`(104)；
- `runner.py` 白名单转发 `PYROMIND_PI_TERMINAL_RESOLVE_TIMEOUT_MS` /
  `PYROMIND_PI_TERMINAL_STARTUP_TIMEOUT_MS`(307-308)。

### 测试

- `test/sandbox-terminal.test.ts`（+309）：staging 通道选择、降级、harness 复用、
  resolve 独立预算、启动前/后掉线语义；
- `test/sandbox-terminal-pty.test.ts`：真实 PTY 下无文件 API 的内联 staging；
- `test/sandbox-operations.test.ts`：write abort 不触发 endpoint 刷新。

## 5. 行为对比

| 场景 | 改动前 | 改动后 |
| --- | --- | --- |
| 每次执行的 HTTP 请求 | 3 文件 × 3 = 9 次 | 短命令 0 次；长命令 3 次 |
| 没有文件 API 的沙箱 | 直接报错 | 走 TTY staging，可用 |
| 文件 API 慢/超时 | 命令失败 | 自动降级，命令继续 |
| harness 脚本 | 每次执行重写 | 会话内只落一次 |
| endpoint 创建 > 30s | `startup_timeout` | resolve 有独立 900s 预算 |
| 重连后的轮询预算 | 沿用已耗尽的 startup 预算 | 每次重新取预算 |
| 未启动就失败的清理 | 也走一次 stop/换 socket | 只清环境缓冲 |

## 6. 验证

```
cd harness-adapter/pi-runtime
npm test        # 104 passed
npm run build   # tsc
npm run check   # tsc --noEmit
```

另：`uv run pytest tests/pyromind_runtime -q` 284 passed；
`uv run pre-commit run --files ...` 全通过。

## 7. 同一分支里的其它改动（非指令路径）

- `pi-session.ts` 增加模型调用重试（`PYROMIND_PI_MODEL_MAX_RETRIES` /
  `PYROMIND_PI_MODEL_RETRY_BASE_DELAY_MS`，默认 2 次 / 2s），用于网关断开后
  重开一轮助手回合；`runner.py` 同步转发这两个变量。
- `sandbox_runtime.py` 布局版本 3：workspace 里补建 `public_data` 目录，
  保证用户按 `workspace/public_data/...` 写的路径在沙箱内可解析。
