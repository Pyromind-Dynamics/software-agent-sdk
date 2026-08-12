# codex 在沙箱内的安装与使用

处理载体为 OpenAI Codex CLI(LLM 命令行代理),在沙箱内以非交互模式
执行数据处理任务。

## 安装(沙箱内,sync 模式)

```json
{"operation": "exec", "sandbox_id": "<id>", "mode": "sync", "command": "node --version || (curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs) && npm install -g @openai/codex && codex --version", "timeout": 300}
```

- 前置要求:Node.js ≥ 18;沙箱镜像通常已带 node,先探测再装
- 安装完成后用 `codex --version` 验证

## 认证

codex 需要 OpenAI 凭据。优先通过环境变量注入,避免把 key 写进脚本:

```json
{
  "operation": "exec",
  "sandbox_id": "<id>",
  "mode": "sync",
  "environment_variables": {"OPENAI_API_KEY": "<key>", "OPENAI_BASE_URL": "<base-url>"},
  "command": "codex exec ..."
}
```

> 敏感 key 只应来自会话 secret(如 `auth_token` 之外的专用 secret),
> 不得硬编码在 skill 或命令里。

## 非交互执行

```bash
codex exec --sandbox danger-full-access -C /workspace/task \
  "对 /workspace/repo 执行 Java 1.8 特性筛选,结果写入 /workspace/result.json"
```

关键参数:

- `exec`:非交互单轮执行(不进入 REPL)
- `--sandbox danger-full-access`:允许 codex 直接读写沙箱文件系统
  (沙箱本身已隔离,无需 codex 再套一层沙箱)
- `-C <dir>`:工作目录
- `--full-auto`:跳过确认,全自动执行(可选,配合非交互)

## 与工具模式配合

| 场景 | 方式 |
| --- | --- |
| 短任务(< 2 分钟) | `mode="sync"` 直接跑 codex exec |
| 长任务(分钟级) | `mode="background"` 启动 + `mode="terminal"` tail 日志 |
| 需要秒级进度反馈 | `mode="terminal"` 直接跑 codex exec |

## 常见问题

- **codex 输出乱码/进度条**:加 `--output-format json` 或
  `--skip-git-repo-check`(非 git 目录时)
- **找不到命令**:确认 PATH;npm 全局 bin 一般在 `/usr/local/bin` 或
  `$HOME/.npm-global/bin`
- **网络受限**:codex 需要访问模型 API 域名;镜像若屏蔽外网,先
  `curl -I https://api.openai.com` 验证
