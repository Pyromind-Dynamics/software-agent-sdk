# 框架说明:Profile 模板体系、Sandbox 工具组合与 skill 分层

## 分层架构

```
environment-data-processing skill(SKILL.md 工作流编排 + 场景识别)
        │ 匹配
        ▼
ProcessingProfile 模板(profiles/*.json,声明式 steps + verdict)
        │ 驱动
        ▼
runner 冻结运行时(scripts/sandbox_runner.py:逐条循环/断点续跑/强制清理)
        │ 调用
        ▼
工具层 sandbox_create / sandbox_terminal / sandbox_upload / sandbox_delete 等
        │ 复用
        ▼
pyromind-sdk(SandboxClient + WebSocket terminal 协议)
```

- **上层 skill** 只描述"何时用、按什么顺序、如何纠错",不含平台细节
- **Profile 模板** 是垂直场景的唯一载体:新场景 = 新增一个 profile json,
  **不新增 skill**;控制流全部冻结在 runner 里,profile 只声明数据契约与步骤
- **底层工具** 负责认证、SDK 调用、WebSocket 建连,对 skill 屏蔽实现细节
- 任意其他 skill 也可以直接依赖 Sandbox 工具组合(不限于本 skill)

## Profile 模板体系

### 扩展模型(通用 skill + N 个模板)

| 层 | 职责 | 变更频率 |
| --- | --- | --- |
| SKILL.md | 路由边界、场景特征识别、模式 A/B 入口 | 低 |
| profiles/*.json | 单个场景的数据契约(manifest 字段)、步骤链、判定规则 | 中(每场景一份) |
| scripts/sandbox_runner.py | 冻结控制流,所有 profile 共用 | 低(改错会污染数据,需单测) |

当前 seed:`profiles/tmax-validation.json`,8 步 pi 链路(create → probe →
write 题面 → install_pi → run_pi → write test_sh → exec → delete),
verdict `kind=reward_file`(优先读 `/logs/verifier/reward.txt`,回退退出码)。

新增场景的唯一扩展方式是新增 profile;发现 runner 的环境适配缺陷时
(镜像缺依赖、exec 通道超时等)应当**回填冻结运行时**(修 runner/profile),
不要降级模式 B 绕开——否则断点续跑/强制清理/标准 verdicts 全部失效。

### 执行模式

- **模式 A(profile 匹配)**:数据形态含 `image` + 题面 + verifier → 匹配
  profile → **默认 `edp_submit` 提交平台任务**(CustomCommandCPUNode 节点跑
  runner:先 smoke `limit=3` 反馈结果,用户确认后同 `run_id` 全量续跑;
  断点续跑/抽样渐进/强制清理全部在节点内 runner 生效);本地
  `sandbox_runner.py` 命令降级为调试/兜底通道
- **模式 B(无匹配 profile)**:对话组装沙箱工具(一次性/探索性任务);
  判定口径与批次策略见 references/batch-orchestration.md

## 工具契约

### 认证链路

1. 会话 `secret_registry` 中取 `auth_token`(键名 `PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET`)
2. `create_sandbox_api_client(env, cluster, auth_token, headers)`:
   `get_api_key` 换 accessKey → `get_pyromind_api_client` 建 SDK 客户端
3. REST 调用走 portal 域名;WebSocket(sandbox_terminal)走 per-cluster
   direct domain —— **portal 不代理 WebSocket**

### 工具矩阵

| 工具 | 用途 | 关键参数 |
| --- | --- | --- |
| sandbox_create | 启动沙箱并等待 running | sandbox_type(osworld/custom), image 或 image_path, cpu, memory, wait_timeout, volume_mounts, port_mappings |
| sandbox_terminal | 沙箱内执行 shell 命令(WebSocket 流) | sandbox_id, command, cwd, timeout_seconds |
| sandbox_upload | 沙箱 → storage 回传(挂载点优先,base64 兜底) | sandbox_id, sandbox_path, storage_path |
| sandbox_delete | 删除沙箱 | sandbox_id |
| sandbox_write_file | 沙箱内写入文件 | sandbox_id, path, content |
| sandbox_read_file | 沙箱内读取文件 | sandbox_id, path |
| sandbox_download | storage → 沙箱下载 | sandbox_id, storage_path, sandbox_path |

### 两种沙箱类型

| sandbox_type | 运行方式 | 镜像参数 | 适用场景 |
| --- | --- | --- | --- |
| osworld(默认) | GUI 系统镜像(qcow2,VNC 访问) | image_path(storage 路径) | 需要桌面/系统级环境,web_vnc_url 可人工介入 |
| custom | 无头 Docker 容器 | image(镜像引用,必填) | 特定工具链/运行时(如 JDK8),可配 volume_mounts/port_mappings |

custom 的挂载/端口条目格式(docker -v / -p 风格):
- volume_mounts: `{host_path, mount_path, read_only}`(host_path/mount_path 必填)
- port_mappings: `{container_port, host_port, protocol}`(container_port 必填)

## 与现有数据处理能力的边界

判断口诀:任务是否依赖特定环境?**依赖 → 沙箱;不依赖 → 平台处理**。
是成批数据的逐条验证 → **本 skill 模式 A**;是单个容器的临时操作 → sandbox;
无环境要求的格式转换/清洗 → data-cleaning / data-preparation。
