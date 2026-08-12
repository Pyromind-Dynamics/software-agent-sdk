---
name: environment-data-processing
description: >
  特定环境数据处理:当任务对运行环境有硬性要求(如筛选 Java 1.8 特性代码
  需要 JDK 8 环境)时,通过 Pyromind 沙箱工具(sandbox_create/sandbox_terminal/
  sandbox_upload/sandbox_delete)启动对应镜像的沙箱,安装
  环境与 codex,在沙箱内执行数据处理,回传产物并强制清理沙箱。适用于
  "需要特定运行时/工具链的数据处理任务";无环境要求的平台数据处理请用
  data-cleaning / data-preparation。
license: MIT
---

# 特定环境数据处理(Environment Data Processing)

## 适用边界

| 场景 | 工具 |
| --- | --- |
| 仅格式转换/字段映射/简单过滤,无环境要求 | data-cleaning |
| 抽样/清洗/生成/评分,平台 DataFlow 处理 | data-preparation |
| **任务对运行环境有硬性要求(JDK 版本、特定工具链、GUI 等)** | **本 skill  (sandbox_create + sandbox_terminal + sandbox_upload + sandbox_delete)** |

典型场景:筛选仓库中仅使用 Java 1.8 特性的代码 —— 需要 JDK 8 环境才能
准确编译/解析验证;这类任务必须在对应镜像的沙箱中执行。

## 前置条件

- 目标环境可用镜像已就绪,二选一:
  - **系统镜像(qcow2)**:已上传到 storage,路径形如 `templates/java8.qcow2`
  - **容器镜像(Docker/OCI)**:可直接拉取的镜像引用,如 `eclipse-temurin:8-jdk`
- 当前会话已配置 Pyromind 认证(secret `auth_token` + env/cluster)

## 工作流(严格按序执行)

### 1. 明确需求与环境要求

- 确定任务需要的运行时(JDK 版本、Python 版本、GPU 等)
- 确认 storage 中镜像路径;不确定时先用 `preview_remote_dataset` / 询问
  用户确认镜像路径,不要猜测

### 2. 启动沙箱

```json
{"operation": "create", "sandbox_type": "osworld", "image_path": "templates/java8.qcow2", "cpu": 4, "memory": 8, "wait_timeout": 300}
```

需要无头容器环境时用 custom 类型(Docker 镜像,可带挂载/端口):

```json
{"operation": "create", "sandbox_type": "custom", "image": "eclipse-temurin:8-jdk", "volume_mounts": [{"host_path": "/workspace", "mount_path": "/data"}], "cpu": 4, "memory": 8, "wait_timeout": 300}
```

- 记录返回的 `sandbox_id`(后续所有操作都要用到)
- 若返回失败,修正参数后重试,不要继续后续步骤

### 3. 安装环境与工具(sync 模式)

```json
{"operation": "exec", "sandbox_id": "<id>", "mode": "sync", "command": "java -version && npm install -g @openai/codex", "timeout": 120}
```

- 先探测环境: `java -version` / `python3 --version` / `node --version`
- 缺少依赖时用 `npm install -g` / `apt-get install` 安装(网络在沙箱内可用)
- 短命令一律用 `mode="sync"`

### 4. 编写处理脚本 / codex 任务命令

- 复杂处理建议先用 `mode="sync"` 把处理脚本写入沙箱
  (`cat > /tmp/process.sh <<'EOF' ... EOF`)
- codex 任务参考 `references/codex-usage.md`;首实例参考
  `references/java-18-code-filtering.md`

### 5. 实时执行(terminal 模式,WebSocket 流)

```json
{"operation": "exec", "sandbox_id": "<id>", "mode": "terminal", "command": "bash /tmp/process.sh", "timeout": 600}
```

- 长任务用 `mode="terminal"` 获得秒级输出反馈
- 超时未完成时,可改为后台 + 轮询日志:
  1. `mode="background"` 启动,记下 `log_path`
  2. `mode="terminal"` 执行 `tail -f <log_path>` 观察进度

### 6. 回传产物(upload)

```json
{"operation": "upload", "sandbox_id": "<id>", "sandbox_path": "/tmp/result.json", "storage_path": "datasets/java8-filtered"}
```

- `storage_path` 为 storage 目标目录(可省略,默认
  `/.pyromind-agent/<conversation_id>/uploads`)

### 7. 清理沙箱(硬约束)

```json
{"operation": "delete", "sandbox_id": "<id>"}
```

> **无论任务成功还是失败,最后一步必须 delete 沙箱**,避免资源泄漏。
> 删除失败时带上 `"force": true` 重试一次;确实无法删除时必须明确告知用户。

## 注意事项

- 每一步都检查上一步返回的 `status`/`returncode`,失败即修正,不要盲进
- `exec` 需要 `sandbox_id`,`create` 失败后没有沙箱可清理
- 沙箱内文件路径以沙箱视角为准;storage 路径与 `upload` 的目标目录分离
- 若需 VNC 图形交互,用 create 返回的 `web_vnc_url`(供用户人工介入)
