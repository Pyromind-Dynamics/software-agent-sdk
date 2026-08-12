# 框架说明:Sandbox 工具组合与 skill 分层

## 分层架构

```
environment-data-processing skill(SKILL.md 工作流编排)
        │ 依赖
        ▼
sandbox_create / sandbox_terminal / sandbox_upload / sandbox_delete
        │ 复用
        ▼
pyromind-sdk(SandboxClient + WebSocket terminal 协议)
```

- **上层 skill** 只描述"何时用、按什么顺序、如何纠错",不含平台细节
- **底层工具** 负责认证、SDK 调用、WebSocket 建连,对 skill 屏蔽实现细节
- 任意其他 skill 也可以直接依赖 Sandbox 工具组合(不限于本 skill)

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

- **data-cleaning / data-preparation**:无环境要求的平台数据处理,数据在
  storage,处理由平台执行
- **本 skill**:数据或代码需要"特定运行时"才能正确处理(JDK 版本、
  特定编译器、GUI 应用等),必须在对应镜像的沙箱内执行

判断口诀:任务是否依赖特定环境?**依赖 → 沙箱;不依赖 → 平台处理**。

## 关键实现细节(维护者须知)

- sandbox_terminal URL:`wss://<direct-domain>/api/v1/sandboxes/{id}/terminal?token=..&cols=120&rows=40`;
  发送 `command
` + `echo '<marker>' $?` 哨兵行判定命令结束,1s 收尾窗
- sandbox_upload 挂载点探测:`ls -d /target-workspace`;挂载存在则 `cp -r`,
  否则 `base64 < file` 回传后经 `upload_file` API 写入 storage
- sandbox_download:通过 storage `/get_url` API 获取预签名 URL,沙箱内用 curl 下载
- 环境变量注入:在 command 前加 `export K=V; ...` 前缀(sandbox_terminal 经 `/bin/sh -c`)
