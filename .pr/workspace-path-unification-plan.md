# Workspace 路径统一工作规划

## 当前进度（2026-11-09）

本地实现已覆盖 Storage 路径转换、Pi 沙箱默认挂载、Storage 工具的输入输出、工作流持久化输出校验，以及 Skill/Knowledge 路径说明。新建 Pi 沙箱默认挂载到 `/workspace`，布局版本升到 5；旧沙箱会在下次准备时重建，Storage 文件不变。

真实平台验证仍待完成：需在测试环境确认 Sandbox API 接受 `/workspace` 挂载，并跑通 Storage 上传、沙箱读写、平台节点读写、暂停恢复和重建。当前只修改本地代码，尚未部署或验证平台侧行为；这一步完成前不能视为发布验收通过。

## 目标

用户、前端、Agent、Pi 会话沙箱和平台工作流统一使用
`/workspace/...`。Storage API 内部继续使用去掉挂载前缀的 `/...`。Pi
会话工作区位于 `/workspace/.pyromind-agent/<conversation_id>`。

`/target-workspace` 不再作为新会话的挂载点，也不再出现在模型提示词、工具说明、
Skill 或工具返回结果中。旧会话和历史轨迹中的该前缀仅在输入边界兼容。

平台批处理 Pod 的内部挂载契约不在本次改动中盲目替换；若平台仍要求
`/target-workspace`，转换留在工具内部，不能暴露给模型。

## 路径契约

| 使用位置 | 标准路径 |
|---|---|
| 用户、前端、Agent、Pi 沙箱、工作流节点 | `/workspace/<path>` |
| Storage API 和工具内部持久化 | `/<path>` |
| Pi 会话文件 | `/workspace/.pyromind-agent/<conversation_id>/...` |
| 模型使用的会话相对路径 | `public_data/...` |

模型输入 `/workspace/models/a` 时，Storage 工具转换为 `/models/a`；工具向模型返回
Storage 路径时转换回 `/workspace/models/a`。

## 实施阶段

### 1. 验证挂载能力

- 验证 Sandbox API 接受 Storage 从 `/workspace` 挂载到容器 `/workspace`。
- 验证镜像不依赖挂载前 `/workspace` 中的文件。
- 验证文件 API、terminal、暂停恢复和容器重建均能访问同一份 Storage。
- 验证写入 `/workspace` 的文件能在 Storage 前端看到。

真实平台探测是切换生产默认值前的发布门槛；本地实现先由挂载请求、路径策略和迁移测试
保证契约正确。

### 2. 统一 Storage 路径转换

- 集中处理 `/workspace/<path>` 与 Storage API `/<path>` 的双向转换。
- 工具内部继续使用 Storage 逻辑路径，模型可见结果统一为 `/workspace/...`。
- 旧 `/target-workspace/...` 只作为兼容输入，不能再次输出。
- 禁止各工具自行重复拼接 Agent 可见路径前缀。

### 3. 切换 Pi 沙箱布局

- Pi Storage 默认挂载点从 `/target-workspace` 改为 `/workspace`。
- 提升工作区布局版本，使旧挂载的沙箱自动重建。
- 会话目录、资源缓存、fork、工作流快照和预热路径随挂载根迁移。
- 保留挂载环境变量作为短期回滚开关。

### 4. 修改 Pi 文件工具与终端

- `read`、`write`、`edit` 的 Storage 根改为 `/workspace`。
- 保留跨会话、内部状态、路径穿越等现有保护。
- terminal 直接使用真实 `/workspace`，不解析或替换 Shell 命令。
- Tool Schema 只说明 `/workspace/...` 和 `public_data/...` 两种模型可见路径。

### 5. 统一 Python 工具边界

- 审计所有 Storage 路径输入、输出和 Observation 文本。
- 输入统一规范化为 Storage API 路径，输出统一转换为 `/workspace/...`。
- 下游平台 Pod 若仍使用其他挂载点，只能在命令构建阶段内部转换。
- 工作流提交复用统一转换，不再单独解析 Pi 挂载前缀。

### 6. 防止训练产物未持久化

- 训练和数据处理节点的持久化输出使用 `/workspace/...`。
- 对已知持久化输出字段增加提交前检查。
- `/models/...` 等容器根路径不能作为 Storage 输出静默提交。
- 不对 `/tmp` 等可能有意使用的临时路径统一添加前缀。

### 7. 兼容 AgentGenome 和旧会话

- 新轨迹只记录 `/workspace/...`。
- 旧轨迹注入前将旧挂载前缀转换成 `/workspace`。
- 不改写历史证据文件；兼容发生在读取和执行边界。
- 旧沙箱记录因布局不一致自动重建，Storage 中的会话文件保持不变。

### 8. 更新模型可读文档

- System Prompt 只说明 Storage 位于 `/workspace`。
- Tool description、Skill 和 Knowledge 不再要求模型区分两种挂载路径。
- 训练节点文档明确：需要持久化的输入和输出必须位于 `/workspace`。
- 内部开发文档可以记录转换关系，但不能把内部路径重新暴露给模型。

## 测试与验收

单元测试覆盖默认挂载、旧沙箱重建、路径权限、双向转换、旧路径兼容、工作流提交、
AgentGenome 和训练输出保护。集成测试完成以下闭环：

```text
Storage 上传 → Pi read/terminal → Pi write → 平台节点读取和写出 → Pi 再次读取
```

还需验证服务重启、沙箱暂停恢复、沙箱重建、旧会话恢复和训练 checkpoint 持久化。

完成标准：新会话全链路只使用 `/workspace/...`；模型不再接触
`/target-workspace`；所有 Storage 工具的输入输出一致；旧路径有受控兼容；训练产物不会
再因写入未挂载的 `/models/...` 而丢失。

## 提交拆分

1. 统一 Storage 路径转换。
2. 将 Pi Storage 挂载切换到 `/workspace`。
3. 统一工作流和业务工具路径。
4. 更新提示词、Tool description、Skill 和 Knowledge。

每个提交只包含本任务文件，不带入当前工作区已有的 Langfuse、Pi vigilant 或启动脚本
改动。
