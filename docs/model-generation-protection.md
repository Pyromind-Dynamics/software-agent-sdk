# Pi 重复生成保护

检测与中断复用 [pi-no-spin](https://github.com/wilhg/pi-no-spin) 1.0.2 的原始扩展工厂，
不复制或修改检测算法。Pi SDK 保持 0.87.0；使用显式依赖 jiti 2.7.0 加载插件发布的
TypeScript，支持编译后的 Node 服务。

## 行为和边界

- 正文和 thinking 共用插件默认阈值：连续重复 7 次，片段长度 4–300 字符。
- Adapter 包装插件的 API，保留检测及 abort，关闭自动 follow-up 和 TUI 通知。
  原生插件配置将提醒冷却期设为 0，让立即发起的手动重试也受到保护。
- 主会话失败码为 `model_repetition_detected`，通过已有执行策略事件暂停自动轮次。
  通知保留在队列中，用户发新消息后恢复；排队的 steering 不会自行启动下一次请求。
- 子 Agent 通过 required-child-extension 强制加载相同适配，失败仅结束子任务，
  取消经宿主转发的模型请求；父会话继续使用已有子任务失败处理流程。
- 使用原生 `turn_end` 边界提交 `context_edit`，替换异常正文和 thinking。
  工具调用、此前工具结果和原始消息记录保留；当前投影与重启恢复读取相同编辑结果。
- 只有 thinking 或空白的最终回复返回 `model_empty_response`。主动取消仍为 cancelled。
- 精确重复以外的语义相似、模型采样设置、模型输出上限、前端设置界面均不在此变更内。

## 架构与兼容性

插件、配置和恢复适配位于 `harness-adapter`。Product Runtime 仅允许 Harness 发出已有的
通用 `execution.policy.updated` 事件；前端、公开 API 和 Product 持久化模型不新增字段。
Runner 到 Adapter 的内部事件类型增加该通用执行策略事件；Adapter 同步更新 attach
recovery，再交给已有 Product 投影与存储链路。依赖方向保持 Adapter → Runtime。

## 验证

- TypeScript 检查和编译通过。集成测试覆盖正文、thinking、请求取消、单次失败上报、
  手动继续、重启恢复、保留工具证据、正常长报告/代码/JSON/分隔线，以及子 Agent。
- 用独立的 `node dist/index.js` 进程验证插件加载，stdout 全部为合法 JSONL。
  `npm test` 的 pretest 先构建，避免使用过期编译产物。
- 将会话 `0e9c756a` 的原始消息按每 40 字符更新回放到实际插件：第 169 行的
  384,890 字符输出在 62,360 字符时中断；第 176 行的 201,498 字符输出在 8,680
  字符时中断。两次均只报告一次故障，不发送 follow-up。原会话文件未修改。
- 全量 Pi Runtime 测试：202 通过、2 失败；Product Runtime：564 通过、4 跳过、1 失败。
  三项失败均在临时 HEAD 基线复现：AgentGenome 测试缺少预期的 `list` 调用，
  sandbox timeout 测试未包含 `queueTimeoutMs: undefined`，以及旧提示词字符串断言。
  本次未修改这些无关断言或其业务实现。

运行：在 `harness-adapter/pi-runtime` 执行 `npm run check` 和 `npm test`；在仓库根目录
执行 `uv run pytest tests/pyromind_runtime`。架构隔离和通用事件投影也包含在后者中。
