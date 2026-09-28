# software-agent-sdk 架构隔离规范

## 核心原则

`harness-adapter` 是唯一的 Harness 分叉点。自 Adapter 往上，必须共用同一套 Runtime、Server、API、协议和前端。

```text
前端 / API / Product Runtime（唯一、共用）
                    |
              HarnessAdapter
               /           \
      OpenHands Adapter    其他 Adapter
```

## 强制约束

- Harness 差异只能存在于 `harness-adapter` 及其下层实现。
- Adapter 上层不得出现 OpenHands、Pi 等 Harness 专属类型、事件、接口或条件分支。
- 新增 Harness 只能实现统一的 `HarnessAdapter`，不得复制或另建 Runtime、Server、API 和前端流程。
- 产品命令、事件、快照、状态恢复和业务编排必须使用统一模型；原生 Harness 事件由 Adapter 转换后才能向上传递。
- Adapter 上层需要新能力时，应扩展通用协议或 capabilities；不得为单个 Harness 打补丁。
- 替换 Harness 后，对外 API 与前端交互语义必须保持不变。

## 依赖方向

```text
pyromind-agent-server -> pyromind-runtime
pyromind-agent-server -> harness-adapter -> pyromind-runtime
具体 Adapter -> 对应 Harness
```

- 依赖只能沿箭头方向，下层不得反向依赖上层。
- `pyromind-runtime` 不得导入 FastAPI、`harness-adapter`、OpenHands、Pi 或服务端代码。
- `harness-adapter` 不得依赖 `pyromind-agent-server`。
- `openhands-sdk` 保持平台无关，不得加入 Pyromind 产品语义或依赖产品层。

## 职责归属

- 产品命令、事件、快照和能力声明放在 `pyromind-runtime/domain`。
- 用例编排放在 `pyromind-runtime/application`，只能通过 `ports` 访问外部能力。
- `pyromind-agent-server` 只负责 HTTP、鉴权、配置和依赖装配；路由必须调用 `ConversationRuntime`，不得直接操作 Harness 会话。
- Harness 原生事件必须由 Adapter 转换为统一事件，禁止直接暴露给 API 或前端。
- 持久化必须通过 Product Store / External Task Registry 端口；Cookie、Token、Authorization 等请求凭据不得落盘。
- 平台工具可放在 `openhands-tools`，但业务流程、任务状态和恢复策略必须留在 Product Runtime。
- 旧实现兼容代码必须封装在 Adapter 或独立兼容模块中，不得污染统一协议。

## 稳定边界

- 前端只依赖 `ProductCommand`、`ProductEvent`、`ConversationSnapshot` 和 capabilities。
- 公开协议或持久化模型变更必须兼容旧客户端与旧数据；不兼容变更需要版本升级和迁移测试。
- 新能力优先扩展通用协议、端口或 capabilities，禁止跨层导入、共享可变全局状态或复制业务逻辑。
- 会话数据、工作区和执行状态必须按 `conversation_id` 隔离。

## 合并门槛

- 跨包改动必须说明所属层、依赖方向及 API / 持久化兼容影响。
- 涉及该边界的改动必须补充架构测试，并运行 `uv run pytest tests/pyromind_runtime`。
- 不得删除或放宽边界测试来迁就实现。
- 确需破例时，必须先提交简短设计说明并获得架构维护者明确同意；临时兼容必须标明移除条件。
