# AVI 单轮 Graph 模板

用户确认计划 Revision 时，平台按 `avi-single-turn/v1` 模板写入本地 Graph。

## 拓扑

```text
dataset -> inference -> evaluator -> train
```

四个阶段 ID 与计划一致，四个节点都属于同一个 Project。Project ID 来自计划 Revision；
Graph ID 由服务端生成。客户端和 Agent 不能提交自造服务端 ID。

## 节点配置

每个节点的 `config` 从对应阶段的 `resources` 和 `tasks` 生成。`resources` 保存已确认的
外部输入及其 URI，`unresolved` 保存仍待补充的信息；平台不读取第二份 requirements
文件，也不维护重复的 AVI 配置事实源。

- `dataset` 绑定 `data-processing` 能力，采用 Agent 开发模式。真实 Schema、图片角色、
  标签语义和划分必须由开发态检查 Storage 数据后确认。
- `inference` 声明模型和阶段任务，输入来自 Dataset 阶段产物。
- `evaluator` 声明评测任务，标准产物格式为 `atif`。
- `train` 声明 SFT/DPO 训练任务和预期产物。

后续计划 Revision 确认时，平台更新同一个 Project 的 Graph。Project ID 和 Graph ID
保持稳定，不创建第二个 Project。

## Contract Probe

首期每个节点带：

```json
{
  "status": "idle",
  "executor": "contract_probe"
}
```

阶段启动接口会创建一次持久化 Run，并产生 `stage_contract_probe` Artifact 验证执行链路。
它不执行真实数据、推理、评估或训练，也不产生业务产物。后续接入真实执行器时保持节点
ID、边、Run 和 Artifact 契约不变，只替换 executor 与具体执行逻辑。

## 参考来源

- AVI demo 的业务阶段：Dataset、Train、Eval、Inference。
- trajectory_viewer 的 ATIF 展示：图片角色、GT/prediction 对照、badcase 和 reward。
- 当前首期收敛为单轮 `Dataset -> Inference -> Evaluator -> Train`，不包含在线 feeder、
  Kafka、checkpoint 轮询和热切换。
