# AVI 单轮 Graph 模板

Project 创建时，平台按 `avi-single-turn/v1` 模板写入本地 Graph。

## 拓扑

```text
dataset -> inference -> evaluator -> train
```

四个节点都属于同一个 Project。节点 ID 和 Graph ID 由服务端生成；客户端和 Agent
不能提交自造 ID。

## 节点配置

### dataset

从 Project requirements 读取：

- `dataset_uri`
- `dataset_format`
- `image_roles`
- `label_schema`
- `split_policy`

### inference

- `model` 来自 `inference_model`
- 输入引用 `dataset.dataset`
- 输出引用 `inference.predictions`

### evaluator

- 输入引用 `inference.predictions`
- `rule` 来自 `evaluation_rule`
- `artifact_format=atif`
- 输出引用 `evaluator.report`

### train

- `method` 为 `sft` 或 `dpo`
- 输入引用 `dataset.dataset` 和 `evaluator.report`
- `artifact_goal` 来自同名 requirement
- DPO 时数据契约还必须提供 rejected answer

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
