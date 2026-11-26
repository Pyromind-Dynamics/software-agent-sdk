# 模型推理评测提交

按 [SDK 执行与回调](../../workflow-execution/references/sdk-execution.md) 使用当前会话的
SDK 身份和关联上下文。SDK 解释器已安装轻量准备函数：业务参数用
`pyromind_common.platform.evaluation_core.EvaluationOptions.model_validate(...)` 校验，
再传给 `pyromind_common.platform.evaluation.stage_evaluation(...)`。
以下为 EvaluationOptions 的业务参数示例，不是 `studio.create()` 的直接参数：

```json
{
  "operation": "inference",
  "input_path": "/datasets/demo/eval.jsonl",
  "model_path": "/models/immutable-checkpoint",
  "served_model_name": "default",
  "dataset_config_path": "public_data/inference_dataset_config.json",
  "evaluation_config_path": "public_data/inference_evaluation_config.json",
  "gpu_product": "NVIDIA-L40S",
  "gpu_count": 1,
  "port": 3000,
  "cpu": 4,
  "memory": 32
}
```

上例省略 model_source，默认 Storage 模型。评测配置 JSON 仍含 mode、rubrics 等评价规则：
添加 `"model_source":{"type":"clone"}` 时，model_path 填中心仓模型标识；
添加 `"model_source":{"type":"download","download_source":"huggingface"}` 时填外部仓模型名称。
下载源还支持 modelscope，省略时默认 huggingface。model_source 只写入评测配置 JSON。

max_model_len 可按模型约束指定。平台实时校验资源；不支持时修正明确的配置，不切换节点类型。
served_model_name 来自服务契约，默认 default，不从 checkpoint 路径猜测。

准备函数接收 `files=LocalFiles(root)`、`staging=StagingDirectory(local, storage)`、
本次 UUID `run_id` 和本 Skill 的 `scripts` 目录 `runtime_dir`，可选 `frozen_reward`、`selection`。
`stage_evaluation` 不发 HTTP、不提交平台任务；返回的 `StagedEvaluation` 包含
`name`、`dsl`、`workflow_path`、`frozen_files` 和 `requires_inference`。

将 `staging.local` 放在当前工作区内，把 `staged.workflow_path` 转为相对工作区路径后调用
`validate_workflow_dsl(dsl_path=该相对路径, name=staged.name)`。
SDK scope 校验成功时，Host 将实际图安全保存到
`public_data/workflow-validation/<sha256>.json`，工具 content 返回
`validated_workflow_path` 和 `sha256`。直接使用返回的路径读取文件，不猜测路径，
不从模型不可见的内部 details 提取图：

```python
import json
from pathlib import Path

workflow = json.loads(Path(validated_workflow_path).read_text(encoding="utf-8"))
```

将该 `workflow` 交给 `pyromind_common.platform.workflows.submit_workflow`。
若返回 `artifact_unavailable` 或没有实际图文件，即使 valid=true 也停止 SDK 续投。
凭据、配置密钥、签名 URL 导致的安全拒存和 IO 失败都不能通过重构或重写图绕过。
发布冻结文件及 Project 元数据更新由调用方完成，不调用旧评测提交 Tool。
SDK 提交该实际图时使用上下文提供的
out_id，并设置 `max_retries=0`；一次成功的 create 之后分别保存平台 task_id 和业务产物位置。
超时、断连或响应无法解析且没有 task_id 时，提交结果待确认，不能自动重投。
storage 使用 Inference → CPU；
clone 使用 CloneAndCacheModel → Inference → CPU；
download 使用 DownloadAndCacheModel → Inference → CPU。
准备节点的 model_path 输出连接 Inference.model_path。
服务端连接 Inference.endpoint 与 CustomCommandCPUNode.param，CPU 通过
直接的小写 `$param` 请求内部服务。CPU 不启动或安装 vLLM，不需要外部部署 API Key。
需要同步画布时使用同一份冻结 DSL，提交的是校验 Tool 保存到返回路径的实际图。

`prepare_evaluation(..., validator=...)` 是本来就持有独立校验登录身份的调用方可用的
便捷包装，需要已鉴权的 `httpx.Client`；SDK Key 不能被假定为旧校验 Cookie 的替代身份。
SDK terminal 默认使用上述 stage → 保留校验 Tool → SDK 创建路径。

输出包括 progress.json、processed.jsonl、report.json、predictions.jsonl、metrics.json 和
独立 evaluation_report.html。request_count 是累计请求尝试数，current_request_count 是本次尝试数，
reused_predictions 是复用的 prediction 数。业务不通过与请求/评分异常分别统计。

同配置恢复：`operation="resume"`、`resume_run_id="原 run_id"`、`input_path="原路径"`。
恢复不传模型、配置或任务参数；省略 cpu、memory 继承原资源配置。
新运行有独立 output_dir；来源、模型或评测配置变化不能混用已有结果。

## 验证集选模后的独立测试

使用 `operation="inference"`、原选模批次 `selection_run_id` 和准确的 checkpoint-n。
从已有 Project 版本、训练产物和选模报告取得已授权证据，构造
`pyromind_common.platform.evaluation_core.ValidationSelection`，传给
`stage_evaluation(..., selection=selection)`：

| 字段 | 冻结证据 |
|---|---|
| `selection_run_id` | 验证集选模批次 ID，与 EvaluationOptions 中相同 |
| `train`、`validation`、`test` | `FrozenDataset(path=..., sha256=...)`；真实 Storage 路径及原始字节 SHA256 |
| `model` | `ValidationCheckpoint(path=..., kind="full" 或 "lora", base_model=...)`；准确 checkpoint-n，LoRA 必须有被冻结的 Storage 基模路径 |
| `frozen_reward` | 采用冻结 Reward 时填验证集使用的同一快照，并与准备函数的 `frozen_reward` 参数一致 |
| `scoring_script`、`scoring_spec` | 不使用冻结 Reward 时提供原验证批次的评分代码正文和规格对象，不是新脚本路径 |

准备函数核对 test 与 train/validation 的路径和摘要不同，核对选中的模型、基模、Reward
及完整测试范围；最终 test 不设置 `generation.limit`。这些检查不替代 Project 授权，
不能从自报 task_id 推导证据归属，也不能证明不存在近重复或业务泄漏。
选择证据冻结到 staging 的 validation-selection.json；恢复从该文件读取，不重新选模。
Project 记录实际平台任务、父 checkpoint 和已验收报告；平台成功不等于业务验收通过。
