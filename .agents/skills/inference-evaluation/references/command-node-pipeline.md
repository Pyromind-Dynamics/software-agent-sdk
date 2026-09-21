# 固定两节点提交

```json
{
  "input_path": "/datasets/demo/eval.jsonl",
  "inference": {
    "model_path": "/models/immutable-checkpoint",
    "served_model_name": "default",
    "dataset_config_path": "public_data/inference_dataset_config.json",
    "evaluation_config_path": "public_data/inference_evaluation_config.json",
    "gpu_product": "NVIDIA-L40S",
    "gpu_count": 1,
    "port": 3000
  },
  "cpu": 4,
  "memory": 32,
  "mode": "full"
}
```

max_model_len 可按模型约束指定。平台实时校验资源；不支持时修正明确的配置，不切换节点类型。
served_model_name 来自服务契约，默认 default，不从 checkpoint 路径猜测。

服务端固定连接 VLLMInference.endpoint 与 CustomCommandCPUNode.param，CPU 通过
直接的小写 `$param` 请求内部服务。CPU 不启动或安装 vLLM，不需要外部部署 API Key。
生成的 DSL 同步到画布，提交的是同一份规范转换得到的工作流。

输出包括 progress.json、processed.jsonl、report.json、predictions.jsonl、metrics.json 和
独立 evaluation_report.html。request_count 是累计请求尝试数，current_request_count 是本次尝试数，
reused_predictions 是复用的 prediction 数。业务不通过与请求/评分异常分别统计。

同配置恢复：`mode="resume"`、`resume_run_id="原 run_id"`、`input_path="原路径"`。
省略 inference、cpu、memory 可复用原配置。新运行有独立 output_dir；不同配置不可混用结果。
