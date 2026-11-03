# 模型推理评测提交

调用 `df_submit_evaluation`：

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
下载源还支持 modelscope，省略时默认 huggingface。model_source 不放在工具入参里。

max_model_len 可按模型约束指定。平台实时校验资源；不支持时修正明确的配置，不切换节点类型。
served_model_name 来自服务契约，默认 default，不从 checkpoint 路径猜测。

工具先生成 workflow.py，再校验、冻结并提交。storage 使用 Inference → CPU；
clone 使用 CloneAndCacheModel → Inference → CPU；
download 使用 DownloadAndCacheModel → Inference → CPU。
准备节点的 model_path 输出连接 Inference.model_path。
服务端连接 VLLMInference.endpoint 与 CustomCommandCPUNode.param，CPU 通过
直接的小写 `$param` 请求内部服务。CPU 不启动或安装 vLLM，不需要外部部署 API Key。
生成的 DSL 同步到画布，提交的是同一份规范转换得到的工作流。

输出包括 progress.json、processed.jsonl、report.json、predictions.jsonl、metrics.json 和
独立 evaluation_report.html。request_count 是累计请求尝试数，current_request_count 是本次尝试数，
reused_predictions 是复用的 prediction 数。业务不通过与请求/评分异常分别统计。

同配置恢复：`operation="resume"`、`resume_run_id="原 run_id"`、`input_path="原路径"`。
恢复不传模型、配置或任务参数；省略 cpu、memory 继承原资源配置。
新运行有独立 output_dir；来源、模型或评测配置变化不能混用已有结果。
