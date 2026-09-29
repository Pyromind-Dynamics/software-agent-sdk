# Inference 评测主线选择性合入

基线为 feature/pre-th9；仅提取 inference 的评测脚本与 rubric 思路，重新适配
df_submit_pipeline。没有执行分支合并或 cherry-pick。

## 调用

Agent 通过 preview_dataset 确认 JSONL 与媒体字段，写出工作区内的两个 JSON 配置，
然后在用户要求实际执行时调用：

```json
{
  "input_path": "/datasets/demo/eval.jsonl",
  "inference": {
    "model_path": "/models/immutable-checkpoint",
    "served_model_name": "default",
    "dataset_config_path": "public_data/inference_dataset_config.json",
    "evaluation_config_path": "public_data/inference_evaluation_config.json"
  },
  "mode": "full",
  "cpu": 4,
  "memory": 32
}
```

df_submit_pipeline 固定生成 VLLMInference.endpoint → CustomCommandCPUNode.param，
实时校验通过后上传内置脚本和配置，写入画布并正式提交。未传 inference 的旧单节点请求不变。
推理节点默认一张 NVIDIA-L40S、端口 3000；不支持的资源由平台校验返回错误。

## 层与依赖

- Product Runtime domain 定义 PipelineRun；application/PipelineRuns 管理冻结配置、
  版本竞争、活跃任务限制、跨会话隔离及同配置恢复。它只访问 ProductStore 和
  ExternalTaskRegistry 端口，不导入 Harness、FastAPI 或 OpenHands。
- FileProductStore 增加版本化 pipeline-runs.json，保存在本会话 product 目录；
  配置含模型引用、rubric、字段映射、生成参数、资源和脚本 SHA256，不保存请求凭据。
- 平台工具实现配置验证、白名单、固定 DSL 构建、上传、校验及提交。工具通过结构化
  Protocol 接受运行服务，不依赖产品 Server。
- Pi Adapter 注入运行服务和请求上下文，将任务转换为已有 data_preparation 事件。
  Server 仅装配依赖；回调沿用已有 Product Runtime 路径。
- 未改 OpenHands 专属路由、Adapter、SDK 或 run_workflow。唯一打包修改是将新增
  Pi Skill 资源加入共享二进制清单。

## 协议与恢复

新增可选 inference 工具参数和 df_check_progress.artifact_urls，不更改 REST、
ProductEvent、ConversationSnapshot 或任务 kind。旧 Product Store 没有 pipeline-runs.json
时按无新运行记录处理，原清洗任务仍走旧提交路径。

mode=resume、resume_run_id 和原 input_path 可复用冻结配置，省略 inference/cpu/memory
时继承原值。活跃或提交结果不确定的运行拒绝自动重复提交。

配置/资源/脚本变化由 Product Runtime 指纹拒绝；JSONL 和媒体内容的 SHA256 在平台
评测脚本的 run_manifest.json 中记录并在请求前核对。模型引用必须不可变，不哈希权重。
已有 prediction 只重新评分，不重复请求；缺失 prediction 才补发。
checkpoint 尾行不完整可恢复，完整损坏行会报错。请求 ledger 记录实际开始的尝试，
不表示服务端成功处理次数。

## 输出与交付

progress.json / processed.jsonl 与现有 df_check_progress 对齐。report.json 包含
执行状态、指标和 Storage artifact 路径；metrics.json、predictions.jsonl 与独立
evaluation_report.html 供详细分析。进度工具只为本运行目录中的 HTML 获取实时下载地址。
业务通过率为零仍可成功完成评测；零有效 prediction 或零完成评分不能生成成功报告。

## 验证与限制

- 本机 HTTP 模拟 endpoint 验证报告、请求计数与同配置恢复；另覆盖字段评分、bbox、
  必需项、部分请求失败、空输出、重复 ID、数据/媒体变更和截断 checkpoint。
- 平台 API 使用 mock 验证固定拓扑、资源校验失败不上传、不注入 debug 标志、
  画布与提交来源一致、Pi 事件、停止解析及不确定提交禁止重复。
- Product Runtime 的 170 项测试中 169 项通过。原有
  test_pyromind_start_scripts_use_composed_server_entrypoint 对 start_inference.sh
  的字符串断言失败；git show HEAD:start_inference.sh 已可复现，本次未改脚本或放宽测试。
- 尚未做真实平台 GPU 验收：需要明确模型 Storage 路径、小评测集路径、环境/集群和
  有效会话鉴权。验收需执行一次正式运行、一次中断续跑和一次整任务停止，记录 task_id、
  HTML 地址和请求次数。
