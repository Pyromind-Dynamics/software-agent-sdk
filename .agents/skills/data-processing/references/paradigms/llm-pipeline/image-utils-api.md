# image_utils API

`image_utils.py` 由 `df_run_pipeline` 和 `df_submit_pipeline` 自动投递。Pipeline
只能显式导入以下 API：

```python
from image_utils import ImagePipelineConfig, run_image_pipeline_from_cli
```

## ImagePipelineConfig

必填：

- `labeling_system_prompt`：发送给 VLM 的任务规则。
- `output_format="vision"`（默认）：生成训练 messages，必须填写
  `training_system_prompt`（system 消息 text block）。
- `output_format="structured"`：直接输出响应对象，运行时注入源样本 `id` 和源图关联
  `source_images`；不需要 `training_system_prompt`、`reasoning_key`、`answer_key`、
  `answer_is_json`。`response_json_schema` 必须描述对象且不包含保留字段 `id`、
  `source_images`；模型返回保留字段会触发重试。该模式不使用训练标签纠错配置
  `allow_reference_correction`。

常用字段映射：

- `id_key="id"`
- `images_key="images"`
- `image_labels_key="image_labels"`
- `user_prompt_key="user_prompt"`
- `sample_system_prompt_key=None`
- `user_prompt_template=None`

Prompt 优先读取样本的 `user_prompt_key`；为空时使用
`user_prompt_template.format_map(sample)`。模板引用缺失字段会立即失败。

响应配置：

- `response_json_schema`：传给 DataFlow VLM Serving 的严格 JSON Schema。
- `reasoning_key="reasoning"`
- `answer_key="answer"`
- `answer_is_json=False`

`structured` 使用原始 JSON 对象响应；不解析自然语言或 `<answer>`，不生成消息包装。
工具参数 `output_schema` 必须与 `output_format` 一致；图片模型仍用
`model_profile="vision"`。PCB 预打标默认采用 structured，参见其场景模板。

响应 Schema 按任务或下游契约配置，PCB 案例不构成固定字段或坐标校验要求。

已有人工标签的数据需要补 CoT 时，使用 vision 并配置：

- `metadata_filename`：可选 sidecar 文件名，例如 `meta.json`。
- `reference_label_path`：人工标签的 dotted path，例如 `metadata.label` 或
  `reference_annotations.label`。
- `reference_note_path`：可选人工备注路径。
- `reference_label_map`：原始标签到训练标签的显式映射。
- `allow_reference_correction=True`：默认保留人工标签；只有模型声明 `correct` 并提供
  非空纠错原因和具体视觉证据时才采用新标签。纠错审计不进入训练 JSONL。

共享运行时会把严格 Schema 传给 VLM；vision 模式兼容响应整体为单个
```` ```json ... ``` ```` 或 ```` ``` ... ``` ```` 代码块。Pipeline 不得自行增加
JSON fence 解析、响应修复或日志逻辑。

执行配置：

- `batch_size=8`：失败时整个未提交 batch 重跑；设为 1 可获得逐条恢复。
- `max_attempts=3`
- `max_workers=8`
- `timeout=1800`

模型配置统一来自 `DF_API_KEY`、`DF_API_BASE_URL` 和 `DF_MODEL_NAME`，不得写入
Pipeline。

## 执行入口

标准脚本结尾：

```python
if __name__ == "__main__":
    run_image_pipeline_from_cli(CONFIG)
```

该入口解析固定的 `input_path output_path [limit]` 参数，构建并运行 DataFlow
Pipeline。需要程序化调用时可使用：

```python
run_image_pipeline(CONFIG, input_path, output_path, limit)
```

`MultiImageSemanticLabelOperator` 是底层 DataFlow Operator，主要用于测试或受控扩展；
常规 Agent Pipeline 不直接实例化。

## 来源与恢复

`source_manifest.jsonl` 完整保留样本 ID、原始图片路径、角色和输入元数据。structured
要求源 ID 唯一，结果注入源 ID 与 `source_images`，不按输出行号关联（失败记录可能
跳过）。`source_images` 是角色到图片路径的映射，取自输入的 `image_labels`；平台运行时
写入 Storage 对象路径（`/datasets/...`），本地 Sample 运行时写该次运行可读的本地路径。
于是每一行都自带源图关联，后续 Label Studio 等消费方直接读行内字段，不需要再和
Manifest 做一次拼接；重复角色带 `#2`、`#3` 后缀。Manifest 仍保留输入侧完整路径基准，
模型预标注不改写人工标签。

`runtime_metadata.json` 冻结输出模式和 structured 响应 Schema。恢复时契约不一致
立即拒绝；旧运行缺少 output_format 时按 vision 处理。不同契约应创建新 run。
structured 不使用固定产物校验器；响应按任务配置的 Schema 在执行时处理。
