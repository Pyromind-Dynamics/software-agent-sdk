# Inference Evaluation 输入契约

配置前建立以下语义映射。左侧是必须掌握的信息，右侧端口名必须以当前画布节点的
`nodeDefinition` 或平台校验结果为准，不能把示例名称当成固定 API。

| 信息 | 来源 | 配置原则 |
|---|---|---|
| 模型引用 | 当前画布或用户给出的 Storage 路径 | 通过模型路径节点输出连接推理节点 |
| 推理来源 | Storage 模型路径或已有 endpoint | 前者由 VLLMInference 提供 endpoint，后者由 CPU 命令节点直接调用 |
| endpoint model | 用户配置或推理服务契约 | 使用明确的 served model name；`VLLMInference` 未暴露名称时使用 `default`，不得从 checkpoint 目录名推断 |
| 测试集 | Storage 中的 JSON/JSONL 文件 | 使用现有路径节点输出或节点接受的路径字符串 |
| ground truth | 用户说明或 `preview_dataset` 的真实字段 | 显式映射，不能从字段名猜测 |
| 样本输入与媒体 | 数据集真实字段 | 按节点已有 dataset/mapping 输入表达 |
| Rubric 配置 | 当前 Agent 对测试集预览的分析和用户要求 | 写入 JSON 文件，运行时只读取、不生成 |
| 报告输出 | Storage 目录或 HTML 路径 | 使用 `/workspace/...` 平台路径 |
| 环境变量 | 节点 ENV 输入 | JSON string；Secret 只写 Secret 名或平台引用 |

## 字段映射键

运行脚本使用以下规范键；不要自行发明 `image_field`、`image_root`、`gt_field` 等别名：

| 键 | 作用 |
|---|---|
| `id_field` | 样本 ID 字段，可选 |
| `messages_field` | 已组装的 OpenAI messages；与 `user_prompt_field` 二选一 |
| `system_prompt_field` | system prompt 字段，可选 |
| `user_prompt_field` | user prompt 字段 |
| `media_field` | 图片或媒体路径字段，可为字符串或列表 |
| `media_base_dir` | 相对媒体路径的基目录，可选；默认相对 JSONL 所在目录 |
| `image_order` | 按文件名规定多图顺序，可选 |
| `reference_field` | ground truth 字段，必填 |

多模态任务必须写 `media_field`。媒体路径按下面的规则解析：

```text
JSON/JSONL 所在目录 + media_base_dir（仅在配置时）+ 样本中的媒体路径
```

- 样本已经写 `images/a.jpg`，且 `images/` 与 JSON/JSONL 同目录时，省略 `media_base_dir`。
- 只有媒体不在 JSON/JSONL 目录下时才设置 `media_base_dir`；相对值必须以 JSON/JSONL 目录为起点，
  绝对值必须是 `/workspace/...` 容器路径。
- 不得把 dataset path 中已有的目录再次写入相对 `media_base_dir`。例如 dataset path 是
  `/workspace/datasets/demo/eval.jsonl`、样本媒体是 `images/a.jpg` 时，不能设置
  `media_base_dir="datasets/demo"`，否则会解析成 `/workspace/datasets/demo/datasets/demo/images/a.jpg`。
- 写配置前必须用一个代表性媒体值计算并核对最终绝对路径；不能只验证字段名称。

### 媒体路径硬门禁

对每个 `media_field`，写入配置和上传前都必须满足：

1. 从样本读取一个真实媒体值，不根据字段名构造示例路径。
2. 当媒体值相对 JSON/JSONL 所在目录可直接访问时，配置对象中不得出现 `media_base_dir` 键；`null`、
   空字符串和数据集目录自身都不合格。
3. 解析后的绝对路径必须通过预览确认存在。若路径中出现类似
   `.../datasets/demo/datasets/demo/...` 的重复目录，视为配置错误并停止上传。
4. 修改配置后必须重新预检并重新上传；DSL 只能引用当前会话上传工具返回的新路径，不能沿用旧会话
   的配置文件，即使文件名相同。

发生 `FileNotFoundError` 且路径重复数据集目录时，直接删除错误的 `media_base_dir` 并重新上传配置；
不要修改 `dataset_path`、样本中的媒体值或 CPU 节点 command 来掩盖路径错误。

配置完成后先运行 `validate_pipeline.py --configs-only`；通过后才能暂存和上传执行文件。

## Agent Rubric 默认语义

用户未指定细节且节点支持对应配置时，可采用以下保守默认值：

```json
{
  "mode": "agent_rubric",
  "pass_threshold": 0.8,
  "rubric_pass_threshold": 0.7,
  "rubrics": []
}
```

空 `rubrics` 只是结构示意，不能交付。当前 Agent 必须在预览测试集后写入 2～5 条任务相关规则；
每条包含名称、标准、权重和确定性 evaluator。核心正确性或关键定位项应设置 `required: true`。
运行时只执行这些规则，并输出每项 0～1 分数、证据、失败原因和加权总分。详细格式见
`rubric-authoring.md`。

## 最小必要信息

以下任一信息无法从当前画布或数据预览中得到时，向用户询问一个合并后的短问题：

- 模型 Storage 路径
- 测试集 Storage 路径
- ground truth 字段
- Storage 模型场景存在 `VLLMInference` 和 `CustomCommandCPUNode`；已有 endpoint 场景存在 CPU 节点

GPU 型号、样本上限、输出目录或阈值未指定时，可以保留画布已有值；画布也没有时使用平台
契约允许的保守默认值，并在交付中说明。
