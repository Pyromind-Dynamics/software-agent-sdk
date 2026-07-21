---
name: data-cleaning
description: >-
  在 Pyromind 平台探索、清洗、转换、修复并校验 Storage、共享空间或 Hugging Face 数据集，统一生成符合 Transformers chat message patterns 的 messages JSONL。适用于 CSV/JSON/JSONL、日志、对话与 Agent 轨迹、工具调用、偏好、多选题及脏生成数据；也适用于编写或复用确定性清洗脚本、试跑检查和全量执行。用户可能使用"清洗""clean""格式化""转换""修复数据""数据预处理"等表达。本 Skill 通过 run_dataset_cleaning 异步提交 Pyromind 任务，并在 Kafka callback 自动续跑会话后使用 preview_dataset 检查、修复或继续任务。
triggers:
- 清洗
- 数据清洗
- clean
- 格式化数据
- 转换数据
- 数据预处理
- 修复数据
- data cleaning
---

# 数据清洗

将源数据统一转换为符合 [Transformers chat message patterns](https://huggingface.co/docs/transformers/chat_content_patterns) 的 `messages` JSONL，用同一份确定性脚本在 Pyromind 平台完成试跑和全量清洗。

## 核心约束

- 目标格式固定为 `messages` JSONL，不询问用户选择其他输出格式。
- 不根据字段名猜测语义；字段映射、过滤、长度阈值或有损转换有歧义时，展示样本并确认关键决策。
- 不逐行调用 LLM。可以一次性生成并经用户确认全局常量，如系统提示词或字段映射，再将其固化到脚本中。
- 试跑与全量执行使用同一脚本和同一代码路径；`limit` 只限制本次读取的源记录数。
- 保留有意义的 Unicode、缩进、Markdown 换行、代码及工具调用关联；只清理控制字符和结构错误，不改写内容风格。
- 检查数据集说明、许可证及 `canary`、`do_not_train` 等显式元数据；排除禁止训练的基准数据。
- 不把 `chosen`/`rejected` 偏好数据静默压成 SFT；必须确认使用哪个分支生成 `messages`，否则停止转换。
- 不覆盖原始数据。无法可靠转换的记录写入 `errors.jsonl`；不要伪造缺失标签或静默丢弃。
- 清洗过程应流式、幂等、可恢复；单条失败不得中断整体，结构性错误必须以非零状态退出。

## 目标格式

`output.jsonl` 每行只包含一个 `messages` 对象：

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}
```

遵循以下规则：

- `messages` 是非空数组；每条消息包含 `role` 和 `content`，role 只使用 `system`、`user`、`assistant` 或 `tool`。
- 文本 `content` 使用字符串；多模态内容使用显式类型列表，如 `text`、`image`、`video`、`audio`，媒体使用 `url` 或 `path`。
- 工具调用放在 `assistant.tool_calls` 中，`type` 为 `function`；工具结果使用 `tool` role，且 `content` 必须是字符串。
- 保留工具调用 ID、工具名和结果之间的关联。
- 多轮对话按原顺序保留，不把多轮内容拼成单个字符串。
- 顶层不输出 `messages` 以外的字段；溯源、错误和统计信息写入对应的状态文件。

## 平台工具与执行边界

| 工具 | 用途 |
|---|---|
| `preview_dataset` | 浏览源数据；清洗完成后查看 Pyromind Storage 中的 `stats.json`、`errors.jsonl`、`output.jsonl` 和 `checkpoint.json` |
| `upload_file_to_pyromind` | 将生成的 `clean_script.py` 上传到 Pyromind Storage |
| `run_dataset_cleaning` | 在 Pyromind 平台异步执行清洗；用 `limit` 试跑，用 `resume_run_id` 恢复原任务 |

不要在本地执行或模拟清洗任务，也不要自行拼装 Shell 命令。生成兼容 `run_dataset_cleaning` 契约的 `clean_script.py` 并上传；该工具负责构造平台命令、传入输入与输出 Storage 路径、创建 run 目录，并提交异步任务。

任务产物全部位于工具返回的 Pyromind Storage `output_dir`：

| 文件 | 要求 |
|---|---|
| `output.jsonl` | UTF-8 `messages` JSONL；恢复执行时安全追加且不重复 |
| `stats.json` | 最终汇总，至少包含 `read`、`written`、`errors`、`duplicates` |
| `errors.jsonl` | 可恢复的行级错误，包含位置、错误原因和经过截断、脱敏的原始片段 |
| `checkpoint.json` | 已安全落盘的处理位置与时间戳；周期性更新 |

`dedupe_state.json` 可用于保存去重状态。不得把凭据或完整超长记录写入统计与错误样本。缺少 `stats.json` 表示任务未完整成功。

`run_dataset_cleaning` 提交后，平台终态通过 Kafka callback 注入 `<system_reminder>` 并自动续跑当前会话。任务处于 `Pending` 或 `Running` 时等待 callback，不重复提交。

## 工作流程

### 1. 探索并采样真实数据

使用 `preview_dataset` 检查用户提供的共享数据集或 Storage 路径。目录只有一个候选文件时直接预览；有多个候选文件且无法从上下文确定目标时，列出文件并询问用户。

样本同时覆盖普通记录和结构异常点：空值、最长记录、字符串化 JSON、内容块、工具调用/结果、自定义 role、格式错误，以及可能需要按 session/trace 分组的事件。把预览截断视为样本不完整，不根据被截断内容设计修复规则。

### 2. 确认语义映射与清洗策略

目标格式已经固定，只确认会改变数据语义的问题：源字段到 `messages` 的映射、过滤规则、长度阈值、事件分组方式和有损转换。

按源数据形态选择解析方式：

| 数据形态 | 处理方式 |
|---|---|
| 标准 JSONL | 逐行解析、转换、校验、过滤、精确去重 |
| CSV/TSV | 解析表头、确认字段语义与缺失值策略、转为 `messages` |
| 非结构化日志 | 提取轮次与角色、按会话分组、转为 `messages` |
| 对话或 Agent 轨迹 | 保留轮次、内容块及工具调用关联，归一化 role |
| 混合格式 | 按可识别结构分流，再统一为 `messages` |

若确定性转换无法生成 assistant 答案或必要标签，明确缺失信息并停止该映射；需要时建议独立的标注或生成步骤。

缺少 system message 时，只确定一个固定提示词并对全部记录复用；保留已有的非空 system message，除非用户确认替换。工具轨迹优先使用源数据中的工具定义，并扫描样本中的工具名称，避免遗漏。

### 3. 生成并上传脚本

先检查 Storage 中是否已有适用的 `clean_script.py`。满足当前目标格式、平台契约和清洗语义时复用，否则生成或修改脚本。

生成脚本时：

- 输出只包含 `messages`，并在写入前逐条执行完整格式校验。
- 流式读取；仅在必须按 session/trace 分组时做有界聚合。
- 隔离解析、映射、格式、长度、训练排除和去重错误；每 1000 条左右输出一次进度。
- checkpoint 只推进到已经安全写入输出和状态的数据。
- 恢复时避免重复输出，并保持去重状态一致。
- 相同输入、脚本和配置产生相同输出；所有阈值和映射写成命名常量。
- 生成单文件、可上传的 Python 脚本，不依赖本地 skill 目录或本地 `PYTHONPATH`。

使用 `upload_file_to_pyromind` 上传脚本，再把返回的 Storage 路径传给 `run_dataset_cleaning`。

### 4. 在 Pyromind 试跑

提交有界样本任务：

```text
run_dataset_cleaning(
  script_path="<uploaded_storage_path>/clean_script.py",
  input_path="<input_data_path>",
  limit=100
)
```

保存返回的 `run_id` 和 `output_dir`，告知用户任务已提交并等待 callback。不要在本地运行脚本，也不要轮询式重复提交。

### 5. 收到 callback 后检查 Storage 产物

所有文件都在 Pyromind Storage 中；只使用 `preview_dataset` 查看。成功终态按以下顺序检查：

1. 对 `<output_dir>/stats.json` 调用 `preview_dataset`，核对读取、写入、错误、重复数量及转化率。
2. 对 `<output_dir>/errors.jsonl` 调用 `preview_dataset`，归纳错误类型并识别系统性问题。
3. 对 `<output_dir>/output.jsonl` 调用 `preview_dataset`，抽查 3–5 条，确认每行只有 `messages`，且内容、轮次与工具关联完整。
4. 对 `<output_dir>/checkpoint.json` 调用 `preview_dataset`，确认最终处理位置与统计一致。

空输出不能作为有效训练产物。若全部记录都因明确的训练排除规则被过滤，报告“不应生成训练产物”，不要把空文件当成功结果。

向用户展示 1–3 组精简的清洗前后样例和汇总统计，并明确固定 system prompt、偏好分支选择等有损决策。

### 6. 修复、恢复或全量执行

- 解析、规则或格式错误：根据 Storage 中的错误分布修改脚本，重新上传后发起新的试跑；不要机械重试相同脚本。
- 任务中断但脚本与规则正确：先用 `preview_dataset` 检查 checkpoint 和错误文件，再使用原 `run_id` 调用 `run_dataset_cleaning(input_path=..., resume_run_id=...)`。恢复任务使用平台保存的冻结脚本，不传新脚本路径。
- 输入路径或清洗规则需要改变：不要 resume；上传新脚本并新建任务。
- 错误率超过 30% 或出现系统性错误：先修复并重新试跑。低错误率也不能替代样本语义检查。
- 试跑结果经用户确认后，使用同一脚本且不传 `limit` 发起新的全量任务；大数据集提前说明耗时和可恢复机制。

全量 callback 到达后，再使用 `preview_dataset` 完整检查 Storage 产物，并报告最终 `output_dir`、统计和剩余错误。
