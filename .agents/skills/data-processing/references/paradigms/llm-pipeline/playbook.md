# 数据准备（llm-pipeline）

- **基底**：本地 Sample（`df_run_pipeline` 隔离子进程执行 DataFlow 算子）→
  用户确认后平台全量（`df_submit_pipeline`）
- **适用**：内容级处理——规则清洗（词数/语言/MinHash 去重/PII/毒性/HTML）、
  LLM 生成与改写、质量评分、格式化；覆盖 SFT、推理、代码、知识问答、
  Agentic RAG、多轮对话、Function Call、质量评估、Text2SQL、化学抽取和
  多模态标注
- **不适用**：纯格式转换/字段映射/简单过滤 → format-conversion；
  需在特定环境执行复杂流程的编排式处理（验证/筛选/执行判定）→ environment-processing

先在本地验证最多 3 条 Sample；用户明确确认后，才提交 Pyromind 全量任务。

## 强制边界

- 工作区中间文件放在 `public_data/data-preparation/`。
- Storage 数据和平台产物只能用 `preview_dataset` 查看，不得用本地文件工具读取。
- `df_run_pipeline` 只运行本地 Sample；用户确认前不得调用 `df_submit_pipeline`。
- 平台全量输入必须已经存在于 Storage：`df_submit_pipeline` 不接受工作区路径。
  工作区写好的 Manifest 用 `upload_file_to_pyromind` 落到图片所在的 Storage 目录
  （见[平台全量输入](#平台全量输入)），不要用 sandbox 搬运。
- Sample 结果不符合预期时自行修正并重跑，迭代过程不向用户展示；只展示符合预期的结果。
- 新链路直接生成规范 JSONL，不以 `df_convert` 或 Parquet 作为正式产物。
- 新链路优先复用 DataFlow Storage 和 Operator 编排；生成、打分、过滤、去重等已有
  算子能覆盖的环节，尽量不要手写重复实现。
- 使用 LLM 的 DataFlow 算子必须由 `LoggingLLMServing` 包装。
- 用户提供打标模型网关时，把 `api_url`/`model`/`api_key` 作为
  `labeling_gateway` 传给 `df_run_pipeline` 与 `df_submit_pipeline`（配
  `model_profile="vision"`），本地试跑和平台全量用同一个网关；用户没有网关时
  不要传，走平台 `DF_*` 配置。网关整体替换平台视觉模型，不与平台配置混用。
- LLM 批处理必须分批调用（`BATCH_SIZE`，可用 `DF_BATCH_SIZE` 调整）、增量写入
  （append 模式，每批 flush 落盘）、每批更新 `progress.json`（供
  `df_check_progress` 观测），并内置断点续跑（启动时统计已处理行数并跳过）；
  否则全量运行无法观测进度、中断后只能从头重跑。

## 平台全量输入

本地试跑只验证链路是否打通，正式批量和正规化执行在平台侧完成。`df_submit_pipeline`
的 `input_path` 是 Pyromind Storage 路径（平台 pod 把 Storage 挂载到
`/target-workspace`），工作区文件不会自动上传。按输入形态二选一：

- **目录输入**：数据都在同一 Storage 目录时可直接传该目录。目录输入按**直接子项**
  划分样本：每个子目录算一个样本（递归收集其中的图片），散落的单张图片各算一个
  样本。同目录多组图对（如 PCB 待检图 + CAM 参考图）两种划分都不对，必须用 Manifest。
- **Manifest 输入**：Manifest 必须与它引用的图片处于同一 Storage 目录树内 —— 运行时
  按 Manifest 所在目录解析其中的相对图片路径，越出该目录的路径会被拒绝。

工作区写好的 Manifest 用
`upload_file_to_pyromind(file_path=..., target_dir=<图片所在的 Storage 目录>)` 上传，
把返回的 Storage 路径直接作为 `input_path`。源数据本来就在 Storage 时，只需要落一个
Manifest 小文件；不要为此创建 sandbox。

## 执行流程

1. `preview_dataset(mode="inspect")` 确认结构；目录输入先读取
   `directory_summary`，再决定继续 inspect、sample 哪些路径，或向用户确认格式意图。
   随后调用 `preview_dataset(mode="sample", n<=3)`；本地 Pipeline 的输入直接使用返回的
   `df_run_input_path`，多输入时使用 `local_sample_paths`。Storage `source_path` 只用于全量
   提交，不得作为本地路径，也不得复制 Sample 到另一个文件。
2. 按下表（运行规则）只读取相关场景 case 文档，同时读取
   [通用约定](dataflow-common.md) 和
   [输出契约](schema-conventions.md)。
3. 优先从 case 文档的 DataFlow 算子模板修改 Pipeline；只有图片任务使用
   [图片模板](multimodal_pipeline.py)，PCB 预标注使用其场景模板。case 文档中的算子链负责处理中间
   字段，Pipeline 末尾负责映射正式 Schema。
4. 调用 `df_run_pipeline`，显式设置 `model_profile` 和 `output_schema`，检查
   `processed.jsonl`、`validation.json` 和 `report.json`。
5. Sample 结果不符合预期（质量、格式、字段映射等问题）时，直接修正 pipeline 并
   重新试跑，直到结果符合预期；迭代过程不向用户展示。
6. 展示符合预期的 Sample 结果并等待用户明确确认。
7. 确认输入已按[平台全量输入](#平台全量输入)落到 Storage 后，调用
   `df_submit_pipeline(mode="full")`。收到 Kafka callback 后，调用
   `preview_dataset` 查看 `<output_dir>/report.json`；如失败，再查看同目录的
   `failure.json`、`validation.json` 和必要的 `llm_calls.jsonl`。
   图片任务若 `report.json.label_reconciliation.corrected > 0`，继续查看
   `label_corrections.jsonl`，并向用户说明修正数量、原/新标签和证据；训练数据中不写
   审计字段。
8. Agent 修复后先在本地重跑失败记录、失败前一条和同类成功记录：
   - 旧结果仍可用：`mode="resume"`，提交 `reuse_assessment` 和可选新脚本。
   - 旧结果不可用：重新执行 Sample、人工确认并创建新的 full run。
9. 提交后可用 `df_check_progress`（传 `output_dir`）查看实时进度、ETA 和最近产出。
   若用户预览后发现不符合预期、要介入调整，**先调用 `df_stop_task`**（传 `task_id`，
   或 `df_submit_pipeline` 返回的 `run_id` / `output_dir`）停掉平台任务，再修改
   pipeline 并重新提交，避免旧任务继续消耗资源或覆盖输出目录。
## 运行规则（范式内 case 路由）

| 需求 | case 文档 | `output_schema` |
|---|---|---|
| 规则清洗、语言/长度过滤、去重 | [文本规则清洗](cases/text-cleaning.md) | 下游 Schema |
| 通用生成、改写、打分、过滤 | [通用 LLM 处理](cases/generic-llm-processing.md) | `text` |
| SFT 合成与筛选 | [SFT 数据](cases/sft-data.md) | `text` |
| DPO 偏好对清洗或生成 | [DPO 数据](cases/dpo-data.md) | `dpo` |
| 推理问题和答案合成 | [Reasoning 数据](cases/reasoning-data.md) | `text` |
| 代码指令和代码生成 | [Code 数据](cases/code-data.md) | `text` |
| 文本/Markdown 清洗并生成 QA | [通用 LLM 处理](cases/generic-llm-processing.md)（暂无专属 case） | `text` |
| Agentic RAG 任务与 QA | [通用 LLM 处理](cases/generic-llm-processing.md)（暂无专属 case） | `text` |
| 多轮对话生成或整理 | [通用 LLM 处理](cases/generic-llm-processing.md)（暂无专属 case） | `multiturn` |
| 工具定义和调用轨迹 | [通用 LLM 处理](cases/generic-llm-processing.md)（暂无专属 case） | `function_call` |
| 样本质量评分、保留、改写或丢弃 | [通用 LLM 处理](cases/generic-llm-processing.md)（暂无专属 case） | `quality_evaluation` |
| 已有 SQLite Text2SQL 数据精炼 | [通用 LLM 处理](cases/generic-llm-processing.md)（暂无专属 case） | `text2sql` |
| 图片 OCR、理解和多图语义标注 | [多模态标注](cases/multimodal-labeling.md) | `vision` |
| PCB AVI/AOI 真点/假点预判、问题分类及区域定位 | [PCB 预打标](cases/pcb-inspection.md) | `structured` |
| 从文本抽取 SMILES | [通用 LLM 处理](cases/generic-llm-processing.md)（暂无专属 case） | `text` |

标注“暂无专属 case”的场景按[通用 LLM 处理](cases/generic-llm-processing.md)
的流程执行，输出格式严格以[输出契约](schema-conventions.md)中对应
schema 为准；专属 case 文档待补充。

PCB 预标注直接输出可读的标注 JSONL，具体结构按任务确定，使用 `output_format="structured"` 与
`output_schema="structured"`；用户指定其他格式时调整响应 Schema。只有准备训练
messages 时选择 `vision`。具体完成标准见 PCB 场景文档。

## 运行与完成条件

- 只读取需求匹配的场景 case 及其模板；例如 DPO 不读取文本清洗等相邻模板。
- 按 `failure_stage` 修复：`input_resolution` 只改用工具返回的本地路径，
  `pipeline_resolution` 只修正工作区相对路径，`pipeline_execution` 根据 stderr/report
  修脚本；仅 `runtime_dependency` 可检查运行环境，且不得浏览 SDK 仓库源码。
- 相同参数得到相同 `error_code` 后不得原样重试；本地输入、输出和报告路径均使用工具
  返回值，不用 terminal 搜索。
- 文本任务使用 `model_profile="text"`；图片任务使用 `model_profile="vision"`。
- 图片 Pipeline 只配置 `ImagePipelineConfig`，不得自行实现 HTTP、Base64、重试或
  Checkpoint。
- `directory_summary` 只是结构摘要，不是数据 Schema；低置信度、混合结构或需要
  类别/异常/大小/命名模式覆盖时，继续 inspect 或自行选择 `sample_paths`。
- Text2SQL Sample 使用 Python 3.10 的 `DATAFLOW_PYTHON`；Pyromind 固定
  `open-dataflow==1.0.10`、CPU 执行。
- `processed.jsonl` 必须通过所选 Schema 校验，ID 唯一且不含运行审计字段。
- Report 必须包含输出数、模型调用、失败、Checkpoint、校验、Revision，以及场景
  Pipeline 提供的可选 `scenario_metrics.json`。

## 暂不使用

- PDF/MinerU、PDF-VQA、Speech、FlashRAG/Retriever、GPU/本地模型算子。
- 需要独立 Embedding Serving 的完整 Text2SQL 合成。
- `KBCChunkGenerator`、RDKit SMILES 等价评估、默认 CUDA 的质量过滤器。
- `CodeSandboxSampleEvaluator`；不得执行生成代码。

## 图片补充参考

- [image_utils API](image-utils-api.md)
