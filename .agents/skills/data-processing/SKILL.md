---
name: data-processing
description: >-
  Pyromind 数据处理统一入口。用于格式转换、DataFlow 内容处理、从源数据分析
  标签分布、PCB AVI/AOI 图片预打标、按已确认 Gap 合成文本或图片，以及数据自带镜像中的
  环境编排处理。统一执行预览、小样、人工确认、异步全量和报告校验。
license: MIT
---

# 数据处理

数据处理任务统一从本 skill 进入：先按路由表选定处理范式并读取对应 playbook
（范式内按 case 路由表读取场景 case 文档），再按通用 SOP 执行。用户 Storage
已挂载为 `/workspace`，优先用 `read`/`terminal` 直接查看；`sandbox_sample.py` 的
`--input` 直接接受 `/workspace/...`，不需要先物化到会话工作区。参数形态与 `--config`
的使用条件见 `references/terminal-cli.md`。

## 数据中的文件引用

执行时用 `/workspace/...` 访问 Storage；持久化到数据文件中的媒体引用不带
`/workspace` 或 `/target-workspace`。数据集内部引用保留 `images/001.jpg` 这样的
相对路径，并明确以数据集目录为基准；跨目录引用使用 `/datasets/...` 这样的
Storage 根路径。不要把脚本读取文件时的容器路径直接写进输出数据。

确定性转换脚本使用 `scripts/cleaning/cleaning_utils.py` 的
`normalize_media_references` 处理显式媒体字段，使用 `write_jsonl` 或
`run_cleaning` 输出时会自动处理。读取 Storage 引用可用
`resolve_storage_reference(value, mount=当前挂载目录, dataset_root=数据集目录)`；
得到的执行路径只用于读文件，不写回数据。默认 Vision collator 直接读取的最终训练清单使用训练挂载路径，见格式转换 playbook 的「平铺 Vision 训练清单」。
只处理确定的文件引用，不全文替换正文、提示词、来源记录或日志，也不改动远程 URL。

领域参考（如 PCB AVI/AOI）不内联在本文件：由对应 case 文档按需引入，并在那里
给出运行时知识库逻辑路径的读取方式。执行方式遵循对应范式：模型批处理在小样确认后
通过平台任务执行全量；确定性格式转换可以在会话沙箱执行并写回 Storage。

需要平台提交时，读取[SDK 提交与回调](../workflow-execution/references/sdk-execution.md)，
按 playbook 保留脚本冻结、小样与产物约定，再以当前场景关联调用 SDK。
SDK Main 不再注入 Pipeline、评测、正式工作流、调试、清洗及 EDP 三段共八个提交 Tool，
不按历史任务恢复注入。Pipeline 支持 none/text/vision，EDP 分片也通过普通准备函数与 SDK
执行。依赖未就绪先修复执行环境；SDK 未启用的部署按当前实际清单使用原入口。
一次任务只执行一次平台 create，不因元数据绑定失败重复提交。
含 runtime 环境声明的校验图用普通 `submit_workflow` / `submit_workflow_batch` 提交，
不能直接 `studio.create`。本地冻结资产不保存密钥；helper 仅在最终 Studio POST 前将
必要环境值填入内存请求。平台可能保存该任务定义，不保存或打印绑定后的图。

## 范式路由（先做这一步）

| 需求特征 | 处理范式 | playbook |
|---|---|---|
| 格式转换/字段映射/简单过滤，无 LLM、无跨行操作 | format-conversion | references/paradigms/format-conversion/playbook.md |
| 内容级清洗/抽样/生成/评分，用 DataFlow 算子或 LLM | llm-pipeline | references/paradigms/llm-pipeline/playbook.md |
| PCB AVI/AOI 真点/假点预判、问题分类及区域定位 | llm-pipeline（PCB 预打标，默认 structured JSONL） | references/paradigms/llm-pipeline/playbook.md → cases/vision/pcb-inspection.md |
| 从源数据统计/推断标签分布并提出 Gap，不依赖 eval | distribution-analysis | references/paradigms/distribution-analysis.md |
| 按已确认 Gap 合成文本或图片数据 | gap-driven-synthesis | references/paradigms/gap-driven-synthesis.md |
| 单条数据要在特定环境（数据自带镜像）执行复杂流程（跑命令/测试/判定），需多阶段编排 | environment-processing | references/paradigms/environment-processing/playbook.md |
| S2/LeRobot 机器人数据的多模态对齐、静止帧清理、批量转换与校验 | environment-processing（embodied case） | references/paradigms/environment-processing/cases/embodied-data-cleaning.md |
| 编排任务无匹配 ProcessingProfile（一次性/探索性） | environment-processing（模式 B） | 同上 |

负向边界：仅创建/管理单个沙箱容器 → sandbox；训练效果/loss 分析 →
training-analysis；模型效果评测 → inference-evaluation；
生成训练工作流 → generate-workflow-dsl；工作流调试 →
debug-workflow。不要为搬运或组装 Pipeline 的 Storage 输入另起独立 sandbox 容器
（Pi 会话自带的执行沙箱不属于此列）。路由不确定时先向用户提问确认，不要猜。

本 skill 中的「向用户提问确认」：用文字把缺口/候选方案讲清楚并结束本轮，等用户
回复后再继续，不要自行假设一个答案继续执行。

## 通用 SOP（所有场景共享的控制面骨架）

1. **探查**：直接用 `read`/`terminal` 查看 `/workspace/...`。目录条目很多时用
   `ls`/`find`/`grep` 在目标目录内定位，不要反复整目录翻页；每个数据集的结构
   确认一次完成（列表 + schema + 样例）。数据集一律 `/workspace/<path>`，skill
   脚本一律 `.agents/skills/<skill>/...`（terminal 每次从工作区根开始）；禁止
   `find /` 或扫描 Storage 挂载——挂载里有很多会话目录，全盘扫描会跑满命令超时。
2. **选型**：按上表读取范式 playbook；范式内按 case 路由表只读取当前场景
   相关 reference；领域参考由 case 文档按需引入。
3. **小样执行**：Agent 按真实 schema 写 Python Pipeline，用 `sandbox_sample.py`
   直接传参（`--pipeline/--input/--output/--model-profile/--output-schema/--timeout`，
   详见 `references/terminal-cli.md`），只有 flags 覆盖不到的字段才用 `--config`。
   两种方式都完整处理传入的输入（可直接传 `/workspace/...`），不负责抽样；
   清洗/合成小样由所选输入和计划控制，精确分布统计可直接统计 `/workspace`
   全量文件。执行发生在会话自己的平台沙箱内：Pipeline、输入和产物都不离开
   沙箱，只有日志尾部与最多 3 条样例记录回到对话。
   图片 Pipeline（`model_profile="vision"`）按相对 manifest 的路径读图：manifest
   与图片放在同一工作区目录或同一 Storage 目录时传 **manifest 文件**，运行环境
   就地读取它引用的图片；传目录会走目录发现，manifest 里的字段（`image_labels`
   等）不会被读取，只适用于无 manifest 的散图。图片确实缺失时，报错会说明缺
   什么、该改传什么。
   普通 Python 脚本直接通过 terminal 执行。需要 DataFlow 环境、模型配置、格式校验和报告时，
   阅读 [terminal-cli.md](references/terminal-cli.md)，通过 terminal 调用运行脚本；
   不要调用旧的专用执行 Tool。模型配置由执行环境注入，不在命令或配置文件中填写密钥。
   直接执行小样入口，由脚本选择解释器并复用或等待后台预热；无需先探查环境。
   需要单独确认时只用 `--check`。`ready=false` 也可以运行小样，脚本会按需安装。
   如果托管安装失败，记录脚本报告的具体原因与已完成的输入准备，等待环境修复后续跑。
   不要使用默认 `python3` 反复测试导入、手动 `pip install`，或另写原生图片解码器来绕过依赖。
   算子导入路径照抄
   `references/paradigms/llm-pipeline/pipeline-reference.md` 的导入路径表，算子名、参数与
   语义以 case 文档为准；不要用 `import dataflow` 加 `getattr` 探测，或翻
   site-packages 反查算子签名。
4. **门禁**：Taxonomy 与合成小样通过后必须获得用户明确确认才提交后续全量；
   已有标签的确定性全量统计不设人工门禁，且模型调用必须为零。
5. **全量**：需要平台执行时，按范式准备实际图并通过 SDK 提交。提交前先按
   llm-pipeline playbook 的「平台全量输入」确认 `input_path` 已指向 Storage 输入，
   工作区文件直接写入 `/workspace/<dir>/` 落位（Storage 已挂载且可写）。
   EchoMind 面向 SFT 的可复用 Dataset 使用真实源路径作为 `input_path`。需要生成
   manifest 时传 `prepare_script_path`（原始输入、输出 manifest 两个位置参数）；
   传 `sft_export_script_path`（processed.jsonl、输出目录、可选固定训练中验证集）。
   按已确认的业务口径传 `batch_label_source`。原始采集由 Dataset 标注选 `dataset`，
   封批数据已经包含可训练标签选 `collection`。口径待定时先确认。
   脚本使用已通过小样的同一实现。指定 `fixed_validation_path` 时原样保留该固定集。
   验证可复用性时，传原成功 Dataset 的 `replay_of_task_id` 和另一份 1–5 条的小样。
   固化回放独立登记，省略原研发的 `plan_task_id` 与 `plan_attempt`。先核对已成功的回执，只补缺少的小样。
   可信回调确认任务归属后，单独登记真实 task ID、Recipe、代码、工作流及输出元数据；
   登记失败只补登记，不重新提交。EDP 保存每片 task ID、run ID、manifest 与输出目录，
   部分成功后不整批重投。
   DataFlow/EDP 平台任务（含具身
   case 的逐 episode 执行）等待终态回调。用户需要查看运行进度时，按
   [挂载进度参考](references/storage-progress.md) 从已知输出目录读取一次；不循环轮询。
   需介入平台任务时先停止；启用 SDK 时按 [停止参考](references/sdk-stop-task.md)
   调用 SDK 并确认终态，其他会话使用 `df_stop_task`。
6. **分诊**：回调后先看 report.json / validation / verdicts，按失败分类决定
   resume 还是新 run；终止/失败不自动重提交，交用户决策。
   report.failures 非空时（failures.jsonl：无响应/截止/解析失败的记录），
   把其中的 input 行重组为子集输入补跑同一 pipeline 并合并产出，单条挂起
   不再阻塞整轮（墙钟截止见 llm-pipeline 的 pipeline-reference.md）。子集保留相对图片路径时，
   文件与原 source 放在同一目录；另存到其他目录时，先将图片引用按原清单目录转换为
   `/workspace/...` 绝对路径。图片必须在当前用户挂载的 Storage 内，禁止 `..` 和符号链接越界。
7. **交付**：展示产物与统计；场景内的产物契约与失败分类学以 playbook 为准。
   生成平铺 Vision 数据准备脚本时，train 和 val 共用同一导出逻辑：写每行 JSON 前，
   按该样本的源图片根将图片列转换为训练容器可读的 `/workspace/...` 绝对路径，
   保留图片顺序和监督字段。复用格式转换 playbook 的 `training_image_record`；统计与审计单独保存。返回实际 Storage 文件路径，
   下游训练直接使用这些文件，不要求再次复制或登记 Dataset Version。
   图片预览用 `get_storage_url` 取完整签名 URL 并以 Markdown 嵌入对话，让用户
   直接看图；保留完整签名 URL，不放进代码块。非图片产物（CSV/JSON/报告等）同样
   原样使用返回的 URL（含完整 `?X-Amz-...` 签名参数），去掉签名会 403，不要缩短
   或改写链接。仅在用户明确只要链接时改用普通链接。

一期边界：分布分析不读取 eval/rubric/badcase；不注册 Dataset Version 或
SQLite 元数据；不处理音视频。图片合成可复用、组合或编写策略；适用条件、
业务规则和验证依据写入合成计划，详见 gap-driven-synthesis，不默认沿用
其他场景的像素或类别规则。
