# PCB AVI/AOI 预打标

用于 PCB 裸板线路与板面质检的报警复判、问题分类和区域定位；局部报警图的
结论只覆盖该样本，不代表整板合格。领域术语（结构缺陷、外观因素与业务处置）
通过文件读取工具读取 `knowledge/business-domain/pcb-avi-aoi.md`：这是运行时
知识库逻辑路径，直接传给文件读取工具，不要相对本 skill 目录拼接或查询环境变量。
当前客户规则优先；不把裸板类别直接套用于 PCBA 元件和焊点；现场阈值和标签约定
不默认迁移至其他产线。

## 打标模型（先问用户）

开始预打标前，先确认**打标模型的 LLM 网关**及可用凭据来源：

- `api_url`：OpenAI 兼容的 chat-completions 完整地址，例如
  `https://inference-cn-east-1.pyromind-asia.cn/inference/inf-xxxx/v1/chat/completions`；
- `model`：该网关上的模型名，例如 `pcb_avi_sft_merge_v10`；
- `api_key_env`：当前执行环境中已授权的网关密钥变量名；不把密钥值写进对话配置。

CLI 的 `gateway` 配置只写 endpoint、model 和宿主密钥环境变量名，不能写密钥。
平台按 [playbook](../../playbook.md)准备并提交 DataFlow 全量任务：设置
`model_profile="vision"`，在准备和提交 Python 进程的内存中整体替换
`PYROMIND_DATAFLOW_PROFILES` 的 `vision` 项；端点、模型及密钥都来自同一用户网关。
密钥只从该已授权变量取值，不写入脚本、冻结图或落盘配置。经校验 Tool 后的新进程须
重新构造相同配置，公开配置指纹不一致时停止提交。
用户说没有、或明确用平台默认时不替换 profile，沿用执行环境的视觉模型配置。

用户网关与平台配置不混用，缺少网关密钥时不能拿平台 key 顶替。
本地试跑和平台全量必须用同一个网关；换了网关或模型先重做小样，按复用评估决定
兼容续跑或新 run，不自动复用旧结果。带环境声明的图通过普通 `submit_workflow`
提交，最终 Studio 请求可能保存 command 环境；不保存或打印绑定后的图。

## 完成标准

预标注应保留四项核心信息，字段名、JSON/JSONL 和整图或逐区域组织优先遵循
用户要求或下游工具契约：

- **来源**：通过原始图片路径或有明确映射的标识准确关联图片；多图时区分待检、
  参考和辅助图，区域能关联到对应图片，不让模型猜路径或丢失目录信息。structured
  运行时已按输入角色把图片路径注入每行的 `source_images`（平台运行为 Storage
  对象路径），模型既不需要也不应输出图片路径；区域框只画在待检原图上。
- **判定**：给出有依据的标签或问题类别。类别取自当前任务或领域参考，
  多类别任务可让每个区域独立带类别，不强行压成一个主类别。
- **区域**：能定位时提供区域，并说明对应图片、表示方式和坐标尺度。
  使用裁剪或变换视图时保留回到原图的映射；无法定位则说明，不猜框补位。
- **描述**：简短说明可见现象，以及影响判断的具体不确定性。

交付可读、可用于人工复核和后续转换的标注数据，不让用户从训练 messages 中自行
提取标注；Label Studio 与训练格式转换在后续按需进行。已有产物满足需求即可交付，
不要求独立导出脚本。

下游接入按行内字段声明，不重写数据：进 Label Studio 复核时，`source_images` 的
角色键、判定字段和区域字段用 `field_map` 绑定到控件（见 label-studio skill 的
`references/custom-bindings.md`），产出文件原样导入。行内缺这些字段时应回到本
场景补契约，不要让下游或 Agent 手工拼接 `source_manifest.jsonl`。

## 判断依据

区分视觉观察与客户验收。有依据时给出真点/假点预判；缺少客户规则只影响依赖
该规则的结论，例如允许尺寸或特定位置的放行标准。不机械地将所有判定置空，
也不因此丢掉已能判断的类别、区域和描述。氧化、异物、错位等不天然等于假点；
保守拦截不等于已确认实体缺陷。预标注与原始人工标签分开保留。

图像角色和配对关系依据数据说明、metadata 与实图确认，不能只靠文件名或明暗。
跨模态、未对齐图片不直接逐像素差分，但仍可根据可靠的结构对应和实拍证据判断。
反色、增强或差分响应仅作辅助，不能替代原图证据。

## 输出引导示例

以下案例仅用于说明标注数据可以怎样呈现，不是固定输出协议，也不据此增加专用
校验代码。字段名、层级、类别、区域表示和文件格式按当前任务或下游需求确定。
此例的 id 关联原始待检图，label=false 表示假点，boxes 是待检原图上的
0–1000 归一化 xyxy（横轴按宽、纵轴按高）。

```json
{"id":"3-0803A/18/P6K05774A0/3/内　层/2","label":false,"category":"垃圾","boxes":[[383,339,635,711]],"note":"垃圾加氧化"}
```

区域信息应与观察一致：可定位的垃圾、氧化等假点也可以给框；无异常或无法定位时
说明情况，不伪造框。多类别任务可按区域分别记录类别，不必压成一个主类别。

## 复用执行链

少量探索可直接看图；批量任务通过文件读取工具读取以下逻辑路径：

- 图片执行 API：`.agents/skills/data-processing/references/paradigms/llm-pipeline/vision/image-utils-api.md`
- 可修改的配置示例：`.agents/skills/data-processing/references/paradigms/llm-pipeline/vision/templates/pcb_inspection_pipeline.py`

PCB 默认设置 `ImagePipelineConfig.output_format="structured"`，执行工具使用
`model_profile="vision"`、`output_schema="structured"`，直接生成经过 Schema 校验的
标注对象，不生成 messages。只有准备训练数据时选择 `vision` 输出模式。
小样与全量沿用上层 playbook；恢复不得更换输出模式或 structured Schema。

输出契约（`output_schema="structured"`）：每行是任务 `response_json_schema` 校验后的
对象，业务字段由该 Schema 定义，模型不能返回或覆盖 `id`；运行时再注入源样本 `id` 和
源图关联 `source_images`（用输入的图片角色做键，值是图片路径），所以整行自包含，
下游拿这一行就能定位原图。structured 不走固定产物校验器，也不含训练 messages 包装；
Label Studio 与训练格式转换在后续按需进行。

输入准备需保留样本边界和原始路径映射。同目录多组图对不能直接套通用目录发现，
它会把子目录中的所有图片合为一个样本；应提供显式 Manifest，并保留相对路径基准。
平台全量提交时 Manifest 必须与图片处于同一 Storage 目录树，经
`/workspace/<同一目录>/` 落位（见 playbook 的「平台全量输入」）。按源标识及
图片路径关联结果，不按行号拼接，失败记录可能被跳过。

## EchoMind Dataset 节点的三图输入

已核实为「每个样本目录一组图」，且 `defect.jpg`、`diff.jpg`、`gt.jpg` 分别代表待检实拍图、
差分图、同位置参考图时，可使用下列确定性转换。先在图片根目录上层生成清单，再把清单提交给
已有的 PCB 预打标 Pipeline。应保存经确认的业务规则和训练提示词，供同一版本的执行图复用。

```sh
python3 .agents/skills/data-processing/scripts/preparation/pcb_dataset_inputs.py manifest \
  /workspace/dataset/images /workspace/dataset/source.jsonl \
  --inspection-rules-file /workspace/dataset/inspection_rules.txt
```

生成执行轮次的独立清单时，先创建该轮的 `incoming` 目录，再给 `manifest` 加上 `--stage-images`，
例如把 `/workspace/dataset/images` 转为 `/workspace/run/incoming/source.jsonl`。脚本会复制已核实的图片
到清单旁的 `images/`，预打标输出应引用这份清单的图。`messages` 使用同一清单完成匹配，产物审计记录
原始 Storage 路径。清单与旁边的 `images/` 均不能已存在。实际挂载为 `/target-workspace` 时，两个命令
都使用该挂载下的路径和 `--storage-root /target-workspace`。

将已复核的 structured JSONL 与原清单按 ID 和三个角色下的原图路径精确匹配。每条已复核行应附
`confirmed: true`，可通过 `gt` 提供已确认的完整答案。只有用户明确批准本版本使用预标注预测
作为训练标签时，才在发布配置加入 `--allow-pseudo-labels`。此时脚本仍校验分类、置信度和区域框，
且在审计清单区分预测标签与人工确认标签。

```sh
python3 .agents/skills/data-processing/scripts/preparation/pcb_dataset_inputs.py messages \
  /workspace/run/reviewed.jsonl /workspace/run/dataset \
  --source-manifest /workspace/dataset/source.jsonl \
  --system-prompt-file /workspace/dataset/training_system_prompt.txt \
  --user-prompt-file /workspace/dataset/training_user_prompt.txt --val-size 2
```

已封批并带有确认标签的 Tracer 输入可直接传入批次目录。脚本只读取 `train.jsonl`，从本批数据
里抽取 Train 节点内部验证样本。固定业务 Eval 数据单独绑定到 Eval 节点。若批次中省略
`system_prompt`，使用同一份版本化训练提示词文件。

```sh
python3 .agents/skills/data-processing/scripts/preparation/pcb_dataset_inputs.py messages \
  /workspace/project-batches/tracer/batch-1 /workspace/run/dataset \
  --system-prompt-file /workspace/dataset/training_system_prompt.txt --val-size 2
```

输出目录须是新目录。`train_messages.jsonl` 和 `val_messages.jsonl` 的所有角色均使用同一形状的
`type/value` 内容块，图像引用相对于输出目录。脚本保留三图顺序，复制原图字节到输出目录，
并在 `source_manifest.jsonl` 记录原始 ID、图片角色、路径、标签来源与切分。
平台任务通过冻结脚本的 Storage 路径执行该命令；实际挂载为 `/target-workspace` 时，输入及输出参数
应使用该挂载下的路径，并加 `--storage-root /target-workspace`。不依赖宿主 Python 或临时安装依赖。

配置示例的 Prompt 和 Schema 应按任务调整，不将上述案例固化成通用协议。
类别按任务或领域参考确定。任务文本使用
`id/image_labels/inspection_rules`；无客户规则时如实注明，不以此否定视觉预判。
旧顶层 `user_prompt/prompt` 会优先于模板生效，即使 `user_prompt_key=None`；
不适用于当前任务的旧文本应移入 `source_metadata` 保留。
