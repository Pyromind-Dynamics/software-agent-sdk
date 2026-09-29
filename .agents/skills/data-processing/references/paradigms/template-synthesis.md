# 缺陷模板合成

通用组件负责变形、注入、搜索和证据；Agent 在 Python 驱动中实现当前数据的
材料解释、模板提取和验收策略。类别名不决定生成机制。结构缺陷可检查连接、
距离或形状，外观缺陷可检查纹理、色调和承载区域，不统一要求连通块变化。

源图、正常参考和异常证据分别保留。先确认正常参考确实正常、模板表达当前
缺陷；标签图不能直接充当正常外观图。缺少语义或材料依据时保留不确定性。

## 通用入口与阶段接口

[template_synthesis.py](../../scripts/preparation/template_synthesis.py) 的
`synthesize_template(source, reference, ...)` 不要求二值材料或配对 GT。
输入图片为 RGB uint8，掩码为二维 bool，坐标为像素 xyxy，右/下边界不包含。

| 类型 / 参数 | 内容 |
|---|---|
| `SourceEvidence` | image、可选 reference、annotation 掩码、Template、可选命名 regions；均为源坐标 |
| `Template` | 异常 mask、可选源纹理 texture；从标注范围内提取，不能用整块正常材料代替 |
| `reference` | 目标正常图；可直接使用已验证正常实拍或策略生成的正常参考 |
| `candidate_centers` / `allowed_mask` | 候选中心与允许修改区域，均为目标坐标 |
| `regions` | 任意数量的命名区域掩码，可包含材料、边界或需要验证连接关系的区域 |
| `Acceptance` | min_area、max_area、max_clip_fraction、visibility_threshold、min_visible_pixels |
| 搜索与渲染 | seed、max_attempts、scale_xy、angle_deg、pair_transform、feather、可选 fill_rgb |

三个回调均返回 `Sequence[Check]`。`Check(name, passed, reason, measurements)`
中 passed 为 True / False / None，None 表示无法判断；空结果不是通过。
组件自动记录 stage，回调收到输入副本，不可通过修改它们改变后续渲染。
旧调用可缺失源 image，但不能通过源验证；缺少配对 reference 时，策略须说明
可用的标注或其他验证依据，不能把目标正常图假装成源正常参考。

| 回调 | 输入及职责 |
|---|---|
| `validate_source(SourceEvidence)` | 验证源模板、正常参考和材料解释支持本轮意图；失败/不确定即停止该源的搜索 |
| `validate_candidate(CandidateEvidence)` | reference、transformed、constrained、allowed_mask、regions、clipped_fraction；检查完整模板及约束后的几何 |
| `validate_rendered(RenderedEvidence)` | image、reference、mask、changed、regions、bbox_xyxy；对实际成图验证缺陷效果 |

transformed 是裁到目标画幅、尚未约束到 allowed_mask 的模板，uncropped_area
记录变形后完整面积；裁剪比例包含出界和允许区裁剪。不得只检验理想 mask，
忽略羽化、纹理覆盖对最终结构的影响。rendered 的所有图、区域和框已经执行
pair_transform；需要定位的锚点请表示成 regions 掩码，避免手动坐标漂移。

固定执行：源验证 → 有限候选搜索 → 面积/裁剪与策略几何检查 → 注入 →
整组变换 → 实际像素变化与策略成图检查。可见性使用合成图与正常参考的绝对
差分，正常边缘的高通响应不能证明缺陷存在。

返回 `GeneratedSample`，包含图、掩码、同步后的 regions、artifacts、checks
和 parameters。可用 `reference_synthesis.export_sample` 导出，区域掩码随产物
保存，源图、可用参考及源掩码也进入证据绑定。省略回调仅用于旧流程探索，
validation_coverage 不完整时 program_status
为 needs_review，不能升级质量。新增完整验证版本为 validation_version=2。

## 搜索与验收

在计划中分别记录 search 与 acceptance，并保存来源、用途、类别定义、策略
代码和修改依据。一个运行内固定验收；只在既定搜索空间中尝试位置、变形和
素材。`SynthesisFailure` 提供 code 与 attempts，逐次保存阶段、检查、测量值
和原因。返回不足数量是合法结果，不能靠放宽验收凑数。

修改验收需要独立数据依据、新运行及新的源/小样验证；不复用旧复核。回调与
驱动是 Agent 编写的 Python，这些记录提供可审计性，不是任意代码的安全隔离。

## 可选的两材料配对组件

[reference_synthesis.py](../../scripts/preparation/reference_synthesis.py) 保留
align_pair、build_normal_reference、extract_defect、context_check、structure_check。
[paired_reference_example.py](../../scripts/preparation/paired_reference_example.py)
演示两材料策略；多材料或复杂光照直接使用通用入口，不伪装成二值材料。

- `normalized_box_to_pixels` 仅转换显式 0..1000 标注；配准移动实拍及标注，GT 不动。
- `build_normal_reference` 从排除异常后的可信区域采样外观，按 GT 几何重绘。
- `extract_defect` 支持 extra/missing 的材料差异、appearance 的局部外观差异及
  reviewed 掩码；分割、配准、面积和外观阈值由当前数据确定。
- `synthesize_sample` 是通用入口的兼容包装，保留原参数及 `mask -> Check` rules；
  上下文占比、边界关系和相对距离检查仍执行，不替代源及成图语义验证。
  新增 source_evidence、regions 和三个 validate 回调；source 回调需要真实源证据。
- 示例新计划声明 validation_version=2、material_mapping_basis，并按当前任务
  定义 structure_rule。其示例策略测量实际分割变化；结构操作未提供拓扑期望时
  返回不确定。它仅演示一种可替换策略，不是所有缺陷的验收定义。
- 原有计划仍可运行，缺失阶段不会被伪装成通过。原 GT、输入及旧运行保持只读。

两材料示例保留 donor_rgb/donor_cam、annotation_box_1000、source_material_threshold、
gt_foreground_min/max、alignment、reference、extraction、placement、structure_rule、
synthesis、variants、context_policy、source_split、purpose 字段。
ContextPolicy 的起点仍是材料比例 0.05、边界比例 0.25、相对距离 1.0、边界带宽
0.25；必须依据样例确认，不能因候选耗尽自动放宽。跨参考材料映射需要独立依据。
未知 split 只允许 purpose=method_validation；正式候选需确认 train 来源。

示例保留高通：gray(image) - GaussianBlur(gray(image))，signed .npy 与幅值 PNG
均保存。它是纹理观察图，不是 GT 差分；通用入口默认导出 absolute_reference_difference。

## 打包、目录和复核

[打包器](../../scripts/preparation/bundle_template_pipeline.py) 将两个组件与
Agent 驱动冻结成单文件。领域策略写在驱动中，无需注册插件或新增 DSL。

```bash
python <skill目录>/scripts/preparation/bundle_template_pipeline.py \
  <驱动.py> <新运行>/pipeline.py
```

生成仍使用 df_run_pipeline / df_submit_pipeline，model_profile=none、
output_schema=artifacts。平台仅上传单个 Pipeline 和固定 runtime；素材与计划
必须在执行环境可访问。

每次运行使用新目录，阶段输出分别放置：

- generation/：processed.jsonl、assets、augmentation_plan.json、frozen_pipeline.py、
  review_input.jsonl、review_binding.json；业务报告命名 synthesis_report.json、
  synthesis_validation.json、synthesis_progress.json、synthesis_report.html。
- review/：review_output.jsonl；读取 generation/review_input.jsonl，图片相对该
  输入文件解析。执行器报告落在 review/，不覆盖生成报告。
- quality/：reviewed.jsonl、reviewed.quality.json 及复制后的可交付图片。
  各阶段 report.json / validation.json 仅表示执行器自身结果。

## 固定视觉复核路径

视觉复核继续复制 [image_synthesis_review.py](../../scripts/preparation/image_synthesis_review.py)
作为 Pipeline，model_profile=vision、output_schema=vision，不自建传输或算子。
输入行包含 id、images、image_labels、user_prompt。模板提供源图、正常结构、
标注和掩码；候选盲审提供无框图、正常参考、观察差分、尺寸及完整类别定义。
候选预期类别和位置保持隐藏，源上下文可作为对照。模型返回既有 answer 字段：
category、bbox_xyxy、realistic、extra_anomalies、uncertain、context_consistent。

计划中的 category 使用明确 ID，category_aliases 显式声明别名到 ID 的映射。
复核前固定类别表与 review_min_iou（默认 0.5）；不做模糊归一化，不尝试换轴、
翻转或调阈值迎合预期框。

`freeze_review_binding` 绑定计划、样本、资产、复核输入、selected_ids、IoU，
并用 strategy_path 绑定生成时保存的单文件策略。路径相对 generation 根目录，
不可越界。新产物必须提供策略文件；旧 binding 可读，但缺证据不能通过。
`min_iou` 必须与计划一致，计划还应记录 requested 或 variants。

使用打包后的 [finalize_synthesis_review.py](../../scripts/preparation/finalize_synthesis_review.py)
读取 review_job.json（binding_path、review_output_path，相对 job 文件），输出到
quality/reviewed.jsonl。它保留原始生成与模型输出，重新校验哈希并复制产物。
只把最终汇总当作交付依据：visual_assessment 分列类别、定位、真实性、额外异常、
确定性和上下文；quality_reasons 保留缺失、过期及阶段失败原因；摘要列 requested、
generated、generation_shortfall、quality_passed。未复核、不确定或缺验证不放行，
training_ready 始终为 false；执行成功不表示质量通过。
