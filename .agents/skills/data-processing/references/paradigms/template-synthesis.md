# 配对参考图上的模板合成

将三种来源分开：**GT 提供正常结构，实拍可信区域提供色调/纹理，已标注异常
提供缺陷形状/像素**。默认使用同一样本配对的 GT；跨样本需单独验证尺度、材质
和结构兼容性。源图与原 GT 始终只读。

## 执行入口

优先使用 [reference_synthesis.py](../../scripts/preparation/reference_synthesis.py)
提供的阶段函数与单样本流程；[template_synthesis.py](../../scripts/preparation/template_synthesis.py)
是低层变形/注入组件，单独调用不代表完成质量验证。

1. `normalized_box_to_pixels` 显式转换 0..1000 xyxy 框；像素框不做此转换。
   所有框为右/下边界不包含的 xyxy。图片为 RGB uint8，掩码为二维 bool。
2. `align_pair` 将实拍图/材料掩码平移到 GT 坐标，返回 shift、有效区与失配率；
   GT 不移动。标注、所有异常排除区、已复核掩码也必须映射到同一坐标。
   超出支持范围的旋转/缩放配准由策略实现，不能静默拉伸。
3. `build_normal_reference` 在缺陷排除区之外、双方材料一致的内部区域统计
   前景/背景的中位色和有上限的稳健纹理幅度，按 GT 几何重绘。颜色统计不足或
   配准失败即失败，不改用异常实拍图作底图。当前实现适合两种材料；其他材料
   或空间光照变化由策略适配并试样。输出 NormalReference 保存 GT、重绘图、
   材料掩码、来源与参数。
4. `extract_defect` 返回带源 ID、标注框、提取方法的 DefectTemplate。reviewed
   模式接收已复核掩码；extra/missing 用配对材料结构差异；appearance 用实拍与
   重绘参考的局部外观距离，并排除正常材料边界。均限制在标注与有效区内，
   过滤小分量、检查占框比例。整块正常铜或标注矩形本身都不是异常证据。
   同时保存源正常结构的材料掩码，用于独立检查迁移关系。
5. `placement_region` 提供边缘/内部候选中心，`synthesize_sample` 固定执行：
   变形 → 允许区约束 → 裁剪损失/面积与业务检查 → 注入 → 整组变换 →
   重算 bbox 与高通 diff → 检查掩码内可见性及修改范围。候选按 seed 稳定打乱，
   尝试数有限；返回失败原因，不用大幅裁剪强行塞入。
6. `export_sample` 保存图像、掩码、原 GT、有符号高通数组及三联复核图；
   新产物默认为 visual_status=not_reviewed、training_ready=false。

`structure_check` 支持加/减材料后的连通块变化与贴边关系。线宽、间距等额外
业务规则作为 `mask -> Check` 回调传给单样本流程；所有规则通过才渲染。
模板像素和掩码同步变形；`fill_rgb` 可替换为按目标材料生成的纹理。
`Template.uncropped_area` 记录变形后未裁剪面积，约束损失与出界损失一起检查。
`inject_anomaly` 接受 normal_image 或 gt_recolored_reference；原始标签图不能
仅靠改角色名变成正常外观图。

## 源上下文约束

`context_candidates` 根据源缺陷的材料占比过滤候选中心，与策略候选取交集；
`synthesize_sample` 对完整候选和实际修改像素分别运行 `context_check`，不能
用自定义 rules 替代。它比较材料占比、边界附近像素比例、距边界的最小距离；
距离按缺陷面积平方根归一化，不依赖类别名、颜色或固定像素尺度。图像外边缘
不作为材料边界；多材料关系及更复杂拓扑由策略另行补充。

`ContextPolicy` 默认材料比例容差 0.05、边界比例容差 0.25、相对距离容差 1.0，
边界带宽为缺陷尺度的 0.25。这些是两材料试样起点，须与策略一起记录并在
小样确认时固定，不为凑够数量逐步放宽。统计关系只是必要条件，仍须视觉复核。
源/目标材料同时反相不改变判断，前景名称本身不证明材料语义。

同一参考默认使用相同区域映射。跨参考须先核实尺度/材质对应，再显式传入
material_mapping=same/inverted 和 mapping_basis；字符串声明仅记录依据，
不是兼容性证明。改变承载材料的策略需独立验证其业务合法性，不把关系检查关掉。

## diff 的含义

默认不是合成图与 GT 的像素相减，而是：

`residual = gray(synthetic) - GaussianBlur(gray(synthetic), radius)`

`highpass_residual` 返回 float32 有符号残差；`highpass_display` 将绝对值按固定
gain 映射到 0..255，零响应为黑色。导出同时保留 signed .npy 和显示 PNG，
difference_kind=defect_gray_gaussian_highpass，radius/gain 进入计划与血缘。
这表现颗粒、划痕、氧化、垃圾等局部纹理变化，也会响应正常线路边缘，不能据此
直接判定缺陷或类别。修改范围另用合成图与重绘参考的变化掩码检查。
其他方向差分必须显式提供独立定义，不复用这个名称。

## 可运行示例与打包

[paired_reference_example.py](../../scripts/preparation/paired_reference_example.py)
是两材料、小批次候选的可执行驱动。计划字段如下，数值须依据实际样例确定：

| 字段组 | 内容 |
|---|---|
| 来源 | donor_id、donor_rgb、donor_cam；图像路径相对计划文件或为可访问绝对路径 |
| 任务 | category、annotation_note、category_definitions、source_split、purpose |
| 标注/分割 | annotation_box_1000、exclusion_margin、source_material_threshold、gt_foreground_min/max |
| alignment | radius、max_mismatch、min_pixels |
| reference | seed、min_material_pixels、boundary_margin、texture_std_cap |
| extraction | method、min_area、max_box_fraction、boundary_margin，appearance 另设 appearance_threshold |
| placement | relation、target_foreground、margin；allowed_margin 单独控制允许区边界 |
| structure_rule | operation(add/remove/appearance)、relation(edge/interior/any)、target_foreground、component_delta、margin |
| synthesis | min_area、max_area、max_clip_fraction、max_attempts、feather、highpass_radius、highpass_gain、visibility_threshold、min_visible_pixels |
| variants | 每项 seed、scale_xy、angle_deg、pair_transform |
| context_policy | material_fraction_tolerance、boundary_fraction_tolerance、clearance_tolerance、boundary_band_ratio；省略时将默认值写入计划 |
| review_min_iou | 复核定位通过阈值，默认 0.5；在调用模型前固定 |

示例仅作为配对试验，不承担通用全量编排；新策略可复用阶段函数。
purpose=method_validation 才允许未知 split 的方法回归，必须排除训练使用；
正式合成需已确认 train，沿用来源隔离、复用上限与小样确认。

平台只上传单个 Pipeline 和固定 runtime，新增模块不在自动上传名单。
[打包器](../../scripts/preparation/bundle_template_pipeline.py) 将 template_synthesis、
reference_synthesis 和驱动一起冻结为单文件并输出 SHA-256：

```bash
python <skill目录>/scripts/preparation/bundle_template_pipeline.py \
  <驱动.py> public_data/data-preparation/pipeline.py
```

纯生成用同一文件调用 df_run_pipeline / df_submit_pipeline，
model_profile=none、output_schema=artifacts。其他自定义代码须写入驱动；
计划和输入都必须在执行环境可访问，不能依赖开发机 Downloads。
输出 role 沿用 image/gt/diff/annotation/review/other；转换后的正常参考用 gt，
原始结构 GT 用 other 并标明语义，不新增公开 role 枚举。

## 固定视觉复核路径

复制 [image_synthesis_review.py](../../scripts/preparation/image_synthesis_review.py)
为复核 Pipeline，直接运行 df_run_pipeline，model_profile=vision、
output_schema=vision。它只导入 ImagePipelineConfig / run_image_pipeline_from_cli，
不要改写传输、认证或 DataFlow 算子。

输入 JSONL 每行：id、images（相对该 JSONL 所在目录）、image_labels、
user_prompt。模板审查提供标注源图、GT、提取掩码及局部对照；合成盲审提供
无框合成图、重绘正常参考、高通图、图像尺寸和完整业务类别定义，不泄露预期
类别、注入框或有答案含义的文件名。示例自动生成合成复核输入。
另提供源缺陷图、源正常结构与源掩码，检查材料承载与边界关系；源标注可以
展示，候选预期位置保持隐藏。类别定义不能把允许位置泛化到任意材料。

模型返回 reasoning 和 answer；answer 包含 category、像素 bbox_xyxy、
realistic、extra_anomalies、uncertain、context_consistent。用 `merge_review_outputs` 将候选、托管
输出及本轮 selected_ids 合并：按图片路径对应，读取现有 <answer> 输出包裹，
不按行号对齐。内部 `review_status` 比对类别、bbox IoU 和质量判断；
min_iou 在计划中确认。未选中 not_reviewed，选中但无有效输出 review_failed，
不一致/不确定 needs_review，均明确保留。通过视觉检查仍不自动标为训练就绪。
模板检查与合成检查分别记录；模板不合格时不扩量。详见
[Gap 驱动合成](gap-driven-synthesis.md) 与 [PCB 案例](template-synthesis-pcb.md)。

示例在调用模型前执行 `freeze_review_binding`，生成 review_binding.json，绑定
计划、代码哈希、样本与资产哈希、复核输入、抽检 ID 和 IoU 阈值；输入中加入
不含答案的 evidence ID。复核结果须包含同一 ID，资产或策略变化后旧结果失效。
这用于检测过期证据，不是对任意 Agent 代码的隔离或防篡改机制。

复核结束后用打包器打包
[finalize_synthesis_review.py](../../scripts/preparation/finalize_synthesis_review.py)，
以 model_profile=none、output_schema=artifacts 调用 df_run_pipeline；本地也可运行：

```bash
python <打包后的汇总.py> review_job.json reviewed.jsonl
```

review_job.json 只有 binding_path 和 review_output_path，均相对此文件。
汇总输出新的 reviewed.jsonl 和 reviewed.quality.json，不覆盖生成阶段或原始
模型输出。缺结果、旧 evidence ID、坐标越界/不匹配、缺上下文判断不能放行。
quality_status 同时要求程序检查和视觉通过；未抽检不升级，training_ready
始终为 false。最终说明引用汇总状态，不能将模型调用成功或单项 realistic
解释为质量合格。坐标转换须有实际变换证据，不自动尝试翻转来迎合预期框。
