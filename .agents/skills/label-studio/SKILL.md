---
name: label-studio
description: >-
  使用 label_studio_project 工具将用户数据集创建为 Label Studio
  标注项目。用户要求创建标注项目、标注数据、打标、review 图片质检
  结果时使用。需要先 preview_dataset 理解数据再生成 XML。
---

# Label Studio 标注集成

将用户 Storage 中的数据集导入 Label Studio 进行人工标注。
认证由服务端管理，不需要用户提供 Token 或 UID。

理解 PCB 裸板 AVI/AOI 的标签含义和复判要求时，按需读取共享的
领域参考：向文件读取工具传入 `knowledge/business-domain/pcb-avi-aoi.md`。
这是运行时知识库逻辑路径，不相对于本 skill 目录，无需拼接路径或查询环境变量。
保留源标签，
不将现场判定约定直接用作所有项目的分类或放行规则。读取参考不触发其他 skill 流程。

## 固定流程

1. **理解数据**：调用 preview_dataset(dataset_path=...) 分析目录结构、
   图片文件、meta 字段。需要对比多个样本时，把其余路径放进
   dataset_paths 一次看完（如同批 10_B1/10_B2/11_B1 的 meta.json），
   不要一个样本调一次。
2. **生成配置**：根据样本结构生成 label_config.xml。先读下面的
   「控件契约」，再参考范例：
   - references/pcb-avi-review.xml —— `avi_train`，含缺陷区域标注
   - references/aoi-export-review.xml —— `aoi_export`，整图判定
   - references/label-studio-schema.md —— 语法与官方校验依据
3. **校验 XML**（可选，快速自检）：技能目录带了自检脚本，规则与 create 在服务端跑的
   完全一致：
   `python <技能目录>/scripts/validate_label_config.py <label_config.xml> --adapter <adapter>`
   其中 `<技能目录>` 必须是**绝对路径** —— 技能目录在 workspace 之外，
   相对路径 `.agents/skills/...` 只在 read/write/edit 工具里有效，terminal 里会找不到文件。
   - **路径不确定就别折腾**：直接跳到第 4 步。create 会在转换**之前**做同样的校验并返回
     可读的错误，结论一致；这一步只省一次全量转换的时间。
   - 退出码：0 通过，1 不通过（打印原因），2 用法错误。
4. **创建项目**：调用 label_studio_project(operation="create", ...)。

## 控件契约

工具导入时把 meta 字段写成**预标注**（predictions），预标注按固定名字找控件。
名字对不上**不会报错**：项目照建，只是标注员看不到任何预标注、导出时字段为空。
所以这些标签必须原样存在：

| adapter | 必须包含 |
|---|---|
| `avi_train` | `<Image name="defect_image" value="$defect_image"/>`<br>`<Choices name="quality_label" toName="defect_image">`<br>`<RectangleLabels name="finding_category" toName="defect_image">`<br>`<TextArea name="finding_observation" toName="defect_image">` |
| `aoi_export` | `<Image name="defect_image" value="$defect_image"/>`<br>`<Choices name="quality_label" toName="defect_image">`<br>`<TextArea name="overall_note" toName="defect_image">` |

- 可以**增加**其他控件（`Header`、额外的 `<Choices>`、布局 `<View>` 都行），
  上面的必须在
- **不要改名**。create 会直接拒绝并报
  `does not match the '<adapter>' prediction contract`
- `quality_label` 的 `<Choice>` 至少要有 `ok` 和 `defect`，这是 meta 判定值的映射目标
- 两个 adapter 的判定值都按同一张同义词表归一化（`NG`/`BAD`/`FAULT`/`TRUE`→`defect`，
  `PASS`/`GOOD`/`FALSE_POSITIVE`→`ok`，忽略大小写）。认不出时的处理不同：
  - `avi_train` 读的是自家 VLM 的产物，认不出就**原样写进预标注**，并在 create 返回里带一句
    `warning=unmapped_quality:<值>` —— 看到它就把对应的 `<Choice value="...">` 补上，
    否则那条预标注在界面上不会显示。
  - `aoi_export` 读的是外部检测系统的判定，认不出说明确实不知道，**留空**给人工标注。
- 用户强烈要求自定义控件名时：那些名字拿不到预标注，要说明这会是纯人工标注项目

### 区域预标注（框）

两个 adapter 都会把 meta 里的坐标转成框预标注，写到 `finding_category`（矩形），
该区域的说明写到 `finding_observation`（`perRegion` 文字）。

- 坐标接受这几种写法，混用也行：`bbox` 的 `x_min_norm`/`y_min_norm`/`x_max_norm`/`y_max_norm`
  （norm1000）、`value` 或 finding 顶层平铺的 `x`/`y`/`width`/`height`（0-100 百分比，
  也兼容 0-1 归一化）、以及 `[x1,y1,x2,y2]` 列表（norm1000）
- 每个区域必须有 `category`。只有坐标没有类别时**该区域会被跳过** ——
  Label Studio 不渲染没有标签的矩形，写进去也看不见
- 坐标超出画面会被裁到边界，零面积的框直接丢弃

`aoi_export` 的 `meta.json` 通常**不含图内坐标**（`vrs_xy`/`map_xy` 是机台坐标，
不能当框用），这类项目就是纯人工标注。若 meta 里出现 `boxes`（norm1000 的
`[x1,y1,x2,y2]` 列表，配 `category`/`vlm_category`）或 `findings`，就会一并生成框预标注 ——
**这时配置里必须有 `finding_category` 和 `finding_observation`**，否则框不会显示。

## 回复用户的链接格式

项目创建成功后交付标注入口时，**URL 必须写成能点的形式**：

- 正确 —— Markdown 链接：`[打开标注项目](https://...)`
- 正确 —— 裸 URL 单独一行：`https://...`
- 错误 —— 用反引号把 URL 包起来。产品对话按 Markdown 渲染，
  代码跨度内的 URL 不做自动链接，用户点不动。
- 错误 —— 把 URL 放进代码块。

入口用工具返回的 `open_url`（SSO 直连），`project_url` 作为备用。
`project_ref`、`dataset_path` 这类**标识符**继续用行内代码 —— 它们不是给人点的。

交付时一并报出：项目名、导入样本数。

## 数据集适配器（adapter）

create 时按数据布局选择 `adapter`（默认 `avi_train`）：

- `avi_train`：样本目录含 `meta_vlm.json`（`quality`/`findings` 预标注字段）。
- `aoi_export`：样本目录含 `meta.json`（整图判定字段，无 bbox）。
  `vlm_verdict`/`label` 映射为 `quality_label`（defect/ok），
  `note` 映射为 `overall_note`，`vlm_confidence` 作为预测分数。

两种布局的样本目录都须包含 `defect.jpg`、`diff.jpg`、`gt.jpg`；
无法映射的判定值会留空给人工标注。

## 禁止

- 未 preview_dataset 就生成 XML
- 在 operation 参数中传递 UID、Token 或认证信息
- 用 terminal/curl 直接调用 Label Studio API
- 由 Agent 遍历完整数据集或生成 Manifest
- 改上表中固定的控件名或 toName
- 把交付给用户的 URL 包进反引号或代码块（会被渲染成代码，用户点不动）

## 修改已有项目

调用 label_studio_project(operation="get", project_ref=...) 获取当前
配置 → 在 workspace 中修改 XML → 校验 → 调用
label_studio_project(operation="update_config", ...)。
如果新 XML 删除了已有标注使用的控件名，Tool 会拒绝更新。

## 导出标注结果

调用 label_studio_project(operation="export", project_ref=...)。
Tool 自动将标注结果转回 PyroMind 格式并保存到用户 Storage。

## 图片链接

任务数据里的图片地址是**导入时签发的**，Label Studio 不会重签。但 portal 对已签发的
地址**不设到期时间**：只要该地址指向的账号仍然可用，旧项目里的图就一直能渲染，
标注员不会因为项目放久了而看到裂图。

因此**不要**向用户提"图片有效期""多久后失效""需要刷新"这类内容 —— 不存在这件事。
`refresh_media` 仍然保留（原地重签，任务 ID、标注与预测都不受影响，重复执行也安全），
但它是可选的维护动作，不必按期执行。

## 没有删除操作

工具**不提供删除项目**。建错了、或者用户要推倒重来，只能在 Label Studio
界面里删（把 open_url 给用户，让他自己删）。
如果只是参数写错，先用同参数重跑 create（幂等，会复用已有项目），
不要试图"删了重建"。确实需要另一个项目时，传一个不同的 `idempotency_key`。
