# 数据集适配器与控件契约

create 时按数据布局选一个 `adapter`（默认 `avi_train`），三种都有一套**内置
绑定**；内置绑定不满足需求时改用 `references/custom-bindings.md` 的声明式绑定，
不要另造 adapter、也不要靠改 XML 去猜。

## 三种输入布局

- `avi_train`：样本目录含 `meta_vlm.json`，字段为 `quality`（整图判定）与
  `findings`（区域数组，每项含 `category`/`observation`/`bbox`）。
- `aoi_export`：样本目录含 `meta.json`，字段为整图判定（`vlm_verdict`/`label`）
  与 `note`，通常不含图内坐标；`vlm_confidence` 会作为预测分数。
- `jsonl`：`dataset_path` 指向**一个 JSONL 文件**（不是目录），一行一个样本，
  行内自带元数据和图片路径。预处理/清洗的产出可以原样导入，不必先摊成样本目录。

前两种布局的样本目录**默认**都须包含 `defect.jpg`、`diff.jpg`、`gt.jpg`。文件名
不同、或某张图不是每个样本都有，用 `field_map` 覆盖
（`references/examples/field-maps/`）。

输入侧的最小可运行样例在 `references/examples/`：两个目录 adapter 各一组
**输入 → 期望输出**（meta、`label_config.xml`、`expected_predictions.json`），
对照方法见 `references/examples/README.md`；`jsonl` 产出的控件与 `avi_train`
是同一套，可以直接套它的 `label_config.xml`。转换是确定性的，自己产出的
`result` 应与 `expected_predictions.json` 逐字段一致；不一致就是问题所在，
不必等导入 Label Studio 才知道。

输出侧的完整 XML 模板另有两份：`references/pcb-avi-review.xml`
（`avi_train`，含缺陷区域标注）与 `references/aoi-export-review.xml`
（`aoi_export`，整图判定）。按任务裁剪，不要照抄业务标签。

### JSONL 行契约

一行就是一条任务，行的顶层键直接对应内置绑定的字段名：判定位 `quality`、区域
数组 `findings`，图片按 `defect_image`（必需）/`diff_image`/`gt_image`（可选）
三个键给**存储对象路径**（不是文件名，渲染地址由 create 现场签发）：

```json
{"id": "s0001", "quality": "defect", "defect_image": "/datasets/pcb/0001/defect.jpg",
 "findings": [{"category": "开路", "observation": "断线", "bbox": [100, 200, 400, 500]}]}
```

- 行内字段名与内置绑定不同时声明 `field_map`；绑定里的 `source` 对 `jsonl` 是
  **行内点号路径**，所以 `images.defect_image` 这类嵌套写法也支持。
- 图片值必须是**存储对象路径**（`/datasets/...`）。行里只有图片 ID 或文件名
  （如 `B0`）时，先让 data-processing 把路径写进行里 —— 路径来自数据本身，
  从名字反推是猜。
- 样本 id 依次取 `sample_id`、`id`，都没有时按行号（`line-2`）。
- 行内可以再带 `boxes` + 样本级 `category`：内置绑定会把 `category` 复制到每个
  框上（Label Studio 不渲染没有标签的矩形）。
- `jsonl` 数据集是**原地覆盖写**的，create 会把文件内容算进 `project_ref`：
  重新导出的文件落到新项目，不会复用上一版任务的旧项目。

## 内置绑定（不声明 field_map 时就是这套）

| adapter | 必须包含 |
|---|---|
| `avi_train` | `<Image name="defect_image" value="$defect_image"/>`<br>`<Choices name="quality_label" toName="defect_image">`<br>`<RectangleLabels name="finding_category" toName="defect_image">`<br>`<TextArea name="finding_observation" toName="defect_image" perRegion="true">` |
| `aoi_export` | `<Image name="defect_image" value="$defect_image"/>`<br>`<Choices name="quality_label" toName="defect_image">`<br>`<TextArea name="overall_note" toName="defect_image">` |
| `jsonl` | `<Image name="defect_image" value="$defect_image"/>`<br>`<Choices name="quality_label" toName="defect_image">` |

- 可以**增加**其他控件（`Header`、额外的 `<Choices>`、布局 `<View>` 都行），
  上面列出的必须在。
- `aoi_export` 和 `jsonl` 的矩形控件是**条件必需**的：数据里出现 `boxes`/
  `findings` 时配置里必须有 `finding_category`（和 `finding_observation`），
  否则框不会显示；没有坐标的纯整图项目不需要它们。
- `quality_label` 的 `<Choice>` 至少要有 `ok` 和 `defect`，这是 meta 判定值的
  映射目标。

三个 adapter 的判定值都按同一套解析归一化，不必先把数据洗成 XML 的写法：
先用控件自己的 `<Choice>` 值匹配（忽略大小写、全半角、空白与分隔符），再用
绑定声明的 `synonyms` 和内置判定词表（`NG`/`BAD`/`FAULT`/`TRUE`/`1`
→`defect`，`PASS`/`GOOD`/`FALSE_POSITIVE`/`0`→`ok`…）翻译，最后与控件值做一次
**唯一最近者**的近似匹配。`synonyms` 只是补充外部拼写，**不会遮蔽控件已有的
值**，所以一张只写了 `true`/`false` 的表不会把上游的 `defect` 顶掉。认不出时
处理不同：

- `avi_train` 和 `jsonl` 读的是自家 VLM / 预处理产物，认不出就**原样写进
  预标注**，create 会把这个值补进**项目上**的配置（控件名、布局不动），返回里
  带一句 `widened=<控件>:<值>`，所以那条预标注照常显示。调用者的 XML 文件不受
  影响。
- `aoi_export` 读的是外部检测系统的判定，认不出说明确实不知道，**留空**给人工
  标注。

绑定指向的控件在配置里根本没有时，create 返回 `warning=unmapped_controls:<控件>`
—— 扩值无处可加，要回去补上这个控件或改绑定的 `control`。

## 区域预标注（框）

三个 adapter 都会把数据里的坐标转成矩形预标注，写到 `finding_category`，
该区域的说明写到 `finding_observation`（`perRegion` 文字）。

- 坐标接受这几种写法，混用也行：`bbox` 的 `x_min_norm`/`y_min_norm`/
  `x_max_norm`/`y_max_norm`（norm1000）、`value` 或 finding 顶层平铺的
  `x`/`y`/`width`/`height`（0-100 百分比，也兼容 0-1 归一化）、以及
  `[x1,y1,x2,y2]` 列表。
- 一个区域可以带**多个框**：`bbox`/`boxes` 里放一组框（如
  `"boxes": [[100,200,400,500],[700,600,900,800]]`）时，每个框渲染成一个矩形，
  该区域的说明挂在每个矩形上。多个框合并成一条区域记录所以不必摊平数据。
- 每个区域必须有 `category`。只有坐标没有类别时**该区域会被跳过** ——
  Label Studio 不渲染没有标签的矩形，写进去也看不见。
- 区域标签按和整图判定同一套解析归一化（控件值 → `synonyms`/内置词表 → 唯一
  最近者）。解析不到、又不在控件 `<Label>` 列表里的标签，同样由 create 补进
  项目配置并报 `widened=<控件>:<标签>`，框正常画出来。（`aoi_export` 读外部
  检测结论，认不出的标签按丢弃处理，不扩值。）
- 绑定的 `source` 指向数据里根本没有的字段时，create 返回会带一句
  `warning=unmatched_regions:<字段>` —— 说明这些框一个都没建出来，要回去改
  绑定的 `source` 或上游数据的字段名。（`aoi_export` 不报，它的整图判定本来
  就大多数不带坐标。）
- 坐标超出画面会被裁到边界，零面积的框直接丢弃。
- `aoi_export` 的 `meta.json` 通常**不含图内坐标**（`vrs_xy`/`map_xy` 是机台
  坐标，不能当框用），这类项目就是纯人工标注。

## XML 语法依据

控件语法、`toName` 指向规则、官方校验查什么、以及本地校验比官方严的两处，
见 `references/label-studio-schema.md`。
