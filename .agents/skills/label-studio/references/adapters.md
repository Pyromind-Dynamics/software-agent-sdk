# 数据集适配器与控件契约

create 时按数据布局选一个 `adapter`（默认 `avi_train`），两者都有一套**内置
绑定**；内置绑定不满足需求时改用 `references/custom-bindings.md` 的声明式绑定，
不要另造 adapter、也不要靠改 XML 去猜。

## 两种布局

- `avi_train`：样本目录含 `meta_vlm.json`，字段为 `quality`（整图判定）与
  `findings`（区域数组，每项含 `category`/`observation`/`bbox`）。
- `aoi_export`：样本目录含 `meta.json`，字段为整图判定（`vlm_verdict`/`label`）
  与 `note`，通常不含图内坐标；`vlm_confidence` 会作为预测分数。

两种布局的样本目录**默认**都须包含 `defect.jpg`、`diff.jpg`、`gt.jpg`。文件名
不同、或某张图不是每个样本都有，用 `field_map` 覆盖
（`references/examples/field-maps/`）。

输入侧的最小可运行样例在 `references/examples/`：每个 adapter 一组
**输入 → 期望输出**（meta、`label_config.xml`、`expected_predictions.json`），
对照方法见 `references/examples/README.md`。转换是确定性的，自己产出的
`result` 应与 `expected_predictions.json` 逐字段一致；不一致就是问题所在，
不必等导入 Label Studio 才知道。

输出侧的完整 XML 模板另有两份：`references/pcb-avi-review.xml`
（`avi_train`，含缺陷区域标注）与 `references/aoi-export-review.xml`
（`aoi_export`，整图判定）。按任务裁剪，不要照抄业务标签。

## 内置绑定（不声明 field_map 时就是这套）

| adapter | 必须包含 |
|---|---|
| `avi_train` | `<Image name="defect_image" value="$defect_image"/>`<br>`<Choices name="quality_label" toName="defect_image">`<br>`<RectangleLabels name="finding_category" toName="defect_image">`<br>`<TextArea name="finding_observation" toName="defect_image" perRegion="true">` |
| `aoi_export` | `<Image name="defect_image" value="$defect_image"/>`<br>`<Choices name="quality_label" toName="defect_image">`<br>`<TextArea name="overall_note" toName="defect_image">` |

- 可以**增加**其他控件（`Header`、额外的 `<Choices>`、布局 `<View>` 都行），
  上面列出的必须在。
- `aoi_export` 的矩形控件是**条件必需**的：meta 里出现 `boxes`/`findings` 时
  配置里必须有 `finding_category`（和 `finding_observation`），否则框不会显示；
  没有坐标的纯整图项目不需要它们。
- `quality_label` 的 `<Choice>` 至少要有 `ok` 和 `defect`，这是 meta 判定值的
  映射目标。

两个 adapter 的判定值都按同一张同义词表归一化（`NG`/`BAD`/`FAULT`/`TRUE`
→`defect`，`PASS`/`GOOD`/`FALSE_POSITIVE`→`ok`，忽略大小写），认不出时处理
不同：

- `avi_train` 读的是自家 VLM 的产物，认不出就**原样写进预标注**，并在 create
  返回里带一句 `warning=unmapped_quality:<值>` —— 看到它就把对应的
  `<Choice value="...">` 补上，否则那条预标注在界面上不会显示。
- `aoi_export` 读的是外部检测系统的判定，认不出说明确实不知道，**留空**给人工
  标注。

## 区域预标注（框）

两个 adapter 都会把 meta 里的坐标转成矩形预标注，写到 `finding_category`，
该区域的说明写到 `finding_observation`（`perRegion` 文字）。

- 坐标接受这几种写法，混用也行：`bbox` 的 `x_min_norm`/`y_min_norm`/
  `x_max_norm`/`y_max_norm`（norm1000）、`value` 或 finding 顶层平铺的
  `x`/`y`/`width`/`height`（0-100 百分比，也兼容 0-1 归一化）、以及
  `[x1,y1,x2,y2]` 列表。
- 每个区域必须有 `category`。只有坐标没有类别时**该区域会被跳过** ——
  Label Studio 不渲染没有标签的矩形，写进去也看不见。
- 坐标超出画面会被裁到边界，零面积的框直接丢弃。
- `aoi_export` 的 `meta.json` 通常**不含图内坐标**（`vrs_xy`/`map_xy` 是机台
  坐标，不能当框用），这类项目就是纯人工标注。

## XML 语法依据

控件语法、`toName` 指向规则、官方校验查什么、以及本地校验比官方严的两处，
见 `references/label-studio-schema.md`。
