# Label Studio XML 配置语法要点

## 依据与版本

本文与校验规则对齐 Label Studio 官方实现，出处：

- 校验函数：`label_studio/core/label_config.py:validate_label_config`
- Schema：`label_studio/core/utils/schema/label_config_schema.json`
- 端点：`label_studio/projects/api.py` → `LabelConfigValidateAPI`（`POST /api/projects/validate/`，
  免鉴权，通过 204 / 失败 400）与 `ProjectLabelConfigValidateAPI`
  （`POST /api/projects/<pk>/validate/`，200 返回 `config_essential_data_has_changed`）
- 路由：`label_studio/projects/urls.py` —— 只有上述两条，**没有 `/validate-config`**

本仓库对应的部署侧源码在 `~/PycharmProjects/label-studio-raw`。升级 Label Studio 后
重新核对上述文件，尤其是 schema 与 `_tag_attribute_validation`。

## 结构

一个 label_config.xml 的顶层是 `<View>`，内部可以嵌套 `<View>` 做布局。
核心分两类标签：

**Object 标签**（数据源，有 `name` 和 `value`）：
- `<Image name="xxx" value="$data_key"/>` — `$data_key` 对应 Task data 的字段
- `<Text name="xxx" value="$data_key"/>`
- `<Audio name="xxx" value="$data_key"/>`
- `<Video name="xxx" value="$data_key"/>`

**Control 标签**（标注控件，有 `name` 和 `toName`）：
- `<Choices name="..." toName="...">` — 单选/多选，子标签 `<Choice value="..."/>`
- `<RectangleLabels name="..." toName="...">` — 矩形标注，子标签 `<Label value="..."/>`
- `<Labels name="..." toName="..."/>` — 语义标签
- `<TextArea name="..." toName="..."/>` — 文本输入，可加 `perRegion="true"`
- `<BrushLabels>`、`<PolygonLabels>`、`<KeyPointLabels>` 等高级标注

## 官方校验查什么

`validate_label_config()` 依次做四件事：

1. XML 良构 + 能被 XML→JSON 转换，根元素是 `<View>`
2. JSON Schema 校验（**注意：`View.additionalProperties = true`，schema 不做标签白名单**，
   写一个不存在的标签也能过这一步 —— 能通过 ≠ 语义正确）
3. **全文** `name="..."` 查重（正则 `(?:^|[^\w])name="([^"]*)"`，Object 与 Control 一起查）
4. `toName` 必须在名字集合里，且**支持逗号分隔多值**（`toName="a,b"` 逐个检查）

## 本地校验比官方严格的两处

`scripts/validate_label_config.py` 与工具侧 `skill_helpers.py` 实现同一套规则，
其中两处**故意比官方严**，方向都是"把运行时才炸的问题提前"：

- 官方用正则扫原文，`name = "x"`（等号旁带空格）会漏检；本地解析文档，能查出来
- 官方只要求 `toName` 指向"某个名字"，指向另一个 Control 也算过；本地要求必须指向
  **Object** 标签，否则控件渲染时无数据可绑

## 注意

- Object 标签的 `value` 用 `$` 前缀引用 Task data 的 key
- 每个 Object 的 name 在整个 XML 中必须唯一
- 每个 Control 的 name 在整个 XML 中必须唯一
- Control 的 toName 必须指向一个已定义的 Object
- Control 可以写在它引用的 Object **之前**，顺序不影响校验
- 面向 PyroMind 数据集的配置另有一层硬契约（控件名与 toName 固定），见 SKILL.md
  「控件契约」一节；`validate_label_config.py --adapter avi_train` 可一并校验
