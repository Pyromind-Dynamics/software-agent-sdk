---
name: label-studio
description: >-
  使用 label_studio_project 工具将用户数据集创建为 Label Studio
  标注项目。用户要求创建标注项目、标注数据、打标、review 图片质检
  结果时使用。需要先 preview_dataset 理解数据再生成 XML。
---

# Label Studio 标注集成

把用户 Storage 里的数据集导入 Label Studio 做人工标注；认证由服务端管理，
不需要用户提供 Token 或 UID。

本文件只放路由、能力边界和控制面 SOP。数据形状、绑定写法、操作细节按路由表
按需读取对应 reference，不要提前全读；领域参考（如 PCB 裸板 AVI/AOI 的标签
含义与复判要求）不内联在本 skill，按路由表的负向边界到对应 skill 读取。

## 路由（先做这一步）

先判断源数据形状或需求能否命中已有 reference：命中就只读当前那一条；全部不命中
时走「自定义绑定」分支，不要另造 adapter。

| 源数据 / 需求 | 走向 | 读取 |
|---|---|---|
| 预处理/清洗产出的 JSONL（一行一个样本、行内自带图片路径，如 data-processing 的 `source_images`） | `adapter=jsonl` 直接导入，不再做格式转换 | references/adapters.md |
| 样本目录含 `meta_vlm.json`（`quality` / `findings`） | `adapter=avi_train`（内置绑定） | references/adapters.md |
| 样本目录含 `meta.json`（`vlm_verdict` / `note`，无 bbox） | `adapter=aoi_export`（内置绑定） | references/adapters.md |
| 形状命中上表，但要改控件名、图片槽位或坐标量纲 | 自定义绑定（`field_map_path`） | references/custom-bindings.md |
| 形状不在上表（混合布局、要第 4 张图、行内字段名与内置绑定不同） | 先自定义绑定；能力表外的直接说缺什么 | references/custom-bindings.md |
| 改已有项目的配置，或取回标注结果 | `update_config` / `export` | references/operations.md |

负向边界：领域标签语义与预打标产出（如 PCB AVI/AOI）→ data-processing，
领域参考的运行时路径和读取方式由那里的 case 文档给出；训练格式转换 →
data-processing 的 format-conversion；只想创建/管理沙箱 → sandbox。
Label Studio 服务端语义依据见 `references/label-studio-schema.md`，只在升级
Label Studio 后需要回看。

## 能力边界（先看这张表）

转换器**能写进预标注的东西就下面这些**，表外的一律不生成 ——
Label Studio 本身支持的控件（画笔、多边形、关键点）转换器不写预标注。
下表说的是**内置绑定**的产出；`jsonl` 的字段名本来就靠声明决定，它那一列
只是不写 `field_map` 时的默认行为：

| 能力 | avi_train | aoi_export | jsonl | 落到哪个控件 |
|---|---|---|---|---|
| 整图判定 | ✓ | ✓ | ✓ | `quality_label`（Choices） |
| 区域矩形 | ✓ | 有坐标时 ✓ | 有坐标时 ✓ | `finding_category`（RectangleLabels） |
| 区域文字（perRegion） | ✓ | 有坐标时 ✓ | 有坐标时 ✓ | `finding_observation`（TextArea） |
| 整图备注 | ✗ | ✓ | ✗（声明即可加） | `overall_note`（TextArea） |
| 非矩形区域（brush/polygon/keypoint） | ✗ | ✗ | ✗ | Label Studio 支持，**转换器不生成** |
| 每样本一套不同控件 | ✗ | ✗ | ✗ | 一次导入一套配置 |
| 第 4 张及更多图 | ✗ | ✗ | ✗ | 用 field_map 声明即可加 |

**用户要的东西不在这张表里时：直接说明当前不支持、缺哪个能力**，不要造探测
样本反复试探。每次试探都要重传素材、重建项目，而且表外的东西试多少轮都不会
出现 —— 早期真实对话因此在"框为什么不渲染"上耗费了数小时。

## 控制面 SOP

1. **探查**：`preview_dataset(dataset_path=...)` 先看清数据结构（样本目录 +
   meta 字段，或 JSONL 行内字段和图片路径）；需要对比多个样本时，把其余路径
   放进 `dataset_paths` 一次看完，不要一个样本调一次。
2. **路由**：按上表命中一条 reference 并读取；不命中则进入自定义绑定分支。
   数据形状与 `references/examples/` 不一致时，先核对字段名和坐标写法。
3. **生成配置**：按样本结构生成 `label_config.xml`；需要改绑定时同一层再写
   一份 `field_map.json`。绑定与控件名必须成对 —— 名字对不上不报错，只是标注员
   看不到预标注、导出时字段为空。
4. **校验**（可选）：自检脚本用法见 `references/operations.md`，跳过结论一致，
   create 会在转换**之前**做同样的校验并返回可读错误。
5. **创建**：`label_studio_project(operation="create", ...)`。
6. **交付**：给项目名、导入样本数，和一个能点的标注入口（格式见下）。

## 回复用户的链接格式

项目创建成功后交付标注入口时，**URL 必须写成能点的形式**：Markdown 链接
`[打开标注项目](https://...)`，或单独一行的裸 URL；**不要**用反引号把 URL 包
起来，也不要放进代码块 —— 代码跨度内的 URL 不做自动链接，用户点不动。

入口优先用工具返回的 `open_url`，交付时提醒用户存下这一条；它是登录入口，
登录态过期时会先回平台登录，再回到原项目。用户已经保存了 `project_url` 也一样
能用，不需要换链接。不要把登录态、SSO 或"链接会不会过期"展开成额外说明，
只给一个能点的入口。`project_ref`、`dataset_path` 这类**标识符**继续用行内代码。

交付时一并报出：项目名、导入样本数。需要时可说明哪些打标属性已可筛选
（分类控件的值会写进 task data，Data Manager 按这些列过滤）。

## 禁止

- 未 preview_dataset 就生成 XML
- 在 operation 参数中传递 UID、Token 或认证信息
- 用 terminal/curl 直接调用 Label Studio API
- 由 Agent 遍历完整数据集或生成 Manifest —— 批量由 create 负责
- 造探测样本反复试 meta 写法 —— 表里没有的能力试也不会出现
- 手工拼接或重塑上游产出（合并 `source_manifest.jsonl`、改字段名、摊平区域）来凑行
  契约 —— 形状对不上就用 `field_map` 声明绑定；行内确实缺字段就是上游契约缺口，
  回到 data-processing 补，不在这一侧写转换脚本
- 把交付给用户的 URL 包进反引号或代码块
- 试图"删了重建"：工具没有删除项目，参数写错就用同参数重跑 create
