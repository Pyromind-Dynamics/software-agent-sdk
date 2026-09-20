# 自定义绑定（field_map）与物化

**用户的诉求和内置绑定不一致时，改绑定，不要改 XML 去猜，也不要另造 adapter。**
写一份 JSON 声明，用 create 的 `field_map_path` 传入（workspace 相对路径）。

声明**按 section 整体替换**，没写的 section 继续用内置绑定。binding 只验
"能不能落地"，不要求控件必须叫某个固定名字 —— 底线是"预标注一定能渲染"。

## 三类绑定

| section | 作用 | 字段 |
|---|---|---|
| `images` | 图片文件 → `<Image>` 对象 | `field`、`source`（样本目录里的文件名/glob，或 `jsonl` 的行内点号路径）、`required` |
| `samples` | meta 整图字段 → 控件 | `field`、`control`、`type`（`choices`/`textarea`）、`synonyms`、`on_unmapped` |
| `regions` | meta 区域数组 → 矩形 | `source`、`control`、`label`、`geometry`、`unit`、`observation`、`observation_control` |

完整可运行写法见 `references/examples/field-maps/`。三种最常用：

- **参考图可选、A/B 类混跑**：把该图的 `required` 设成 `false`，没有这张图的
  样本**不会写进任务数据**，界面上那格自然空着 —— 不需要再拿缺陷图复制一份去
  顶替，也不会误导标注员。
  ```json
  {"images": [{"field": "gt_image", "source": "*_cam.bmp", "required": false}]}
  ```
- **自定义控件名**：`{"samples": [{"field": "quality", "control": "my_verdict",
  "type": "choices"}], "regions": [{"source": "findings", "control": "my_box",
  "label": "category"}]}` —— XML 里写 `<Choices name="my_verdict">` /
  `<RectangleLabels name="my_box">` 照样渲染预标注。导入和导出**读同一份声明**，
  所以自定义名字也能正常导出。
- **坐标量纲写清楚**：`"unit": "norm1000" | "percent" | "unit"`。不写（`auto`）
  时按数值大小推断；写了就**不再猜**，超出该量纲会**直接报错**而不是静默裁到
  边界。小于 1% 的百分比框正是被"推断"画错位的那类，量纲有把握就写上。
- **接 data-processing 的预打标产出**（`jsonl` 最常见的一支）：
  `references/examples/field-maps/pcb_prelabel.json` 可以直接当起点。structured
  产出的行自带 `source_images`（角色 → 图片 Storage 路径），图片按角色绑定，
  例如 `"source": "source_images.待检原图"`；角色名取自数据集的 `image_labels`，
  对不上时只改这一处 key，不要改数据。同理，判定字段和区域字段各绑一次即可，
  **产出文件原样导入**，不要再拼 `source_manifest.jsonl`。
  区域那一段按行内实际形状写：行里有 `regions`（每项含 `category`/`boxes`/`note`）
  就 `"source": "regions"`；行是顶层 `label`+`category`+`boxes`+`note` 就
  `"source": "boxes"`（`boxes` 是数据里已有键名之一，绑定直接读它），此时
  `note` 按整图字段绑到 `overall_note`。一个区域带多个 `boxes` 时每个框各出一个
  矩形，不必先摊平。`geometry` 声明的是**区域项里**优先读哪个键，不是行里的键：
  行内给裸坐标数组（`source` 直接指向 `boxes`）时，坐标在区域项里叫 `bbox`，
  这时写 `"geometry": "boxes"` 这个名字照样出框（读不到声明的键会回退到
  `bbox`/`boxes`/`box`/`value`）；拿不准就整个不写。

改了绑定要**重新 create**：映射内容参与项目标识，改了会得到新项目，不会复用
旧映射的旧项目。

## 源数据不是 adapter 布局时

先看能不能直接用 `jsonl`：**预处理/清洗产出的 JSONL 按 `references/adapters.md`
的行契约原样导入**（`adapter=jsonl`，字段名不同就声明绑定），不要再写一遍格式
转换脚本把它摊成样本目录 —— 那一步既慢又多一个出错的地方。

只有源既不是样本目录、也不是 JSONL（扁平文件列表、混合布局等）时才**物化**：
把源数据铺成 adapter 认的形状，再按路由回到 adapters 那一支。物化是确定性
转换，只改形状、不改业务语义：

- 目录名用源标识的可读化形式（例如 `a/b/c` → `a__b__c`），保证唯一；
- 图片按角色落成声明的 `source` 文件（如 `defect.bmp`、`*_cam.bmp`）；
- meta 写成 `meta_vlm.json` / `meta.json`，字段名与绑定声明一致；
- 图片路径和角色**来自源数据本身**，不要从目录名反推、也不要靠 `_cam` 之类
  后缀约定去猜。

物化不需要跑模型：它没有 LLM 调用，别按模型 pipeline 提交。

## 声明必须能正反两用

同一份声明同时被正向转换和导出反向读取。只改 XML 不改声明，会出现"导得进去、
标完导不出来"。校验时把声明一起传进自检脚本，检查的就是声明里的绑定而不是
adapter 的内置绑定（用法见 `references/operations.md`）。
