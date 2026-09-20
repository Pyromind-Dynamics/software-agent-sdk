# 自定义绑定（field_map）与物化

**用户的诉求和内置绑定不一致时，改绑定，不要改 XML 去猜，也不要另造 adapter。**
写一份 JSON 声明，用 create 的 `field_map_path` 传入（workspace 相对路径）。

声明**按 section 整体替换**，没写的 section 继续用内置绑定。binding 只验
"能不能落地"，不要求控件必须叫某个固定名字 —— 底线是"预标注一定能渲染"。

## 三类绑定

| section | 作用 | 字段 |
|---|---|---|
| `images` | 图片文件 → `<Image>` 对象 | `field`、`source`（文件名或 glob）、`required` |
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

改了绑定要**重新 create**：映射内容参与项目标识，改了会得到新项目，不会复用
旧映射的旧项目。

## 源数据不是 adapter 布局时

内置 adapter 只认"已经铺好的样本目录"（见 `references/adapters.md`）。源是
JSONL、扁平文件列表或混合布局时，先做一次**物化**，把源数据铺成 adapter 认的
形状，再按路由回到 adapters 那一支。物化是确定性转换，只改形状、不改业务语义：

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
