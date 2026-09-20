# Label Studio 集成优化方案（可执行版）

配套阅读：`.pr/label-studio-friction-analysis.md`（根因分析）
目标：把「169 次单文件上传 / 4h50m 穷举试探 / 用户觉得僵硬」这三件事各自对症下药。

---

## 一、先划边界

**这次要解决的**（都是分析报告里已定案的根因）：

| # | 优化点 | 对应根因 | 改代码 |
|---|---|---|---|
| P0-1 | 能力边界显式化（能力矩阵） | 根因 1 | ✗ 纯文档 |
| P0-2 | 每个 adapter 配可运行样例 | 根因 2 | ✗ 纯文档 |
| P1-1 | **映射声明**替代固定枚举 | 根因 3 + skill 第 3/4/5 条 | ✓ 核心改动 |
| P2-1 | 批量上传 | 根因 4 | ✓ 小改 |
| P2-2 | preview_dataset 超时/分批 | 根因 5 | ✓ 小改 |

**这次不碰的**：镜像重打与部署、`/tmp` 与 heredoc（根因 6，属平台侧）、媒体 token 链路。

---

## 二、P0-1 能力边界显式化（不改代码，收益最大）

### 为什么这条排第一

线上 4h50m 的成因不是模型笨，是**它看不见能力边界**：`_region_results` 这个能力在当天 20:01 才进代码，
而对话期间 agent 只能靠造 9 个探测样本反推"这工具到底能不能画框"。**能力缺失不可怕，可怕的是边界不可见。**

### 落地：两张表，写进 SKILL.md + tool description

**表一 —— 控件契约（已有，保留）**

**表二 —— 能力矩阵（新增，核心）**

| 能力 | avi_train | aoi_export | 落到哪个控件 |
|---|---|---|---|
| 整图判定预标注 | ✓ | ✓ | `quality_label` (Choices) |
| 区域**矩形**预标注 | ✓ | 有坐标时 ✓ | `finding_category` (RectangleLabels) |
| 区域文字（perRegion） | ✓ | 有坐标时 ✓ | `finding_observation` (TextArea) |
| 整图备注 | ✗ | ✓ | `overall_note` (TextArea) |
| 第 4 张图（参考图并排） | ✗ | ✗ | — 需映射声明（见 P1-1） |
| 非矩形区域（brush/polygon/keypoint） | ✗ | ✗ | Label Studio 支持，**converter 不生成** |
| 每样本不同控件集 | ✗ | ✗ | — 需映射声明 |

配套一条**行为规则**（直接消灭穷举）：

> 用户要的东西不在能力矩阵里时，**直接说明当前不支持并指出缺哪个能力**，
> 不要造探测样本反复试探。探测用的每一轮都要重传素材、重建项目。

### 为什么这能省下几小时

agent 拿到"能力矩阵"后，遇到"框不渲染"时的第一次判断就是：
「这个 adapter 能不能生成矩形 → 能 → 那问题在 meta 字段或 XML」，
而不是「我来试试第 7 种 schema 写法」。

---

## 三、P0-2 可运行样例（不改代码）

现状：`references/` 下只有 **输出侧** 的 `pcb-avi-review.xml` / `aoi-export-review.xml`，
以及导出格式 `task-manifest-format.md`。**输入侧的 `meta_vlm.json` 一个样例都没有** ——
agent 原话：「技能目录里没有 `meta_vlm.json` 的输入样例，只有 task-manifest（导出格式）」。

于是它拿**文件大小**反推图片语义（`gt.jpg: 4.6KB → much smaller → cam 参考图`）。

### 落地：每个 adapter 一组"输入 → 期望输出"最小样例

```
.agents/skills/label-studio/references/examples/
├── avi_train/
│   ├── meta_vlm.json            # 真实形状：quality / findings[].{bbox,category,observation}
│   ├── label_config.xml         # 能通过契约校验的最小 XML
│   ├── defect.jpg / diff.jpg / gt.jpg   # 三个占位小图（几 KB 即可）
│   └── expected_predictions.json        # 转换后应当产出的 predictions 结果
└── aoi_export/
    ├── meta.json                # vlm_verdict / note / boxes
    ├── label_config.xml
    ├── defect.jpg / diff.jpg / gt.jpg
    └── expected_predictions.json
```

关键在 `expected_predictions.json`：它给 agent 一个**可自查的终点** ——
"我产出的 result 数组应该长这样"，而不是"我产出的东西渲染没渲染"（后者要等导入 LS 才知道）。

顺带把 SKILL.md 里那句含糊的
「两种布局的样本目录都须包含 `defect.jpg`、`diff.jpg`、`gt.jpg`」（`:106-107`）
改成指向样例文件的明确引用。

---

## 四、P1-1 映射声明替代固定枚举（核心改动）

这是"僵硬"的正解。以下先把硬编码点列全，再给设计。

### 4.1 现状：同一套名字被写死在 6 个地方

| 位置 | 写死了什么 |
|---|---|
| `converter.py:46` `_SUPPORTED_ADAPTERS` | adapter 只能是 `avi_train` / `aoi_export` 二选一 |
| `converter.py:49-53` `_MEDIA_FIELDS` | 槽位↔文件名绑定：`defect_image←defect.jpg` 等三对 |
| `converter.py:394-402` | 三个槽位**强制存在**，缺一个抛 `ConversionError` |
| `converter.py:432/458/471/527/552` | 生成 predictions 时的 `from_name` / `to_name` 字面量 |
| `converter.py:776/780/794/798` | **导出侧**反向读 `from_name` 同样是字面量 |
| `skill_helpers.py:40-51` | `REQUIRED_MEDIA_OBJECT` + `_REQUIRED_CONTROLS` 白名单 |

用户 10:30 那句「**为什么一定要增强适配器，你可以不按照这个来吧**」，
就是被这张表逼出来的：他的需求落在 `avi_train`/`aoi_export` 之外，而表里没有第三个格子。

### 4.2 目标：把「必须叫这个名字」换成「必须给出合法绑定」

**契约不取消，只是从"名字固定"降级为"绑定必须能落地"。**
「预标注一定渲染」这条底线继续守住 —— 守的方式是校验绑定，而不是锁死名字。

### 4.3 载体选型：独立 JSON 文件，用 `field_map_path` 传入

```python
# definition.py, LabelStudioProjectAction 新增
field_map_path: str | None = Field(
    default=None,
    description=(
        "Workspace-relative JSON path declaring how meta fields and image "
        "files bind to label-config controls. Omit to use the adapter's "
        "built-in default bindings."
    ),
)
```

选它的理由（对比另外两条路）：

- **不内联进 action 参数**：长 JSON 由模型手写易错，且落不了 workspace 就没法本地校验、没法进 manifest。
- **不塞进 XML**：Label Studio 的校验会拒绝未知标签/属性，会把原生语义搞脏。
- **与 `label_config_path` 对称**：一个 XML 描述"界面有什么控件"，一个 JSON 描述"数据怎么填控件"，
  两者成对出现，agent 好记，也能被同一个校验脚本一次查完。

### 4.4 数据模型（放 `models.py`）

只做三种绑定，够覆盖现有两个 adapter + 用户已知的三个新需求（参考图 / 混合样本 / 自定义控件名）。

```python
class ImageBinding(BaseModel):
    """一张图 → 一个 <Image> 对象。"""
    field: str                      # task data 字段名，也是 <Image name="...">
    source: str                     # 样本目录内文件名或 glob："defect.jpg" / "*_cam.bmp"
    required: bool = True           # False: 缺图就跳过该字段，不抛错

class SampleFieldBinding(BaseModel):
    """整图字段 → 控件（每样本一条预标注）。"""
    field: str                      # meta 点路径："quality" / "note" / "fire_type"
    control: str                    # 控件名
    type: Literal["choices", "textarea", "number", "rating", "taxonomy"]
    synonyms: dict[str, str] = {}   # 值归一化表；空则原样写入
    on_missing: Literal["skip", "keep"] = "skip"

class RegionBinding(BaseModel):
    """区域数组 → 矩形 + 附属控件。"""
    source: str                     # meta 里的数组路径："findings" / "boxes"
    control: str                    # 矩形控件名 → RectangleLabels
    label: str                      # 区域内取类别的键："category"
    geometry: str | None = None     # 区域内容器键（"bbox"/"value"）；None = 自动探测
    unit: Literal["auto", "norm1000", "percent", "unit"] = "auto"
    label_synonyms: dict[str, str] = {}
    observation: str | None = None          # 区域内文字键："observation"
    observation_control: str | None = None  # → TextArea

class FieldMap(BaseModel):
    version: int = 1
    images: list[ImageBinding] = []
    samples: list[SampleFieldBinding] = []
    regions: list[RegionBinding] = []
```

### 4.5 默认映射 = 现有行为（这是向后兼容的关键）

```python
DEFAULT_FIELD_MAPS: dict[str, FieldMap] = {
    "avi_train": FieldMap(
        images=[
            ImageBinding(field="defect_image", source="defect.jpg"),
            ImageBinding(field="diff_image",   source="diff.jpg"),
            ImageBinding(field="gt_image",     source="gt.jpg"),
        ],
        samples=[
            SampleFieldBinding(field="quality", control="quality_label",
                               type="choices", synonyms=_QUALITY_CHOICE_SYNONYMS),
        ],
        regions=[
            RegionBinding(source="findings", control="finding_category", label="category",
                          observation="observation", observation_control="finding_observation"),
        ],
    ),
    "aoi_export": FieldMap(
        images=[ ...同上三张... ],
        samples=[
            SampleFieldBinding(field="vlm_verdict", control="quality_label",
                               type="choices", synonyms=_QUALITY_CHOICE_SYNONYMS),
            SampleFieldBinding(field="note", control="overall_note", type="textarea"),
        ],
        regions=[
            RegionBinding(source="findings", control="finding_category", label="category"),
            RegionBinding(source="boxes",    control="finding_category", label="category",
                          unit="norm1000"),
        ],
    ),
}
```

**不传 `field_map_path` ⇒ 走 `DEFAULT_FIELD_MAPS[adapter]` ⇒ 行为与今天完全一致。**
现有 169 个测试应当全部继续通过 —— 这就是这次改动的安全垫。

### 4.6 用户在三个场景里要写什么（这就是"自由度"）

**场景 A：参考图并排显示，且 A 类样本没有它**（当前必须复制缺陷图顶替 `gt.jpg`）

```json
{
  "images": [
    {"field": "defect_image", "source": "defect.jpg"},
    {"field": "gt_image", "source": "*_cam.bmp", "required": false}
  ]
}
```
`required: false` ⇒ 没有 cam 图时**该字段不写进 task data**，XML 里 `value="$gt_image"` 的 `<Image>` 自然空白。
→ 语义污染消失（缺陷图不再冒充参考图），A/B 类混跑不再需要人肉分拣。

**场景 B：自定义控件名**

```json
{
  "samples": [{"field": "quality", "control": "my_verdict", "type": "choices",
               "synonyms": {"ng": "defect", "pass": "ok"}}],
  "regions": [{"source": "findings", "control": "my_box", "label": "category"}]
}
```
XML 里写 `<Choices name="my_verdict">` / `<RectangleLabels name="my_box">` 即可 —— 预标注照样渲染。
→ 这就是报告里说的「契约从『必须叫这个名』变成『必须给出合法映射』」。

**场景 C：坐标量纲写清楚（顺带治掉量纲歧义）**

```json
{"regions": [{"source": "findings", "control": "finding_category",
              "label": "category", "geometry": "bbox", "unit": "norm1000"}]}
```
`unit != "auto"` 时**不再按数值大小猜**，超出该量纲范围直接报错而不是静默裁剪。
（`_scaled_geometry:128-147` 的现状是：`0.5/0.5/0.3/0.3` 这种小于 1% 的百分比框会被当成 unit 归一化再 ×100，
落到画面正中。这是静默错位，声明式写法能根治。）

### 4.7 校验规则（替代 `_check_adapter_contract`）

`skill_helpers.py:142-197` 的 `_check_adapter_contract` 改成 `check_field_map(field_map, xml_controls, objects)`：

1. `samples[].control` 必须存在且类型一致（`choices` → `<Choices>`，`textarea` → `<TextArea>` …）
2. `regions[].control` 必须存在且是 `<RectangleLabels>`；`observation_control` 必须存在且是 `<TextArea>`
3. `images[].field` 必须存在且是图片类对象（`Image`/`Video`）；`required=True` 的图缺文件才报错
4. 所有绑定的 `toName` 必须指向图片对象
5. `unit != "auto"` 时校验数值落在该量纲内

报错文案从

> `does not match the '<adapter>' prediction contract: missing <RectangleLabels name="finding_category">`

变成

> `binding 'my_box' → <RectangleLabels> not found in label_config.xml; samples[].control 'my_verdict' is <TextArea> but declared type is 'choices'`

即：**错误指向绑定本身，而不是指向一个必须存在的名字。**

### 4.8 幂等性必须一起改（否则改了映射不生效）

`executor.py:178-186` 的 `project_ref` 目前只 hash XML：

```python
config_hash = hashlib.sha256(xml_content).hexdigest()
project_ref = _project_ref(..., config_hash, action.idempotency_key)
```

**必须把 field_map 的内容 hash 一起算进去**，理由：
`executor.py:190-198` 见到 `status == "READY"` 就直接复用已有项目 —— 若只改映射不改 XML，
会命中这个短路分支，用着**旧映射**却以为生效了。这类"看起来做了其实没做"最容易骗人。

同时 manifest（`models.py` 的 `ManifestData`）要加 `field_map_hash` 与 `field_map_path`，
供导出时反向取用。

### 4.9 导出侧同步（`converter.py:736-800`）

`LabelStudioToAVITrainConverter` 现在反向读的四个 `from_name` 是字面量：

```python
if from_name == "quality_label": ...        # :776
elif from_name == "finding_category": ...   # :780
elif from_name == "finding_observation": ...# :794
elif from_name == "overall_note": ...       # :798
```

改成**用同一份 FieldMap 查表**（`control → field` 反向 index）。
不改这里就会出现最坏情况：**正向能导进去，标完导不出来** —— 控件名自定义了，导出读不认。
映射声明自始至终只有一份，正反向共用，这是这套设计的关键约束。

### 4.10 影响面清单（改动前先过一遍）

| 文件 | 改什么 |
|---|---|
| `label_studio/models.py` | 新增 `FieldMap` / `ImageBinding` / `SampleFieldBinding` / `RegionBinding`；`ManifestData` 加 `field_map_hash` |
| `label_studio/converter.py` | `_MEDIA_FIELDS` → 从 FieldMap 取；`_build_task` / `_build_predictions` / `_build_aoi_predictions` / `_region_results` 的 `from_name`+`to_name` 改为绑定驱动；`_geometry_from` 支持显式 `unit`；导出侧查表 |
| `label_studio/skill_helpers.py` | `_REQUIRED_CONTROLS` → `DEFAULT_FIELD_MAPS`；`_check_adapter_contract` → `check_field_map` |
| `label_studio/definition.py` | `LabelStudioProjectAction` 加 `field_map_path`；`TOOL_DESCRIPTION` 加能力矩阵 |
| `label_studio/executor.py` | `_handle_create` 读 field_map、并入 hash、进 manifest |
| `.agents/skills/label-studio/SKILL.md` | 控件契约改为「绑定规则」；加能力矩阵；禁止清单改造 |
| `tests/tools/label_studio/test_validate_config.py` | `test_required_controls_matches_what_the_converter_writes:207` 等要跟着改语义 |
| `scripts/validate_label_config.py` | 支持 `--field-map` 入参 |

---

## 五、P2-1 批量上传（169 次 → 个位数）

现状：`UploadFileToPyromindAction`（`pyromind_dataset/definition.py:2111-2126`）只有
`file_path: str` 单文件 + `target_dir`。所以"重传 18 个文件"只能点 18 次。

### 落地

```python
file_path: str | None = None          # 保留，单文件路径
file_paths: list[str] = []            # 新增：一次多个文件
dir_path: str | None = None           # 新增：递归整个目录
target_dir: str | None = None
```

- `dir_path` 递归时保持**工作区相对目录结构**（避免落成同名文件互相覆盖 —— 这正是本地对话里
  `1-0805Y/105/B1` 与 `1-0805Y/24/B1` 撞车那类错）
- 返回 `storage_paths: dict[str, str]`（本地路径 → storage 路径），保留 `storage_path` 兼容单文件
- 上限：单次 ≤ 200 个文件 / ≤ 200MB，超限明确报错并提示分批（避免一次调用把会话拖死）

注意这是**通用工具**（不止 Label Studio 用，还服务于 MetricsConfigBuilder 等），
接口要通用，不要塞 LS 专用逻辑。

---

## 六、P2-2 preview_dataset 超时与分批

现象：`preview_dataset timed out after 60 seconds. Narrow dataset_path to a specific file or a smaller directory.`
（线上 4 次、本地 1 次）。大目录不可预览 ⇒ agent 没法"先看清数据"，只能边做边试。

先做两件小事（`pyromind_dataset/definition.py`）：

1. **超时归因**：这是 executor `timeout`（默认 30/60s）还是调用侧超时 ——
   **这一步需要先定位**（本次未查证 `timed out after 60 seconds` 的抛出来源，不要照抄结论）。
2. **目录模式降级返回部分结果**：规模超阈值时不要整体失败，改为"预览前 N 个并明确告知被截断"，
   让 agent 至少能看到形状。`_cap_entry_listing`（`:174-189`）已有截断先例，思路一致。

---

## 七、落地顺序与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| 1 | P0-1 能力矩阵 + P0-2 样例（纯文档） | 拿一条"框不渲染"的诉求跑一遍，agent 不再造探测样本 |
| 2 | P1-1 映射声明（默认映射等价今天的行为） | 现有 169 个测试全绿；新增 A/B/C 三场景用例 |
| 3 | P1-1 导出侧共用映射 | 自定义控件名跑一次「create → 标一条 → export」，字段回填正确 |
| 4 | P2-1 批量上传 | 18 文件一次调用完成，`storage_paths` 覆盖全部 |
| 5 | P2-2 preview 降级 | 一个超大目录返回截断预览而非超时报错 |

**阶段 2 的安全垫**：`DEFAULT_FIELD_MAPS` 与今天的 `_REQUIRED_CONTROLS` + `_MEDIA_FIELDS` 语义一一对应，
先让新代码在"不传 field_map"时跑出与今天逐字节相同的 predictions（可写一个对照测试），再开放自定义。

---

## 八、关于"要不要顺手做更激进的方案"

备选方案 B：不改接口，改成「XML 里第一个 `RectangleLabels` 就是区域控件，`<Image>` 按出现顺序对应三张图」。
**不推荐**：隐式约定会重现"静默不渲染"的老毛病 —— XML 里调换一下控件顺序，预标注就错位，
而错位是不报错的。显式声明多一层写文件的成本，换的是错误可见。
