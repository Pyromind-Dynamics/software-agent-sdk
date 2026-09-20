# 可运行样例（输入 → 期望输出）

这两组目录是**输入侧**的最小样例。每组的形状与真实数据集一致，可以直接拿去对照
自己手上的数据：目录里有什么文件、meta 怎么写、转换后会得到什么。

```
examples/
├── avi_train/
│   ├── meta_vlm.json             # 输入：整图判定 + 缺陷区域
│   ├── defect.jpg / diff.jpg / gt.jpg   # 三个占位图（64x48，仅用于示意）
│   ├── label_config.xml          # 能通过契约校验的最小配置
│   └── expected_predictions.json # 输出：转换后应当产出的 predictions
├── aoi_export/
│   ├── meta.json                 # 输入：整图判定 + 备注（无坐标）
│   ├── defect.jpg / diff.jpg / gt.jpg
│   ├── label_config.xml
│   └── expected_predictions.json
└── field-maps/                   # 自定义绑定声明的写法（P1-1）
    ├── optional_reference_image.json
    ├── pcb_prelabel.json
    ├── renamed_controls.json
    └── pinned_unit.json
```

## 为什么要有 `expected_predictions.json`

它是**可自查的终点**：转换是确定性的，同一个 meta 一定产出同一份 `result`。
所以"我写得对不对"不需要等导入 Label Studio 才能知道 —— 比对这份文件即可。
反过来，如果界面上没看到预标注，也能用这份文件判断是"meta 写错了"还是
"配置/环境的问题"，不必靠反复试探去猜。

两组样例各自印证一件事：

| 样例 | 它证明什么 |
|---|---|
| `avi_train` | 一个 meta 同时给出**两个区域**，且用**两种不同的坐标写法**（norm1000 角点、percent 的 `value`），两个框都能正确产出 |
| `aoi_export` | 整图判定配 `note` → `quality_label` + `overall_note`，**没有任何区域控件**也能成项目 |

## 怎么用

**对照 meta 形状**：读自己数据集的 `meta_vlm.json` / `meta.json`，与这里的样例比字段。
字段名对不上时先看下面两张表，而不是先改代码去试。

**对照产出**：转换是确定性的。把样例的 meta 喂进去，得到的 `result` 应与
`expected_predictions.json` 逐字段一致 —— 若不一致，差异就是问题所在。

**校验配置**（技能目录里带了与服务端同规则的自检脚本）：

```bash
python <技能目录>/scripts/validate_label_config.py \
    examples/avi_train/label_config.xml --adapter avi_train
```

`<技能目录>` 必须是**绝对路径**（技能目录在 workspace 之外，`.agents/skills/...`
这类相对路径只在 read/write/edit 工具里有效）。

传了 `--field-map` 时校验的就是声明里的绑定，而不是 adapter 的内置绑定：

```bash
python <技能目录>/scripts/validate_label_config.py \
    label_config.xml --adapter avi_train --field-map field_map.json
```
