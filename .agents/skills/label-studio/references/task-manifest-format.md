# Label Studio Task Manifest 格式

## 单条 Task 的 JSON 格式

```json
{
  "data": {
    "defect_image": "https://console/label_studio/media?path=...",
    "diff_image": "https://console/label_studio/media?path=...",
    "gt_image": "https://console/label_studio/media?path=...",
    "defect_image_path": "/exports/.../defect.jpg",
    "diff_image_path": "/exports/.../diff.jpg",
    "gt_image_path": "/exports/.../gt.jpg",
    "sample_id": "sample_001"
  },
```

`*_image` 是浏览器渲染用的 URL，指向 portal 的 media 路由（每次渲染时才向存储换取带签名的对象 URL），因此有有效期。
`*_image_path` 是该图片在用户存储中的真实对象路径，用于导出回写和 URL 过期后重新签发；不要把 URL 当作路径使用。
  "predictions": [
    {
      "model_version": "vlm-v1",
      "score": 0.92,
      "result": [
        {
          "from_name": "quality_label",
          "to_name": "defect_image",
          "type": "choices",
          "value": {"choices": ["defect"]}
        },
        {
          "id": "finding_1",
          "from_name": "finding_category",
          "to_name": "defect_image",
          "type": "rectanglelabels",
          "value": {
            "x": 10.0,
            "y": 20.0,
            "width": 30.0,
            "height": 30.0,
            "rectanglelabels": ["开路"]
          }
        },
        {
          "id": "finding_1",
          "from_name": "finding_observation",
          "to_name": "defect_image",
          "type": "textarea",
          "value": {"text": ["线路存在断开"]}
        }
      ]
    }
  ]
}
```

## 批次文件

- `tasks-00001.json`、`tasks-00002.json` 等，每个文件是一个 JSON 数组
- 单批上限 500 条任务或 10 MB，先到哪个限制就切批
- `manifest.json` 记录所有批次的路径、数量和 SHA-256 哈希

## 坐标转换

Label Studio 使用百分比坐标（0-100）。导入时按下面的形状读 meta，
第一个能解析出来的就算数 —— 不同管线的产出形状确实不一样，所以都兼容：

| meta 里的写法 | 量纲 | 换算 |
|---|---|---|
| `bbox.x_min_norm` / `y_min_norm` / `x_max_norm` / `y_max_norm` | norm1000（0-1000） | 各除以 10 |
| `value.{x,y,width,height}`，或 finding 顶层平铺的同名键 | 百分比（0-100） | 直接用 |
| 同上，但值落在 0-1 之间 | 归一化 | 乘以 100 |
| `bbox` / `box` / `value` 是 `[x1, y1, x2, y2]` 列表 | norm1000 | 各除以 10 再算宽高 |
| `x1/y1/x2/y2` 或 `x_min/y_min/x_max/y_max` 键 | 按数值大小判断 | 同上 |

norm1000 转百分比：

```
x = x_min_norm / 10
y = y_min_norm / 10
width = (x_max_norm - x_min_norm) / 10
height = (y_max_norm - y_min_norm) / 10
```

超出画面的框裁到边界；**零面积、没有 `category`、或解析不出来的区域会被跳过** ——
Label Studio 不渲染没有标签的矩形，写进去也看不见。

## Region ID 关联

bbox 和 textarea 通过相同的 `id` 字段关联到同一个 region。
`id` 格式：`finding_1`、`finding_2` 等。
