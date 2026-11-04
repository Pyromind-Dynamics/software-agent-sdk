# 数据与配置

`input_path` 是 Storage JSONL 绝对路径，每行是独立测试 case。
模型来源只写在评测配置 JSON 的 model_source 中，不在工具参数里新增字段。
省略或 `{"type":"storage"}` 时，inference.model_path 是 Storage 绝对路径；
`{"type":"clone"}` 时是中心仓模型标识；
`{"type":"download","download_source":"modelscope"}` 时是外部仓模型名称。
download_source 支持 huggingface（默认）、modelscope。
容器挂载前缀、Clone 目标目录和 Download 缓存目录由工具处理；
准备节点的 model_path 输出绑定下游，不把模型仓库标识伪装成 Storage 路径。

`dataset_config_path` 指向字段映射 JSON。映射值是样本的顶层字段名，
不支持在这里写 `gt.result` 等点路径。

```json
{"user_prompt_field":"prompt","reference_field":"ground_truth","media_field":"images","id_field":"id"}
```

| 配置项 | 对应样本值与用途 |
|---|---|
| `user_prompt_field` | 用户提示词；与媒体组成 user 消息，媒体在前、文本在后 |
| `system_prompt_field` | 可选系统提示词；单独组成 system 消息 |
| `messages_field` | 完整 OpenAI messages 列表；设置后直接使用其中的角色、文本和媒体，不再拼接上述提示词或 `media_field` |
| `reference_field` | 必填 GT 字段；取出的值供所有 rubric 共用，不自动发送给模型 |
| `media_field` | 图片路径字符串或路径列表 |
| `id_field` | 样本标识；每个 case ID 必须唯一，没有 ID 时使用行号 |

GT 可为标量、对象或含 JSON 对象的文本；其解析方式及 rubric 字段路径见
[Rubric 编写](rubric-authoring.md)。只有一个 `reference_field`，分类与框等多个指标
需要的真值应位于同一 GT 对象中，各 rubric 用自己的 `reference_path` 取值。

图片列表默认保留原顺序；只有需要重排时设置 `image_order`，值为文件名列表，
按路径的 basename 匹配，未列出的图片排在后面并保持原有相对顺序。
相对图片路径基于 JSONL 父目录，只有确需另设根目录才传 media_base_dir。
不支持动态 request builder、HTTP 媒体或运行时加载自定义评分模块。

评测配置固定 mode=agent_rubric，rubrics 非空。pass_threshold 和 rubric_pass_threshold
默认 0.7。可选 generation 对象包含 workers=4、max_tokens=1024、temperature=0、
timeout_seconds=240、max_retries=3、limit=0；limit=0 表示全量。
配置中不得包含 API Key、Cookie 或运行时环境变量。
