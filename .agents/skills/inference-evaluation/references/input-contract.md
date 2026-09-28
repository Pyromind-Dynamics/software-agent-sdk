# 数据与配置

`input_path` 是 Storage JSONL 绝对路径，每行是独立测试 case。
模型引用也是 Storage 绝对路径；容器挂载前缀 `/workspace` 由工具统一添加。

数据映射示例：
```json
{"user_prompt_field":"prompt","reference_field":"ground_truth","media_field":"images","id_field":"id"}
```
也可使用 messages_field 指向 OpenAI messages 列表，或 system_prompt_field 提供系统提示词。
每个 case ID 必须唯一；没有 ID 时使用行号。图片可为单路径或路径列表；image_order 可指定文件名顺序。
相对图片路径基于 JSONL 父目录，只有确需另设根目录才传 media_base_dir。
不支持动态 request builder、HTTP 媒体或运行时加载自定义评分模块。

评测配置固定 mode=agent_rubric，rubrics 非空。pass_threshold 和 rubric_pass_threshold
默认 0.7。可选 generation 对象包含 workers=4、max_tokens=1024、temperature=0、
timeout_seconds=240、max_retries=3、limit=0；limit=0 表示全量。
配置中不得包含 API Key、Cookie 或运行时环境变量。
