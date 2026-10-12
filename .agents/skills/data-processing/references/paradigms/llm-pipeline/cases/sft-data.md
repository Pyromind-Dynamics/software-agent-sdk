# SFT 数据合成与筛选

## 适用

从种子文本或领域内容生成 instruction/response，并对可读性、相关性和回答质量筛选。
最终输出 `text`。

## 推荐链路

```python
from dataflow.operators.text_sft import AlpagasusFilter, CondorGenerator, CondorRefiner

CondorGenerator(llm_serving=llm).run(
    storage=storage,
    input_key="text",
    output_instruction_key="instruction",
    output_response_key="response",
)
storage = storage.step()

CondorRefiner(llm_serving=llm).run(
    storage=storage,
    input_instruction_key="instruction",
    input_response_key="response",
)
storage = storage.step()

AlpagasusFilter(llm_serving=llm).run(
    storage=storage,
    input_instruction_key="instruction",
    input_response_key="response",
)
storage = storage.step()
```

导入路径见 [pipeline-reference.md](../pipeline-reference.md) 的导入路径表；`run()`
参数按本 case 的调用形式照抄。最终映射：

```text
instruction → user_prompt
response    → gt
```

输出契约（`output_schema="text"`）：

```json
{"id":"sft-1","system_prompt":"You are helpful.","user_prompt":"问题","gt":"答案"}
```

四个字段均为非空字符串，根字段精确匹配，`id` 唯一。

不使用默认 CUDA 的 `DeitaQualityFilter`、`SuperfilteringFilter`。
