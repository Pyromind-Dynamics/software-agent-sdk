# 通用 LLM 生成、改写与过滤

## 适用

为单字段生成内容、多字段组合评分、文本改写或生成后质量过滤。最小输入是一个或多个
字符串列；最终输出 `text`。

## 算子选择

- 单字段生成：`PromptedGenerator`
- 多字段模板：`FormatStrPromptedGenerator + FormatStrPrompt`
- 原地改写：`PromptedRefiner`
- LLM 分数过滤：`PromptedFilter`
- 已有分数或规则过滤：`GeneralFilter`
- 生成结构化 JSON 时给 PromptedGenerator 提供严格 JSON Schema

```python
from dataflow.operators.core_text import (
    FormatStrPromptedGenerator, GeneralFilter, PromptedFilter,
    PromptedGenerator, PromptedRefiner,
)
from dataflow.prompts.core_text import FormatStrPrompt

generator = PromptedGenerator(llm_serving=llm, system_prompt=SYSTEM_PROMPT)
generator.run(storage=storage, input_key="prompt", output_key="answer")
storage = storage.step()

quality = PromptedFilter(
    llm_serving=llm,
    system_prompt="从正确性、完整性和清晰度评分 1-5。",
    min_score=4,
    max_score=5,
)
quality.run(storage=storage, input_key="answer", output_key="quality_score")
storage = storage.step()
```

Pipeline 最后只输出 `id/system_prompt/user_prompt/gt`。多字段输入必须使用格式模板，
不要串联多个职责相同的 generator。`PromptedRefiner` 前先过滤空值。

## 输出契约

按 `output_schema` 精确匹配根字段，`id` 在文件内唯一；审计信息只进 Report。

### `text`（默认）

```json
{"id":"t-1","system_prompt":"You are helpful.","user_prompt":"问题","gt":"答案"}
```

四个字段均为非空字符串。规则清洗本身是前处理，下游生成、QA、代码等场景同样映射成
此格式。

### `multiturn`

```json
{"id":"m-1","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"问题"},{"role":"assistant","content":"回答"}]}
```

`system` 可选且只能位于开头；后续严格 `user → assistant` 交替并以 assistant 结束。

### `function_call`

```json
{"id":"fc-1","tools":[{"type":"function","function":{"name":"lookup","description":"查询","parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}}}],"messages":[{"role":"user","content":"查询 A"},{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"A\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"{\"value\":1}"},{"role":"assistant","content":"结果是 1。"}]}
```

工具名和调用 ID 唯一；arguments 是 JSON object 字符串；每个调用恰好对应一个 tool
返回；完整轨迹以最终 assistant 回复结束。

### `quality_evaluation`

```json
{"id":"q-1","subject":{"type":"instruction_response","input":"问题","output":"回答"},"evaluation":{"overall_score":4,"decision":"keep","dimensions":[{"name":"correctness","score":4,"reason":"结论正确"}],"issues":[]}}
```

`subject.type` 为 `text/instruction_response/conversation/function_call/text2sql`；
评分 1～5，decision 为 `keep/rewrite/drop`。维度名称可扩展，但同一条内唯一。

### `text2sql`

```json
{"id":"sql-1","question":"查询用户数","database":{"id":"db1","dialect":"sqlite","schema":{"tables":[{"name":"users","columns":[{"name":"id","type":"INTEGER","nullable":false}],"primary_key":["id"]}],"foreign_keys":[]}},"evidence":"","sql":"SELECT COUNT(*) FROM users;"}
```

数据库 Schema 至少包含表、列、类型、主键和外键字段。执行结果、错误和难度进入
`scenario_metrics.json`，不进入训练行。
