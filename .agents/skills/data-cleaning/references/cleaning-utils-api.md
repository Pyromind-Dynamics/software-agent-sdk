# Cleaning Utils API

本文件是生成 `clean_script.py` 时唯一需要读取的 Utils 契约。不要查看或 grep
`scripts/cleaning_utils.py`。如果平台行为与本文不一致，将其作为运行时契约缺陷报告。

`cleaning_utils.py` 与平台冻结的 cleaner 位于同一目录，脚本只实现字段映射；读取、
校验、去重、错误、统计和断点由 runner 负责。

## 推荐入口

```python
run_cleaning(
    *, input_path: str | Path, output_path: str | Path,
    state_dir: str | Path, mapper: Callable,
    resume: bool = False, limit: int | None = None,
) -> CleaningStats
```

`mapper(record)` 返回一个目标格式对象；返回 `None` 等价于 `DropRecord("filtered")`。
可抛 `DropRecord(reason)` 表示有意过滤。解析、映射或行级校验错误写入
`report.json` 的有界错误样本后继续；未处理的结构性错误写入失败状态并非零退出。

## 目标映射

```python
detect_training_format(record: dict[str, Any]) -> Literal["messages", "preference"]
to_training_record(record: dict[str, Any]) -> dict[str, Any]
to_messages_record(record: dict[str, Any], *, preference: str | None = None) -> dict[str, Any]
to_preference_record(record: dict[str, Any]) -> dict[str, str]
messages_from_record(record: dict[str, Any], *, preference: str | None = None) -> list[dict[str, Any]]
normalize_messages(value: Any) -> list[dict[str, Any]]
ensure_system_message(messages: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]
normalize_text(value: Any, *, collapse_spaces: bool = False) -> str
```

`detect_training_format` 在记录同时具备完整 prompt、chosen、rejected 时返回
`preference`，优先级高于 messages。`to_training_record` 按该结果自动转换。
`to_preference_record` 输出顶层仅含三个非空字符串；数组或对象分支会报错，避免
对话 DPO 被隐式展平。`to_messages_record(..., preference=...)` 只用于用户明确要求的
DPO→SFT 有损转换；文本 DPO 会把公共 prompt 映射为 user、选中分支映射为 assistant。
`instruction` 与非空 `input` 使用空行拼接，避免遗漏条件。`ensure_system_message`
保留已有的非空首条 system；缺失或为空时插入同一个固定 system prompt，不逐条生成
不同 prompt。

内置 messages 映射覆盖：

- `messages` / `conversations` / `conversation`；
- `events` / `trace` / `trajectory` / `steps`；
- prompt/instruction/question 与 response/output/completion/ground_truth；
- MMLU 风格 `question` + `choices` + `answer`，保留可选 `subject`。

## 输入读取

```python
iter_records(path: str | Path) -> Iterator[ParsedRecord]
iter_jsonl(path: str | Path, *, unwrap_huggingface: bool = True) -> Iterator[ParsedRecord]
iter_json(path: str | Path) -> Iterator[ParsedRecord]
iter_csv(path: str | Path) -> Iterator[ParsedRecord]
iter_text(path: str | Path) -> Iterator[ParsedRecord]
```

| 输入 | 行为 |
|---|---|
| `.csv` / `.tsv` | 表头映射为对象 |
| `.json` | 流式数组、单对象或 JSONL fallback |
| `.jsonl` | 逐行容错解析 |
| `.txt` / `.text` / `.log` / `.md` | 内容以 `{`/`[` 开始时先按 JSON 探测；没有任何合法 JSON 记录时回退为逐行 `{"text": ...}` |
| 目录 | 只遍历支持的文件，按相对路径稳定排序 |

JSON 对象形如 `{"rows":[{"row_idx":0,"row":{...}}]}` 时自动解开 HuggingFace
viewer 包装；`truncated_cells` 非空的 row 作为错误记录。`limit` 在解包后的源记录上
计数。

## 校验、状态和错误

```python
validate_record(record: Any, *, line_number: int | None = None) -> list[ValidationError]
validate_messages_record(record: Any, *, line_number: int | None = None) -> list[ValidationError]
validate_preference_record(record: Any, *, line_number: int | None = None) -> list[ValidationError]
```

`run_cleaning` 对完整输出记录做 SHA-256 精确去重。`report.json` 集中保存 stats、错误样本、
checkpoint 和 validation；checkpoint 保存已安全提交的源位置、输出偏移和累计统计。
resume 先按输出偏移截断未提交尾部，再恢复去重状态并追加执行。除纯训练数据
`output.jsonl` 外，不生成其他数据产物。
