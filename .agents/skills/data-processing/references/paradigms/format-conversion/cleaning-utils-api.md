# Cleaning Utils API

本文件是生成 `clean_script.py` 时唯一需要读取的 Utils 契约。不要查看或 grep
`scripts/cleaning/cleaning_utils.py`。如果平台行为与本文不一致，将其作为运行时契约缺陷报告。

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

## 图片等媒体引用

保存到数据中的文件引用与执行时读文件的路径分开：

| 用途 | 示例 |
|---|---|
| 数据集内部引用 | `images/a.jpg`，以明确的数据集目录为基准 |
| Storage 内跨目录引用 | `/datasets/demo/images/a.jpg`，不带挂载前缀 |
| 沙箱中实际读取 | `/workspace/datasets/demo/images/a.jpg` |
| 平台处理容器中实际读取 | 按该容器的挂载目录解析，例如 `/target-workspace/datasets/demo/images/a.jpg` |

公共方法直接从 `cleaning_utils` 导入，不需要读取实现：

| 方法 | 使用方式与边界 |
|---|---|
| `normalize_storage_reference` | 对一个已确定为文件引用的字符串，去掉开头一层 `/workspace` 或 `/target-workspace`；相对路径和远程 URL 原样保留。不负责把相对路径转为 Storage 根路径。 |
| `normalize_media_references` | 返回一份处理后的记录，不修改传入对象。处理已知的媒体字段，如 `images`、`image_paths`、`source_images`，以及 `messages` 中明确的图片、音视频地址；不全文替换正文、提示词、来源记录或工具调用参数。自定义字段须先确认含义，再逐值处理。 |
| `resolve_storage_reference` | 读文件时传入引用、当前执行环境的 `mount` 和实际的 `dataset_root`。`/datasets/...` 相对挂载目录解析，`images/...` 相对数据集目录解析；返回的路径仅用于读文件，不写回数据。远程 URL 不由此方法读取，含 `..` 的路径会报错。 |
| `write_jsonl` | 写入前自动处理已知媒体引用，返回写入条数；不负责完整格式校验。 |

例如，保存引用与实际读取分开：

```python
from cleaning_utils import normalize_storage_reference, resolve_storage_reference

reference = normalize_storage_reference("/workspace/datasets/demo/images/a.jpg")
image_path = resolve_storage_reference(
    reference,
    mount="/workspace",
    dataset_root="/workspace/datasets/demo",
)
# reference 保存为 /datasets/demo/images/a.jpg；image_path 仅用于打开图片。
```

平台执行时将 `mount` 换为平台实际挂载目录，不猜测、不改变保存的引用。
相对引用的读取基准必须固定；输出文件搬到其他目录后，不能默认仍以输出目录读取
原图片，需要重算相对引用，或保存不带挂载前缀的 Storage 根路径。

`run_cleaning` 会在 mapper 返回记录后、校验和去重前自动处理媒体引用。
自行用 `json.dump` 等方式输出时，必须主动调用公共方法。
`validate_record` 在检查 messages 格式时，会拒绝已知媒体字段中残留的挂载前缀，
并在错误中返回字段位置和传入的行号；它不检查图片是否存在。仍需用解析后的路径
实际打开图片，不能只看路径字符串或文件存在性。

## 校验、状态和错误

```python
validate_record(record: Any, *, line_number: int | None = None) -> list[ValidationError]
validate_messages_record(record: Any, *, line_number: int | None = None) -> list[ValidationError]
validate_preference_record(record: Any, *, line_number: int | None = None) -> list[ValidationError]
```

`run_cleaning` 对完整输出记录做 SHA-256 精确去重。`report.json` 集中保存 stats、错误样本、
checkpoint 和 validation；checkpoint 保存已安全提交的源位置、输出偏移和累计统计。
resume 先按输出偏移截断未提交尾部，再恢复去重状态并追加执行。`run_cleaning`
自身输出纯训练数据 `output.jsonl` 和状态报告；ID 对照表等附加文件由调用脚本
按任务要求另行生成，不混入训练记录。
