# 格式转换（format-conversion）

- **基底**：会话沙箱（Storage 已挂载为 `storage/`）。需要平台侧冻结脚本和审计报告
  时改用 `run_dataset_cleaning` 的 Pod 路径
- **适用**：数据集转平台训练格式（messages/DPO）、字段映射、对话结构解析、
  格式校验、SHA-256 去重；mapper 可写简单内容过滤（regex/关键词/长度）；
  断点续跑、错误隔离、report 统计
- **不适用**：复杂内容清洗（去重/PII/语言/毒性/评分/生成）→ llm-pipeline；
  需在特定环境执行复杂流程的编排式处理（验证/筛选/执行判定）→ environment-processing

选择执行面：用户 Storage 已挂载为 `storage/`，可直接查看源数据、执行清洗和
查看产物。确定性格式转换默认在会话沙箱内完成，不需要下载数据到沙箱之外。
只有用户明确要求平台执行，或确实需要平台 run 目录、`report.json` 和冻结运行时
时，才走下面的平台路径。

## 沙箱路径（默认）

1. 查看用户给出的 Storage 路径：直接读 `storage/<path>`。目录包含多个候选文件时，
   先让用户确认输入；确认字段语义、对话分组、过滤规则、system prompt 和有损转换。
2. 只使用 [target-formats.md](target-formats.md) 定义的格式。完整的
   prompt/chosen/rejected 自动按 DPO 清洗，不丢弃任一分支；其他数据输出 messages。
3. 用 `write` 写转换脚本，用 `terminal` 执行；路径统一相对工作区根目录，输入用
   `storage/<path>`，输出写到目标 Storage 目录。脚本只依赖标准库，保持确定性，
   不引入平台专用 import。
4. 先跑小样或全量并打印保留/丢弃计数，再按 [target-formats.md](target-formats.md)
   校验输出：顶层字段、messages 角色与 content、DPO 三分支都要检查。校验失败
   修改脚本重跑，不要把未校验的产物交给用户。
5. 校验通过后展示最多三条结果并等待用户明确确认；未通过的试跑迭代不向用户展示。
6. 用户确认后再跑全量，产物落在 `storage/` 下并报告路径与统计。

## 平台路径

需要平台 run 目录、`report.json` 或冻结运行时时才走这里：

1. [cleaning-utils-api.md](cleaning-utils-api.md) 是正常清洗任务中唯一的
   Utils 契约。读取它和 [example_clean_script.py](example_clean_script.py)
   后编写字段映射；不要查看、搜索或反复读取 `cleaning_utils.py` 实现。只有用户明确
   要求维护 Skill/运行时，或平台行为与 API 契约矛盾时，才检查实现。
2. 把脚本写到 `storage/<dir>/<name>`，再调用 `run_dataset_cleaning`，传
   `script_path`、`input_path` 和 `limit=3`。不要自行传 `output_dir`。
   `input_path`、`script_path` 都是 Storage 根路径（以 `/` 开头），不是工作区
   路径；工作区里的脚本先落到 `storage/` 再引用。
3. 收到通用 callback 后，从最近一次工具 Observation 取得 `run_id` 和
   `output_dir`，直接读取 `storage/<output_dir>/report.json` 和
   `output.jsonl`。
   `report.json` 包含格式校验、统计、错误样本和 checkpoint。格式失败或存在系统性
   清洗错误时修改脚本并创建新的试跑，不恢复错误 run。失败且没有报告时按启动失败
   处理，不要解码 callback 的不透明 `error_log`。
4. 格式和语义检查通过后，展示最多三条结果并等待用户明确确认；未通过的试跑迭代
   不向用户展示。确认后优先用
   `<sample_output_dir>/clean_script.py` 创建一个不传 `limit` 的新全量 run。
5. 只有平台或进程中断且脚本逻辑正确时，才传原 `input_path` 和
   `resume_run_id` 恢复；恢复时不传新脚本。Pending/Running 时等待 callback，
   不重复提交。
6. 提交返回不透明 400、且本地 preflight 已通过时，不要原样重试。向用户报告平台
   提交失败，并按默认沙箱路径继续完成转换；用户明确要平台产物时再排查平台侧。

## 平台脚本契约

```text
python3 clean_script.py \
  --input <input_file_or_directory> \
  --output <run_dir>/output.jsonl \
  --state-dir <run_dir> \
  [--resume] \
  [--limit N]
```

`--limit` 限制解包后考虑的源记录数，不是最终保留数。平台会在创建 run 前静态
检查脚本语法和 Utils import；这不读取或执行数据。平台自动把冻结的
`clean_script.py`、`cleaning_utils.py` 和 `validate_format.py` 放在同一 run 目录，
cleaner 完成后在同一 Pod 校验并合并更新 `report.json`。

每个 run 只生成两个数据产物：纯训练数据 `output.jsonl`，以及包含统计、错误、
checkpoint 和格式校验的 `report.json`。
