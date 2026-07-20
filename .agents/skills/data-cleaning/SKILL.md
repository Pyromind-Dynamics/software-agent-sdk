---
name: data-cleaning
description: >-
  当用户要求对数据集进行清洗、格式化、转换、校验或修复时使用。覆盖完整流程：
  数据探索 → 格式分析 → 清洗策略设计 → 脚本生成/复用 → 试跑验证 → 结果检查 → 迭代修复 → 全量执行。
  用户可能用"清洗""clean""格式化""转换""修复数据""数据预处理"等表达同一诉求。
  本 Skill 驱动异步清洗任务提交（run_dataset_cleaning），通过 Kafka callback 自动续跑会话，
  失败则根据错误调整清洗规则重试。
triggers:
- 清洗
- 数据清洗
- clean data
- 格式化数据
- 转换数据
- 修复数据
- 数据预处理
- data cleaning
- 清洗一下
- 帮我清洗
---

# 数据清洗工作流

## 概述

数据清洗是将原始数据（日志、JSON、CSV、文本等）转换为结构化、高质量格式的过程。
本 Skill 驱动完整的清洗闭环：**探索 → 分析 → 设计 → 试跑 → 验证 → 迭代 → 全量**。

清洗任务通过 `run_dataset_cleaning` 工具异步提交到 Pyromind 平台执行，
平台终态经 Kafka callback 注入 `<system_reminder>` 并自动续跑会话。

---

## 核心工具

| 工具 | 用途 |
|------|------|
| `preview_dataset` | 探索数据源（共享空间/用户 storage），预览文件内容、查看目录结构 |
| `upload_file_to_pyromind` | 上传清洗脚本到平台 storage |
| `run_dataset_cleaning` | 提交异步清洗任务（支持 `--limit` 试跑、`resume_run_id` 断点续跑） |
| `read_file` / `execute_bash` | 检查清洗产物（stats.json、errors.jsonl、output.jsonl） |

---

## 清洗脚本 CLI 契约

所有清洗脚本**必须**遵循以下 CLI 接口：

```bash
python clean_script.py \
  --input <input_path> \
  --output <output_path> \
  --state-dir <state_dir> \
  [--resume] \
  [--limit N]
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--input` | ✅ | 输入数据路径（文件或目录） |
| `--output` | ✅ | 输出目录（存放清洗结果） |
| `--state-dir` | ✅ | 状态目录（存放 checkpoint、dedupe_state 等） |
| `--resume` | ❌ | 从断点续跑（读取 checkpoint.json） |
| `--limit N` | ❌ | 仅处理前 N 条记录（用于试跑） |

---

## 清洗产物规范

清洗完成后，`--output` 目录下应包含：

| 文件 | 说明 |
|------|------|
| `output.jsonl` | 清洗后的结构化数据（每行一个 JSON 对象） |
| `stats.json` | 汇总统计：`{"read": N, "written": M, "errors": E, "duplicates": D}` |
| `errors.jsonl` | 行级错误明细：`{"line": N, "error": "...", "raw": "..."}` |

`--state-dir` 目录下：

| 文件 | 说明 |
|------|------|
| `checkpoint.json` | 断点信息：`{"last_line": N, "timestamp": "..."}` |
| `dedupe_state.json` | 去重状态（可选） |

---

## 工作流程

### Phase 1: 数据探索与画像

**目标**：了解数据源结构、格式、规模。

1. 使用 `preview_dataset` 探索用户提供的路径：
   - 若为目录：列出文件清单（含大小、修改时间），**询问用户**确认目标文件
   - 若为文件：预览前 N 行内容，分析数据格式

2. 关键问题清单：
   - 数据格式是什么？（JSON/JSONL/CSV/日志/混合）
   - 每行/每条记录的结构？
   - 数据规模？（行数、文件大小）
   - 是否有明显的脏数据模式？（截断、编码错误、格式不一致）

### Phase 2: 清洗策略设计

**目标**：根据数据特征设计清洗规则。

根据数据格式选择策略：

| 数据格式 | 清洗策略 |
|----------|----------|
| 标准 JSONL | 校验 schema → 过滤无效行 → 去重 → 输出 |
| 非结构化日志 | 正则解析 → 提取字段 → 结构化 → 校验 → 输出 |
| CSV/TSV | 解析表头 → 类型转换 → 缺失值处理 → 转 JSONL |
| 混合格式 | 分流处理 → 统一 schema → 合并输出 |

设计原则：
- **保守策略**：不确定的记录写入 errors.jsonl，不丢弃
- **幂等性**：相同输入产生相同输出
- **可恢复**：支持 checkpoint 断点续跑
- **可观测**：stats.json 提供完整统计

### Phase 3: 脚本生成或复用

**目标**：生成或复用清洗脚本。

**复用检查**：
1. 检查数据源目录是否已有 `clean_script.py` 或类似脚本
2. 若有，分析其功能是否满足需求
3. 满足则直接复用，不满足则基于其修改

**生成新脚本**：
1. 遵循 CLI 契约（见上方）
2. 实现产物规范（见上方）
3. 添加进度日志（每 1000 条输出一次）
4. 异常处理：单条失败不中断整体

脚本模板要点：
```python
import argparse, json, os, sys
from pathlib import Path
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--state-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)
    
    stats = {"read": 0, "written": 0, "errors": 0, "duplicates": 0}
    start_line = 0
    
    # 断点续跑
    if args.resume:
        cp = Path(args.state_dir) / "checkpoint.json"
        if cp.exists():
            start_line = json.loads(cp.read_text()).get("last_line", 0)
    
    # ... 清洗逻辑 ...
    
    # 写入产物
    (Path(args.output) / "stats.json").write_text(json.dumps(stats, indent=2))
```

### Phase 4: 上传与试跑

**目标**：验证清洗逻辑正确性。

1. **上传脚本**：
   ```
   upload_file_to_pyromind(
     local_path="<script_path>",
     remote_path="<storage_path>/clean_script.py"
   )
   ```

2. **试跑**（使用 `--limit` 限制条数）：
   ```
   run_dataset_cleaning(
     script_path="<storage_path>/clean_script.py",
     input_path="<input_data_path>",
     output_dir="<storage_path>/data_cleaning/",
     limit=100  # 试跑 100 条
   )
   ```

3. **等待回调**：平台执行完成后通过 Kafka 注入 `<system_reminder>`，自动续跑会话。

### Phase 5: 结果检查

**目标**：验证清洗质量。

检查顺序（**必须按此顺序**）：

1. **stats.json** — 宏观统计
   - `read` vs `written`：转化率是否合理？
   - `errors`：错误率是否可接受？（>30% 需要调整策略）
   - `duplicates`：去重是否生效？

2. **errors.jsonl** — 错误明细
   - 错误类型分布（解析失败/校验失败/编码错误）
   - 是否有系统性问题？（如某类记录全部失败）
   - 是否需要调整解析规则？

3. **output.jsonl** — 样本抽查
   - 随机抽取 3-5 条检查结构完整性
   - 字段值是否合理？
   - 与原始数据对比是否丢失信息？

### Phase 6: 迭代修复

**触发条件**：
- 错误率 > 30%
- 发现系统性解析问题
- 输出 schema 不符合预期

**修复流程**：
1. 分析 errors.jsonl 中的错误模式
2. 修改清洗脚本（调整正则/校验规则/容错逻辑）
3. 重新上传 → 重新试跑 → 重新检查
4. 循环直到质量达标

### Phase 7: 全量执行

**目标**：对完整数据集执行清洗。

1. 确认试跑结果达标后，移除 `--limit` 参数：
   ```
   run_dataset_cleaning(
     script_path="<storage_path>/clean_script.py",
     input_path="<input_data_path>",
     output_dir="<storage_path>/data_cleaning/"
     # 不传 limit，全量执行
   )
   ```

2. 大数据集（>100MB）提醒用户：
   - 预计执行时间
   - 可通过 `resume_run_id` 断点续跑

3. 全量完成后再次检查 stats.json 确认最终结果。

---

## 关键决策点

### 何时询问用户？

| 场景 | 动作 |
|------|------|
| 目录含多个文件 | 列出文件清单，询问目标文件 |
| 数据格式不明确 | 展示样本，询问预期输出格式 |
| 错误率 > 30% | 报告错误分布，询问是否调整策略 |
| 清洗规则有歧义 | 展示样例，询问保留/丢弃策略 |

### 何时自动继续？

| 场景 | 动作 |
|------|------|
| 试跑成功且错误率 < 10% | 自动建议全量执行 |
| 单文件目录 | 自动选择该文件预览 |
| 已有可复用脚本 | 自动分析并建议复用 |

---

## 输出格式规范

清洗后的 `output.jsonl` 每行应为标准 JSON 对象。推荐 schema：

```json
{
  "id": "unique_identifier",
  "timestamp": "ISO8601",
  "content": "...",
  "metadata": {},
  "source": {"file": "...", "line": N}
}
```

根据实际数据调整字段，但必须包含：
- 唯一标识（`id` 或 `uuid`）
- 原始位置引用（`source.line`）便于溯源

---

## 常见数据格式处理

### 搜索/问答日志
```
原始: [2024-01-01 10:00:00] query="xxx" results=[...]
解析: 正则提取 timestamp、query、results → 结构化 JSON
```

### 多行 JSON
```
原始:  pretty-printed JSON 跨多行
解析: 累积括号匹配 → 完整 JSON 对象 → 压缩为单行
```

### 混合格式文件
```
原始:  部分行是 JSON，部分行是纯文本
解析: 逐行尝试 json.loads → 成功则结构化，失败则作为 text 字段
```

---

## 注意事项

1. **不要自动覆盖原始数据**：输出目录必须与输入分离
2. **保留错误记录**：所有被过滤的记录写入 errors.jsonl，不直接丢弃
3. **大文件分块**：>500MB 文件建议分块处理或提醒用户
4. **编码处理**：默认 UTF-8，遇到编码错误时尝试 latin-1 回退
5. **幂等保证**：相同输入 + 相同脚本 = 相同输出（去重状态除外）
