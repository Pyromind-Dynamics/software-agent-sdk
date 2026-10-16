# DataFlow Pipeline 参考

写 Pipeline 需要的约定都在本文件：导入路径 → 脚本骨架 → 常见模式 → 算子用法 →
运行约束。场景链路（用哪些算子、字段怎么映射）看对应 case 文档；本文件只管怎么落地。

## 导入路径

DataFlow 不在 `dataflow.operators` 顶层导出算子，`getattr(operators, "PromptedFilter")`
一定失败；`dataflow.operators.core_text.filter` 也不是可导入的包（`filter/` 没有
`__init__.py`）。下表是 open-dataflow 1.0.10 实测可用的导入路径，直接照抄，不要在
沙箱里翻 site-packages 反查：

```python
# core_text：生成、改写、打分、过滤
from dataflow.operators.core_text import (
    FormatStrPromptedGenerator, GeneralFilter, PandasOperator, PromptedFilter,
    PromptedGenerator, PromptedRefiner, Text2MultiHopQAGenerator, Text2QAGenerator,
)
# 多字段模板
from dataflow.prompts.core_text import FormatStrPrompt
# general_text：规则清洗，不调用 LLM
from dataflow.operators.general_text import (
    ContentNullFilter, HashDeduplicateFilter, HtmlUrlRemoverRefiner, LanguageFilter,
    MinHashDeduplicateFilter, NgramFilter, RemoveEmojiRefiner,
    RemoveExtraSpacesRefiner, WordNumberFilter,
)
# SFT 合成与筛选
from dataflow.operators.text_sft import (
    AlpagasusFilter, CondorGenerator, CondorRefiner, DeitaQualityFilter,
    RMFilter, RMSampleEvaluator, SuperfilteringFilter,
)
# Code 数据
from dataflow.operators.code import (
    CodeCodeToInstructionGenerator, CodeInstructionGenerator,
    CodeInstructionToCodeGenerator, CodeQualitySampleEvaluator,
    CodeQualityScoreFilter, CodeSandboxSampleEvaluator,
)
# Reasoning 数据
from dataflow.operators.reasoning import (
    ReasoningAnswerFormatterFilter, ReasoningAnswerGenerator,
    ReasoningAnswerGroundTruthFilter, ReasoningAnswerNgramFilter,
    ReasoningQuestionCategorySampleEvaluator, ReasoningQuestionDifficultySampleEvaluator,
    ReasoningQuestionFilter, ReasoningQuestionGenerator,
)
# 知识清洗与 Storage
from dataflow.operators.knowledge_cleaning import KBCChunkGenerator
from dataflow.utils.storage import LazyFileStorage
```

模型 serving 是唯一需要绕开的路径：`APILLMServing_request` 在
`dataflow.serving.api_llm_serving_request`，但 `dataflow/serving/__init__.py` 会拉起
transformers/torch，所以按下面骨架的 importlib shim 加载，不要写
`from dataflow.serving import APILLMServing_request`。调用日志包装是
`from df_logging import LoggingLLMServing`（`scripts/preparation/` 运行库目录，运行时
经 `PYTHONPATH` 注入，Pipeline 不需要自己改 `sys.path`）。
图片标注入口见 [vision/image-utils-api.md](vision/image-utils-api.md)。

## 脚本骨架

完整可运行模板：[templates/text_pipeline.py](templates/text_pipeline.py)（分批 +
断点续跑 + progress）。骨架只做三件事：

```python
# 1) 顶部 shim：加载 APILLMServing_request，绕开 dataflow.serving 的重依赖
import importlib.util
import sys
import types
from pathlib import Path

import dataflow

_pkg = types.ModuleType("dataflow.serving")
_pkg.__path__ = []
sys.modules["dataflow.serving"] = _pkg
_serving = Path(dataflow.__file__).parent / "serving" / "api_llm_serving_request.py"
_spec = importlib.util.spec_from_file_location(
    "dataflow.serving.api_llm_serving_request", str(_serving)
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dataflow.serving.api_llm_serving_request"] = _mod
_spec.loader.exec_module(_mod)
APILLMServing_request = _mod.APILLMServing_request

from df_logging import LoggingLLMServing  # noqa: E402


# 2) storage：每个算子执行后推进一次 step，下一个算子读新字段
storage = LazyFileStorage(input_path, cache_type="jsonl")
storage = storage.step()

operator.run(storage=storage, ...)
storage = storage.step()

rows = storage.read(output_type="dict")


# 3) serving：只读 DF_* 环境变量，使用 LLM 的算子必须包 LoggingLLMServing
raw_llm = APILLMServing_request(
    api_url=os.environ["DF_API_URL"],
    model_name=os.environ["DF_MODEL_NAME"],
    key_name_of_api_key="DF_API_KEY",
    max_workers=8,
    max_retries=int(os.environ.get("DF_LLM_MAX_RETRIES", "2")),
    read_timeout=float(os.environ.get("DF_READ_TIMEOUT", "300")),
)
llm = LoggingLLMServing(raw_llm)
```

`sandbox_sample.py` 注入凭证和模型配置。文本 serving 使用完整 endpoint `DF_API_URL`；
VLM serving 使用 OpenAI-compatible 根地址 `DF_API_BASE_URL`；模型名使用
`DF_MODEL_NAME`，密钥环境变量名为 `DF_API_KEY`。脚本不得读取、打印或硬编码密钥。

## 常见模式

模式片段只写算子编排；shim、storage、serving 和写盘按上面的骨架和模板。使用 LLM 的
模式必须用 `LoggingLLMServing` 包装。`GeneralFilter` 的 lambda 用到 `pd.to_numeric`
时脚本顶部需 `import pandas as pd`。

### 模式 1：生成 + 质量过滤

场景：从单字段生成内容，然后用 LLM 打分过滤低质量行。

```python
from dataflow.operators.core_text import PromptedGenerator, PromptedFilter

generator = PromptedGenerator(
    llm_serving=llm,
    system_prompt="为这道数学题写详细的解题过程。",
)
generator.run(storage=storage, input_key="problem", output_key="solution")
storage = storage.step()

# 保留 4-5 分
quality_filter = PromptedFilter(
    llm_serving=llm,
    system_prompt="评估解题过程的质量，从 1-5 打分。考虑正确性、完整性和清晰度。",
    min_score=4,
    max_score=5,
)
quality_filter.run(storage=storage, input_key="solution", output_key="quality_score")
storage = storage.step()
```

### 模式 2：多字段评分 + 规则过滤

场景：组合多个字段评分，再用确定性规则过滤。多字段必须用
`FormatStrPromptedGenerator`，不要串联多个 `PromptedGenerator`。

```python
from dataflow.operators.core_text import FormatStrPromptedGenerator, GeneralFilter
from dataflow.prompts.core_text import FormatStrPrompt

prompt_template = FormatStrPrompt(
    f_str_template=(
        "请评估这条训练样本的质量。\n"
        "指令: {instruction}\n"
        "回答: {response}\n"
        "只返回一个 1-5 的整数分数。"
    )
)
scorer = FormatStrPromptedGenerator(
    llm_serving=llm,
    system_prompt="你是严格的数据质量评估员。",
    prompt_template=prompt_template,
)
# kwargs: key=模板变量名, value=DataFrame 列名
scorer.run(
    storage=storage,
    output_key="quality_score",
    instruction="instruction",   # 模板 {instruction} → 列 "instruction"
    response="output",           # 模板 {response} → 列 "output"
)
storage = storage.step()

# 健壮解析 LLM 输出后过滤
rule_filter = GeneralFilter([
    lambda df: pd.to_numeric(
        df["quality_score"].str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(0) >= 4,
])
rule_filter.run(storage=storage)
storage = storage.step()
```

### 模式 3：多阶段流水线

场景：评分 → 初筛 → 空行过滤 → 改写 → 再评分 → 过滤。多个 `PromptedGenerator`
阶段允许，但每阶段必须有不同语义职责；`PromptedRefiner` 前必须确保 `input_key`
无空值。

```python
from dataflow.operators.core_text import PromptedGenerator, PromptedRefiner, GeneralFilter

init_scorer = PromptedGenerator(
    llm_serving=llm,
    system_prompt="对这段文本质量打分 1-5，只返回整数。",
)
init_scorer.run(storage=storage, input_key="raw_content", output_key="init_score")
storage = storage.step()

pre_filter = GeneralFilter([
    lambda df: pd.to_numeric(
        df["init_score"].str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(0) >= 2,
])
pre_filter.run(storage=storage)
storage = storage.step()

# 过滤空行，避免 PromptedRefiner 长度不匹配报错
empty_filter = GeneralFilter([lambda df: df["raw_content"].str.len() > 0])
empty_filter.run(storage=storage)
storage = storage.step()

# 原地改写，覆盖 raw_content
refiner = PromptedRefiner(
    llm_serving=llm,
    system_prompt="改写这段文本，提升清晰度和完整性，保持原意。",
)
refiner.run(storage=storage, input_key="raw_content")
storage = storage.step()

final_scorer = PromptedGenerator(
    llm_serving=llm,
    system_prompt="对改写后的文本质量打分 1-5，只返回整数。",
)
final_scorer.run(storage=storage, input_key="raw_content", output_key="final_score")
storage = storage.step()

final_filter = GeneralFilter([
    lambda df: pd.to_numeric(
        df["final_score"].str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(0) >= 4,
])
final_filter.run(storage=storage)
storage = storage.step()
```

### 模式 4：规则清洗流水线（不需要 LLM）

场景：去 emoji、压缩空格、语言过滤、词数过滤、去重。不需要
`APILLMServing_request`，也不需要 importlib shim（除非混用 `core_text` 算子）。

```python
from dataflow.operators.general_text import (
    RemoveEmojiRefiner, RemoveExtraSpacesRefiner,
    LanguageFilter, WordNumberFilter, HashDeduplicateFilter,
)

RemoveEmojiRefiner().run(storage=storage, input_key="text")
storage = storage.step()
RemoveExtraSpacesRefiner().run(storage=storage, input_key="text")
storage = storage.step()
LanguageFilter(allowed_languages=["zh", "en"]).run(storage=storage, input_key="text")
storage = storage.step()
WordNumberFilter(min_words=10, max_words=50000).run(storage=storage, input_key="text")
storage = storage.step()
HashDeduplicateFilter(hash_func="md5").run(storage=storage, input_key="text")
storage = storage.step()
```

### 模式选择

| 用户需求 | 模式 | 核心算子 |
|---|---|---|
| 给数据加一列生成内容 | 1（去掉 filter） | PromptedGenerator |
| 生成内容并保证质量 | 1 | PromptedGenerator + PromptedFilter |
| 根据多个字段打分/评估 | 2 | FormatStrPromptedGenerator + GeneralFilter |
| 清洗/改写/提升质量 | 3 | PromptedGenerator + PromptedRefiner + GeneralFilter |
| 从文档生成 QA 对 | 1 换生成器 | Text2MultiHopQAGenerator |
| 去 emoji/过滤语言/去重/词数过滤 | 4 | general_text Filter + Refiner |
| 先规则清洗再 LLM 生成 | 4 + 1 串联 | general_text + core_text |

## 算子用法要点

### 单字段生成

```python
generator = PromptedGenerator(
    llm_serving=llm,
    system_prompt="...",
    user_prompt="",
    json_schema=None,
)
generator.run(storage=storage, input_key="text", output_key="generated")
```

`input_key` 和 `output_key` 属于 `run()`；`run()` 返回字段名，不返回 storage。

### 多字段生成或评分

```python
prompt = FormatStrPrompt(
    f_str_template="问题: {question}\n答案: {answer}\n只返回 1-5 分。"
)
operator = FormatStrPromptedGenerator(
    llm_serving=llm,
    system_prompt="...",
    prompt_template=prompt,
)
operator.run(
    storage=storage,
    output_key="score",
    question="problem",
    answer="solution",
)
```

关键字参数的名称匹配模板占位符，值是现有 DataFrame 列名。

### 过滤与改写

- `PromptedFilter.run(storage, input_key, output_key)`：单字段 LLM 语义评分过滤。
- `GeneralFilter([lambda df: ...]).run(storage)`：已有字段的确定性过滤。
- `PromptedRefiner.run(storage, input_key)`：原地覆盖字段；调用前过滤空值。
- `Text2MultiHopQAGenerator.run(storage, input_key, output_key)`：输出嵌套 QA 列表。

长度、语言、去重、PII、emoji、HTML 和空格处理使用 `general_text` 中对应 Filter 或
Refiner，不调用 LLM。

### 结构化输出

每个 JSON Schema 中 `"type": "object"` 的节点都设置
`"additionalProperties": false`。在 pipeline 内解析并校验模型返回值，把单样本失败
写入报告，不因一条坏数据丢弃整个批次。

## 运行约束

### 单条硬截止与失败账本（必读）

serving 网关会让挂起的连接一直保持（socket `read_timeout` 不触发，实测单请求可挂
20–30 分钟），所以 `LoggingLLMServing` 对每条数据强制墙钟截止
`DF_REQUEST_DEADLINE`（默认 600s，覆盖该条的全部内部重试）：到点即记
`status="deadline"`（`llm_calls.jsonl`）并放弃等待，其余数据继续。截止无法区分
"挂起"与"慢而活着"（非流式请求无进度信号）——误杀的慢记录进失败账本，由补跑轮
重试，不会丢数据。旋钮：`DF_REQUEST_DEADLINE`（截止）、`DF_SLOW_WARN_SECONDS`
（慢调用 WARN 阈值，默认 120s）。

pipeline 必须把无响应/解析失败的记录写入 `failures.jsonl`（`reason` + 完整
`input` 行），`generate_report.py` 会汇总进 `report.json.failures`。交付/分诊时若
`failures` 非空，提取 `input` 行重组子集输入补跑并合并产出。

### 输出与审计

所有正式产物都是 UTF-8 JSONL，一行一个对象；根字段与所选 `output_schema` 精确匹配，
`id` 在文件内唯一。每个 schema 的字段定义写在对应 case 文档的「输出契约」段
（`structured` 由任务自己的 `response_json_schema` 定义）。

- 算子中间列不是正式产物。Pipeline 最后只写所选 schema 契约允许的字段。
- 模型、源路径、调用耗时、重试、难度和过滤原因不得混入训练行。
- 场景汇总可写入 `DF_LOG_DIR/scenario_metrics.json`，内容必须是 JSON object；
  `generate_report.py` 会把它放入 `report.json.scenario_metrics`。
- DataFlow Checkpoint、`llm_calls.jsonl`、`failure.json` 和 `validation.json` 由共享
  runtime 管理，Pipeline 不复制这些脚本。

### 资源与版本

- 沙箱小样和 Pyromind 都使用 `open-dataflow==1.0.11`。
- Text2SQL 算子在 Python 3.13 存在上游 `re.template` 兼容问题，这类 Pipeline 需要
  用 `PYROMIND_SANDBOX_DATAFLOW_PYTHON` 指向沙箱内 Python 3.10 的解释器。
- 只使用 API Chat/Vision 和 CPU 算子；需要 Embedding、CUDA、外部服务或模型下载的
  算子不进入首批链路。
