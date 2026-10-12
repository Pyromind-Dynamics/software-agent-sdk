# 通过 terminal 执行数据处理

普通脚本直接执行，例如 `python3 public_data/clean.py storage/input.csv public_data/result.csv`。
需要 DataFlow 算子、模型配置或标准输出校验时，用 `sandbox_sample.py`，优先直接传参：

```sh
python3 .agents/skills/data-processing/scripts/preparation/sandbox_sample.py \
  --pipeline public_data/pipeline.py \
  --input storage/input.jsonl \
  --output public_data/result.jsonl \
  --model-profile text \
  --output-schema text \
  --timeout 3600
```

`--input`/`--output` 解析成 Pipeline 的前两个参数，`--arg` 可重复追加后续参数；
以 `-` 开头的值写成 `--arg=--limit` 形式，避免被当作脚本自己的选项。
一条命令即可完成，不需要 `run.json` 之类的中间文件。

只有 flags 覆盖不到的字段（`support_file_path`、`python`、`gateway`）才写不含密钥的 JSON 配置，
用 `--config` 执行：

```json
{
  "pipeline_path": "public_data/pipeline.py",
  "args": ["storage/input.jsonl", "public_data/result.jsonl"],
  "model_profile": "text",
  "output_schema": "text",
  "timeout": 3600
}
```

```sh
python3 .agents/skills/data-processing/scripts/preparation/sandbox_sample.py --config public_data/run.json
```

两种方式都在会话工作区根目录执行，模型配置由执行环境注入，不需要在命令里重复导出。
OS 沙箱下技能资源可能位于工作区外，使用已知的技能绝对路径替换命令中的资源路径。

## 运行环境

Pipeline 用的解释器由脚本自己解析，顺序为：`PYROMIND_SANDBOX_DATAFLOW_PYTHON` 指定的
解释器（在本沙箱内存在且版本匹配时）→ 镜像内置解释器（`/opt/pyromind/dataflow-venv/bin/python`、
`/home/openhands/dataflow-venv/bin/python`、`~/dataflow-venv/bin/python`）→ 沙箱自带
`python3`（版本匹配时）→ `/tmp/pyromind-dataflow-venv-1.0.10` 缓存 venv，缺失时按 pin 安装。
首次安装需要沙箱能访问 PyPI，通常几分钟；同一沙箱内后续运行直接复用，沙箱重建后重新准备。
沙箱初始化完成后会在后台自动执行一次 `--prepare`，所以通常不需要手工预热；
预热失败不影响后续运行，执行小样时会按需重新安装。可用
`PYROMIND_SANDBOX_DATAFLOW_WARMUP=0` 关闭自动预热。

```sh
# 只看会用哪个解释器，不改动沙箱
python3 .agents/skills/data-processing/scripts/preparation/sandbox_sample.py --check
# 提前建好缓存 venv（幂等，可重复执行）
python3 .agents/skills/data-processing/scripts/preparation/sandbox_sample.py --prepare
# 预热：后台跑，与后续写 Pipeline 并行；同一沙箱只需一次
nohup python3 .agents/skills/data-processing/scripts/preparation/sandbox_sample.py --prepare \
  > /tmp/pyromind-dataflow-prepare.log 2>&1 &
```

两条命令都输出一行 JSON：`ready`、`interpreter`、`source`（`configured`/`image`/`reused`/`installed`）、
`version`。安装互斥，预热还没结束时小样执行会等它装完再复用，不会重复安装或写坏 venv；
预热进度看 `/tmp/pyromind-dataflow-prepare.log`。
不要用 `import dataflow`、全盘 `find` 或翻 site-packages 判断环境是否可用：terminal 的默认
`python3` 不是运行 Pipeline 的解释器，只有 `--check` 的结论算数。

flags 与配置字段一一对应：`--pipeline` → `pipeline_path`，`--input`/`--output`/`--arg` → `args`，
`--model-profile`/`--output-schema`/`--timeout` 与同名字段相同；`--config` 额外支持
`support_file_path`、`python`、`gateway`。`--pipeline` 与 `--config` 互斥，直接传参时不要写配置文件。

`model_profile`：`none` 为无模型的 Python，`text` 为会话模型，`vision` 为图片模型。
SDK 在执行环境中注入配置。配置缺失时报告具体缺项，不读取服务端配置或把密钥写入脚本。
可选 `gateway` 仅用于 vision，字段为 `api_url`、`model`、`api_key_env`；后者引用宿主已配置的环境变量名，不填密钥值。

可选 `support_file_path` 为辅助 JSON；`python` 为沙箱内解释器。
`output_schema` 可选 text、dpo、vision、structured、multiturn、function_call、quality_evaluation、text2sql、artifacts；非标准输出省略。
输入完整处理，不自动抽样。前两个参数按工作区输入和输出路径解析；输出只能位于 public_data/。
模型配置、依赖准备、格式校验和报告生成由 CLI 完成；终端返回退出码和 JSON 结果，失败先查看 failure_stage 和报告。
终端中断会停止进程；断线重连恢复日志观察，不再次启动脚本。产物仍用 get_storage_url 获取 Markdown 链接。
平台全量提交继续使用 df_submit_pipeline，不把平台作业改成本地执行。
