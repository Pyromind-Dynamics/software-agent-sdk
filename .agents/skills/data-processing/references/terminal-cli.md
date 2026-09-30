# 通过 terminal 执行数据处理

普通脚本直接执行，例如 `python3 public_data/clean.py storage/input.csv public_data/result.csv`。
需要 DataFlow 算子、模型配置或标准输出校验时，先写一个不含密钥的 JSON 配置：

```json
{
  "pipeline_path": "public_data/pipeline.py",
  "args": ["storage/input.jsonl", "public_data/result.jsonl"],
  "model_profile": "text",
  "output_schema": "text",
  "timeout": 3600
}
```

在会话工作区根目录，通过 terminal 执行：

```sh
PYROMIND_DATAFLOW_PROFILES="$PYROMIND_DATAFLOW_PROFILES" python3 .agents/skills/data-processing/scripts/preparation/sandbox_sample.py --config public_data/run.json
```

OS 沙箱下技能资源可能位于工作区外，使用已知的技能绝对路径替换命令中的资源路径。

`model_profile`：`none` 为无模型的 Python，`text` 为会话模型，`vision` 为图片模型。
SDK 在执行环境中注入配置。配置缺失时报告具体缺项，不读取服务端配置或把密钥写入脚本。
可选 `gateway` 仅用于 vision，字段为 `api_url`、`model`、`api_key_env`；后者引用宿主已配置的环境变量名，不填密钥值。

可选 `support_file_path` 为辅助 JSON；`python` 为沙箱内解释器。
`output_schema` 可选 text、dpo、vision、structured、multiturn、function_call、quality_evaluation、text2sql、artifacts；非标准输出省略。
输入完整处理，不自动抽样。前两个参数按工作区输入和输出路径解析；输出只能位于 public_data/。
模型配置、依赖准备、格式校验和报告生成由 CLI 完成；终端返回退出码和 JSON 结果，失败先查看 failure_stage 和报告。
终端中断会停止进程；断线重连恢复日志观察，不再次启动脚本。产物仍用 get_storage_url 获取 Markdown 链接。
平台全量提交继续使用 df_submit_pipeline，不把平台作业改成本地执行。
