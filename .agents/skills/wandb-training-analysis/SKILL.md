---
name: wandb-training-analysis
description: >-
  分析 Pyromind 平台训练工作流的 wandb 训练数据并给出优化建议,并联动
  data-cleaning/data-preparation skill 分析训练数据集质量。用户说“分析训练效果”、
  “看看 loss 曲线”、“训练发散/NaN/尖峰了”、“对比两次训练”、“优化超参”、
  “评估这次训练怎么样”、“分析一下{task_id}的执行情况”或给出 task_id / wandb run 链接
  时使用;负责定位 wandb run、探查指标与配置、单 run 稳定性诊断、run 对比、
  四阶段分析报告,并与 generate-workflow-dsl 衔接生成调优后的训练工作流。
  不需要本地 workflow.py 文件,通过 task_id 独立分析。
---

# wandb 训练数据分析与优化

把 wandb 训练数据查询、异常诊断和优化建议集成为一条链路:定位 run → 探查契约 →
分析 → 报告 → 检查数据集质量 → 生成调优工作流。所有数据经 wandb SDK 读取,
不依赖页面可访问性。

当报告发现 NaN/过拟合/未收敛时,建议用户检查训练数据集质量。对用户沟通时
一律使用业务语言(如“检查并优化训练数据集”),不得向用户暴露内部 skill 名称;
内部按需路由到数据清洗能力(基于规则的确定性清洗,或需 LLM 理解的数据处理)。

## 数据定位(关键)

wandb 凭证与 run id **均从平台 API 动态获取,不需要用户提供环境变量**:

1. 用户提供 `task_id`(或从 `run_workflow`/`workflow_debug` 返回值获得)。
2. 调平台 API `GET {api_base}/api/task_workflow_result?task_id={task_id}`
   拿节点列表。认证与 `validate_workflow` 工具一致:`cookie`/`x-cluster`/
   `authorization` header 小写透传。在 agent 会话内运行脚本时,**环境变量
   由 agent-server 自动注入,无需手动提供**: cookie 读
   `PYROMIND_VALIDATE_AUTH_COOKIE`,x-cluster 读 `PYROMIND_X_CLUSTER`
   (命令文本中须引用这些名字,如 `--cookie "$PYROMIND_VALIDATE_AUTH_COOKIE"`,
   secret 注入机制才会生效)。也可用 `--cookie`/`--cluster`/`--authorization`
   参数或 `PYROMIND_COOKIE`/`X_CLUSTER`/`PYROMIND_AUTHORIZATION` 回退。
   `{api_base}` 按 `APP_ENV` 推断(prod/production/online 走
   `https://api-portal.pyromind.ai/std2/studio_api/`,否则
   `https://pre-api-portal.pyromind.ai/std2/studio_api/`),可用 `--api-base` 或
   `PYROMIND_API_BASE` 覆盖。
3. **凭证直接取自节点 config**: 响应 `workflow.nodes[]` 自带全部节点 config——
   训练节点 `data.config.wandb_config` 文本块或 `WandbConfigBuilderNode`
   `data.config` 结构化字段(`wandb_api_key` 等),无需额外接口。
4. 过滤训练节点(`data.nodeType` 含 `Train`,如 `ModelTrainSFTNode`)→ 查
   `internal/logs/node/raw?nodeId={node_id}&taskId={task_id}` → 从 stdout 提取
   `wandb: setting up run <run_id>`。
5. entity 由 wandb API `viewer()` 推断(或 `--entity` 指定)。
   config 未含凭证时回退:查 `WandbConfigBuilderNode` 的
   `internal/output/node/raw?node_code={node_id}&task_id={task_id}`。

用户直接给出 wandb run URL 时更优先:URL 固定为
`https://wandb.ai/{entity}/{project}/runs/{run_id}`,末尾段即 run id。

**只有平台 API 不可用时**才回退:解析工作流 DSL 的 `wandb_project`/`wandb_name`
匹配,或按时间窗口匹配最近 finished runs(见 `platform-data-contract.md`)。

## 主流程

> 命令约定:全局参数(`--api-base`/`--cookie`/`--cluster`/`--authorization`/`--api-key`/`--creds-file`/`--entity`)必须写在子命令名之前。

1. **定位**: `python scripts/wandb_analysis.py --api-base {api_base} --cookie "$PYROMIND_VALIDATE_AUTH_COOKIE" --cluster "$PYROMIND_X_CLUSTER" resolve-target {task_id} --creds-out {tmp}/creds.json`
   输出 `entity`/`project`/`run_id`;凭证写至 creds 文件(600 权限)。
   cookie/x-cluster 环境变量由 agent-server 注入;若本机调试无注入,改用
   `--cookie`/`--cluster` 显式传入。
2. **探查**: `python scripts/wandb_analysis.py --creds-file {creds} probe {entity}/{project} --run-id {run_id}`
   确认指标键族(如 `train/loss`)与 config 键族。
3. **分析**: 按场景选择
   - 单 run 诊断: `python scripts/wandb_analysis.py --creds-file {creds} analyze-run {entity}/{project} {run_id} --metric train/loss`
     `--keys` 参数支持多指标同时拉取: `--keys train/loss,train/entropy,train/learning_rate,train/grad_norm`
   - 两 run 对比: `python scripts/wandb_analysis.py --creds-file {creds} compare-runs {entity}/{project} {run_a} {run_b}`
   - 项目分桶: `python scripts/wandb_analysis.py --creds-file {creds} project-summary {entity}/{project} --axis learning_rate`
4. **报告**: `python scripts/wandb_analysis.py --creds-file {creds} report {entity}/{project} {run_id} --out report.md`,
   输出四阶段结论(先验 → 惊奇 → 机制 → 探针实验表)。
   若诊断发现 NaN/过拟合/未收敛,报告会自动包含数据集质量检查建议。
5. **数据集质量检查**: 当报告建议检查训练数据时,内部路由到数据清洗能力
   (基于规则的确定性清洗;需要 LLM 理解时用 LLM 数据处理),分析训练数据集。
   - 读取训练工作流中的数据集路径(`dataset_kwargs`/`dataset_text_field` 等 config 字段)
   - 预览数据样本,检查异常(空值/乱码/重复/标签分布)
   - 输出数据质量报告,与 wandb 训练报告合并分析
   - 对用户沟通时用业务语言(如“我帮你检查一下训练数据集的质量,看看是否需要
     清洗/增强”),不要直接说出内部 skill 名称。
6. **调优**: 把报告中的探针实验(单变量参数修改)+ 数据集质量发现交给
   `generate-workflow-dsl` 生成修改后的训练工作流;用户确认后提交平台执行,
   新 run 完成后回到第 1 步做闭环对比(`compare-runs` 新旧 run)。

## 技术说明

核心分析函数(probe/analyze-run/compare-runs/report)集成自
[wandb-primary skill](https://github.com/coreweave/skills/tree/main/skills/wandb-primary)
(CoreWeave, Apache-2.0):
- `probe_project` / `scan_history` / `diagnose_run` / `compare_configs`
  从官方 wandb_helpers_impl.py 适配,见 `scripts/wandb_helpers.py`
- `diagnose_run` 提供收敛检测、过拟合检测、NaN 检测,替代手写
  `_history_stats`
- `scan_history` 使用 `run.scan_history(keys=...)` 替代 `run.history()`,
  避免大项目(万级 metric) 502 超时
- `resolve-target` / `report` 四阶段模板为 Pyromind 平台特有逻辑

## 大项目性能规则(CRITICAL)

- 始终 `wandb.Api(timeout=120)`,避免默认超时。
- 拉取 history 必须显式 `keys=[...]`,禁止无 keys 全量拉取。
- 列 run 必须 `per_page` 分页;计数用 lazy,不展开所有对象。
- system 指标(如 GPU 利用率)单独 stream 拉取,不与训练指标混拉。

## 已知约束

- GRPO 节点的本地 `WANDB_DIR=/tmp/wandb_logs` run 元数据随容器销毁不可访问,
  不影响云端 wandb 数据。
- 训练节点 stdout 是 run id 的唯一日志来源;若日志接口不可用,回退到
  URL/DSL/时间窗口定位。
- `wandb_config` 中的 `WANDB_API_KEY` 属于 Secret,只写入 creds 文件(600)或
  内存,不得打印到对话/报告。
- 分析结论与平台数据契约见 `references/platform-data-contract.md`;
  四阶段方法论见 `references/analysis-methodology.md`;
  异常 → 探针实验映射见 `references/optimization-playbook.md`。
