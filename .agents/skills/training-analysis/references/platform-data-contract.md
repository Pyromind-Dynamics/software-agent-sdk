# Pyromind 平台 wandb 数据契约

平台训练工作流通过 `WandbConfigBuilderNode` 配置 wandb,训练节点注入环境变量后由
训练框架 `wandb.init()` 记录。本文件描述数据形状与定位规则(已按 task 7710
真实校准)。

## 定位策略(优先级从高到低)

1. **节点 config 直接取凭证**: `task_workflow_result` 响应自带全部节点 config——
   训练节点的 `data.config.wandb_config` 文本块(含 `WANDB_API_KEY`/`WANDB_PROJECT`),
   或 `WandbConfigBuilderNode` 的 `data.config` 结构化字段
   (`wandb_api_key`/`wandb_project`/`wandb_name`)。无需额外接口。
2. **训练节点日志提取 run id**: 训练节点(节点类型含 `Train`,如
   `ModelTrainSFTNode`)→ `internal/logs/node/raw?nodeId={node_id}&taskId={task_id}`
   → stdout 中 `wandb: setting up run <run_id>`(或 `View run at` 行中的 url)。
3. **WandbConfigBuilderNode 输出接口(回退)**: config 未含凭证时查
   `internal/output/node/raw?node_code={node_id}&task_id={task_id}`,输出文本含
   `wandb_config:` 块。
4. **用户 run URL**: `https://wandb.ai/{entity}/{project}/runs/{run_id}`。
5. **DSL 匹配**: `wandb_project`(project 名)+ `wandb_name`(display_name)过滤。
6. **时间窗口**: 按任务执行时间匹配最近 finished runs。

entity 由 wandb API `viewer()` 推断(如 `pengtao-shi-pyromind`),或 `--entity` 指定。

## 平台 API 端点

| 端点 | 参数 | 用途 |
|---|---|---|
| `GET {api_base}/api/task_workflow_result` | `task_id` | 节点列表(节点 id/类型/config) |
| `GET {api_base}/internal/logs/node/raw` | `nodeId`, `taskId` | 节点 stdout 日志 |
| `GET {api_base}/internal/output/node/raw` | `node_code`, `task_id` | 节点输出(含 wandb_config) |

- `{api_base}` 默认按 `APP_ENV` 推断,与 `validate_workflow` 端点选择一致:
  prod/production/online → `https://api-portal.pyromind.ai/std2/studio_api/`;
  其他(含 dev/空)→ `https://pre-api-portal.pyromind.ai/std2/studio_api/`;
  可用 `PYROMIND_API_BASE` 或 `--api-base` 覆盖。
- 认证与 `validate_workflow` 工具一致: `cookie`(`PYROMIND_COOKIE`)、
  `x-cluster`(`X_CLUSTER`)、`authorization`(`PYROMIND_AUTHORIZATION`)三个
  header 小写透传,也可用对应 `--cookie`/`--cluster`/`--authorization` 参数。
- 请求需浏览器风格 `User-Agent`,否则 Cloudflare 返回 403
  `browser_signature_banned`(脚本内置)。

## 响应形状(真实校准, task 7710)

- **task_workflow_result**: 顶层 `{path, workflow, task_status}`;节点在
  `workflow.nodes[]`,节点字段为 `id`(字符串)、`data.nodeType`
  (如 `ModelTrainSFTNode`/`WandbConfigBuilderNode`)、`data.config`
  (含 `wandb_config` 文本块或 `wandb_api_key` 等结构化字段)。
- **节点日志/输出**: `{size: {cols, row}, entries: [{t, m}]}`,
  文本在 `entries[].m` 逐段拼接。

## 数据形状

- **config 键**: 训练 argv 超参(真实样例: `learning_rate`=0.0001、
  `num_train_epochs`=2、`per_device_train_batch_size`=2、
  `gradient_accumulation_steps`=2、`weight_decay`=0、`optim`=adamw_torch_fused)。
- **指标键**(真实样例): `train/loss`、`train/entropy`、`train/epoch`、
  `train/global_step`、`train/grad_norm`、`train/learning_rate`、
  `train/mean_token_accuracy`、`train/num_tokens`、`total_flos`。
- **summary**: 各指标终值;真实 run `sulnf0t5`(SFT, 10 步)的
  `train/loss` 0.96 → 0.84,无 NaN/尖峰。
- 训练节点 stdout 示例:
  ```
  wandb: setting up run sulnf0t5
  wandb: Run data is saved locally in /workspace/.../wandb/run-20260805_060652-sulnf0t5
  ```

## wandb SDK 兼容要点(wandb 0.28 实测)

- `run.history()` 返回 pandas DataFrame,`list()` 得到的是列名而非行;
  需 `to_dict("records")`(脚本已统一处理)。
- `run.summary` 为 `SummarySubDict`,直接迭代触发 `__getitem__` 抛 `KeyError: 0`;
  必须先转 `dict()`(脚本已统一处理)。
- `dict` 子类(`SummarySubDict`/`Config`)需递归转原生类型后才能 `json.dumps`(脚本已处理)。

## 已知约束

- DSL 中不存在 run id(8 位 hash 为训练时生成);wandb_name 为空时 run 显示名为随机名。
- GRPO 节点本地 `WANDB_DIR=/tmp/wandb_logs` 的 run 元数据随容器销毁不可访问,
  云端 wandb 数据不受影响。
- 已知缺陷: `run_gkd_node` 及所有 `*_test` 变体未应用解析出的环境变量,
  GKD 节点 wandb 实际不生效(仅记录,不修复)。
- 平台侧更彻底的增强建议: 训练节点在输出中增加 `wandb_run_url` 字段
  (本次未实现)。
