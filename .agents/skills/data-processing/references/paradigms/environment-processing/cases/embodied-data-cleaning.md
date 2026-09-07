# 具身机器人数据清洗

本 case 处理 S2 自采集录制或挂载的 Hugging Face LeRobot v2.1 数据，执行
多模态时间对齐、静止帧清理、动作与标签关联、批量转换和 LeRobot v2.1 校验。

执行形态与 [tmax 批量验证](tmax.md) 同构：**EDP 平台编排**（`edp_render` 渲染
分片 → `edp_submit` 逐 episode 执行 → 聚合发布 merge 记录），复用平台
runner 的硬截止、心跳、账本与 verdicts 契约；agent 不持有长任务。

## 执行入口

- 确定性 runtime：`openhands-embodied-runtime==1.29.5`。交付方式：wheel 随
  `edp_submit` 冻结包自动分发（skill bundle `edp/wheels/` 随包暂存进 run 目录，
  exec 经 Storage 挂载安装，第三方依赖由公网 PyPI 解析）。agent 无需任何
  wheel 预置动作；skill bundle 缺 wheel 时 edp_submit 直接报错。若安装仍失败，
  属部署配置错误，停止并上报。
- ProcessingProfile：`scripts/edp/profiles/embodied-cleaning.json`
  （逐 episode 执行）与 `scripts/edp/profiles/embodied-cleaning-merge.json`
  （聚合发布）。
- 数据契约、质量门和执行顺序见 `../../../../../embodied-data-cleaning/SKILL.md`。

## 架构边界

- `data-processing` 负责统一路由和用户确认门禁。
- `embodied-data-cleaning` 描述本 case 的数据契约、质量门和执行顺序。
- `edp_render` / `edp_submit` 是唯一的平台执行通道；**禁止**
  sandbox_create/sandbox_terminal 手工编排、禁止 Studio 工作流、禁止已弃用
  的 `run_embodied_cleaning_sandbox` 专用工具。
- `openhands-embodied-runtime` 在平台沙箱内完成确定性处理；plan/full 与
  episode/merge 共用同一套 CLI（`--mode episode` 单条、`--mode merge` 聚合，
  均支持 `--config <json>` 读挂载配置，避免 shell 引号转义）。
- rejected episode 只输出结构化错误报告（`rejection_diagnostics`），不自动
  repair；重跑由 `edp_submit` 的 shards 断点续跑承载，只重跑 failed。

## 状态与 verdict 语义

| episode 状态 | reward | verdict | 聚合行为 |
|---|---|---|---|
| accepted | 1.0 | usable | 参与合并发布 |
| needs_review | 0.5 | usable（需人工复核标记） | **不参与**合并，人工复核后重跑 |
| rejected | 0.0 | usable（数据级拒绝） | **不参与**合并，诊断进 report |
| failed | 无 reward | error | 断点续跑重试 |

- merge 阶段只合并 accepted 片段（与 batch 全量语义一致），并做 LeRobot v2.1
  最终校验后才发布到 target。
- 全部 rejected：`processing_complete=true` / `published=false` 的终态结论，
  不调用续跑。

## 渲染模板示例

**数据源索引**：LeRobot v2.1 源用 `meta/episodes/chunk-*.parquet`；
**S2 自采集源没有 episode 索引**——先用 Terminal 从 episode 目录清单构建一个
索引 parquet（列：episode_id）并 `upload_file_to_pyromind` 到 Storage
（如 `episodes_index/episodes_index.parquet`），渲染 `data_source` 指向它。
模板 `fields` 只强制 `task_id`；`prompt` 仅 tmax 需要。

```json
{
  "schema_version": 1,
  "name": "embodied-cleaning-render",
  "data_source": "datasets/<name>/meta/episodes/chunk-000.parquet",
  "fields": {
    "task_id": "episode_index",
    "episode_id": "episode_index",
    "image": {"fixed": "pyrominddynamics/jupyter-lab-with-ssh:v0.9"},
    "config_json": {"kind": "json_config", "fields": {
      "episode_id": "episode_index",
      "source": {"fixed": "/target-workspace/datasets/<name>"},
      "work_root": {"fixed": "/target-workspace/.pyromind-agent/<conversation>/embodied-cleaning/<run-id>"},
      "task_text": {"fixed": "<一次性向用户确认的 dataset 级 task text>"},
      "motion_speed_threshold": {"fixed": 0.02},
      "idle_min_duration_s": {"fixed": 1.5},
      "context_s": {"fixed": 0.5},
      "robot_type": {"fixed": "s2"}
    }}
  },
  "shard_size": 20
}
```

- 挂载契约：平台节点 `/workspace` 读写挂载进容器 `/target-workspace`（由
  profile 的 `volume_mounts` 声明），episode 输出/结果 JSON 直接落在共享
  work_root，沙箱销毁后仍持久——这是跨记录状态（无 checkpoint 文件）。
- 沙箱策略：profile 声明 `execution.sandbox_reuse`。本 case 为
  `per_shard`（全量同镜像 + 确定性 runtime：一个分片共享一个沙箱，记录在
  沙箱内串行执行，runner 终结器统一删除；探针失败自动重建沙箱后继续），
  把 N 次 create/pip-install 摊薄成每分片一次；tmax 每条记录镜像不同，
  保持默认 `per_record` 逐条隔离。小数据集（≤ ~20 episodes）也可直接用
  runtime 的 `--mode full` 单沙箱跑全量，不走编排。
- task text / 阈值是**全数据集一次确认**的门禁产物，作为模板 fixed 值写入；
  用户确认前禁止渲染/提交。
- merge 记录（单条）用 `embodied-cleaning-merge` profile 单独提交，其
  config_json 含 `work_root` / `target` / `expected_episode_ids`。

## 平台约束（必须遵守）

- **沙箱 exec 单命令超时上限 600s**：profile 的 exec 步骤设 590s。单个
  episode 的清洗必须能在 590s 内完成（episode 级处理 fits；merge 只做
  拼接+校验，episode 数过大时需先分批或推动平台放宽上限——部署侧已有
  filing）。**绝不**提交 timeout > 600 的 profile，会被
  `SandboxExecRequest` 校验直接拒绝，全部记录 exec_failed。
- `edp_submit` **必须显式传 `profile_name`**；不传会默认冻结
  `tmax-validation` profile，probe 语义完全不同，整批 probe_failed。

## 门禁与流程

1. 预览探查（preview_dataset）→ 确认源形态（LeRobot v2.1 / S2 自采集）；
   自采集源先构建并上传 episode 索引（见上）。
0. **无需 wheel 预置**：runtime wheel 已内置于 skill bundle，由 edp_submit
   随冻结包自动暂存；提交报"wheel bundle is missing"才是部署配置错误。
2. `edp_render` 渲染 episode 分片（shard_size 与用户确认）。
3. **plan 阶段**：对代表 episode 先小样跑 `--mode plan`（或以 smoke 片代替），
   向用户展示 representative plan + inspection 摘要。
4. 用户一次性确认：task text、子任务区间约定、next-state 动作约定、阈值、
   target path。
5. `edp_submit(manifest=..., limit=3, profile_name="embodied-cleaning")`
   smoke → verdicts 分诊（镜像缺件类 error 单独分桶，可修镜像回收）。
6. 用户确认后分批/全量提交；`df_check_progress` 观察进度。
7. 全部 episode 终态后提交 merge 记录（`embodied-cleaning-merge`），
   Kafka 回调后用 `preview_dataset` 校验 target 发布物与聚合 report。
