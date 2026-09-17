# 具身智能数据清洗架构与流程

## 目标

具身数据清洗是 `data-processing` 下的一个环境型 case。Agent 负责路由、确认和
报告；平台 EDP 编排（`edp_render` / `edp_submit`）负责逐 episode 执行，
与 tmax 批量验证同构；Python 3.10 的 `openhands-embodied-runtime` 负责确定性
数据处理。该流程不通过 Studio 工作流提交具身清洗任务，也没有具身专用的提交
Tool——episode 级执行复用通用的 `edp_submit` + ProcessingProfile。

## 分层架构

```text
data-processing（统一入口）
├── data-cleaning（字段映射、格式转换、简单过滤）
├── data-preparation（DataFlow / LLM 数据准备）
└── environment-data-processing（需要特定运行环境）
    ├── embodied-data-cleaning（Python 3.10、LeRobot v2.1）
    └── 其他 case（JDK、编译器、GUI 等）

通用执行层（EDP 编排，与 tmax 同构）
├── edp_render（episode 分片渲染，CustomCommandCPUNode 平台任务）
├── edp_submit（逐 episode 执行：冻结 runner + profile + 节点 shim）
└── merge 记录（accepted 片段合并、校验、发布）

具身确定性 runtime（沙箱内 CLI）
├── source adapter 与格式探测
├── 多模态时间对齐
├── 静止段检测与可逆 plan
├── episode 级质量隔离（--mode episode 单条 worker）
├── LeRobot v2.1 生成、合并与校验（--mode merge 聚合发布）
└── 结构化 batch/reject report
```

## 平台执行顺序

1. `data-processing` 根据数据结构和任务语义路由到
   `embodied-data-cleaning`。
2. 写渲染模板：`data_source` 指向 LeRobot 源的 `meta/episodes/chunk-*.parquet`，
   每条记录 = 一个 episode（`episode_id` + 钉死镜像 + `config_json` 组合配置），
   `edp_render` 分片落 Storage。
3. plan 阶段：对代表 episode 跑 `--mode plan`（小样沙箱或 smoke 片），向用户
   展示 representative plan + inspection 摘要。
4. 用户一次性确认全数据集 task text、子任务区间、next-state 动作约定、阈值
   和 target path（门禁，确认前禁止渲染/提交）。
5. `edp_submit` 提交验证：平台节点按分片拉起 CUSTOM 沙箱
   （`/workspace` 读写挂载到 `/target-workspace`；profile 声明
   `execution.sandbox_reuse=per_shard`，同一分片的 episode 共享一个沙箱内
   串行执行，探针失败自动重建），runtime 执行 `--mode episode --config
   <挂载配置>`，episode 输出与结果 JSON 直接落在共享 work_root；runner
   冻结了硬截止/心跳/账本，长任务不占用 agent。
6. verdicts 语义：accepted=reward 1.0 / needs_review=0.5 / rejected=0.0
   （均为 usable），failed 无 reward 归 error 桶；镜像缺件类 error 单独分桶。
   断点续跑用 `edp_submit` 的 shards offset，只重跑 failed。
平台约束：沙箱 exec 单命令上限 600s，profile exec 步骤设 590s（单 episode
清洗必须 fits；超限会被 `SandboxExecRequest` 校验拒绝）；`edp_submit`
必须显式传 `profile_name="embodied-cleaning"`（默认会冻结 tmax profile）。

7. 全部 episode 终态后提交单条 merge 记录（profile `embodied-cleaning-merge`）：
   runtime `--mode merge` 只合并 accepted 片段，执行 LeRobot v2.1 校验后发布
   到 target。全部 rejected 为 `processing_complete=true` / `published=false`
   的终态结论。
8. Kafka 回调后用 `preview_dataset` 校验 target 发布物与聚合 report；agent 不
   持有任何沙箱生命周期（runner finally 保证删除）。

## 对齐质量门

主相机相对 state 的首尾偏移、state 流内部空洞，以及主相机相对第二路 RGB/depth
流的领先、落后和内部空洞，采用同一规则：

- `≤100ms`：正常；
- `>100ms 且 ≤500ms`：保留 episode，同时写 warning；
- `>500ms`：拒绝该 episode。

500ms 边界本身允许通过。领先和落后对称处理，内部 state gap 使用相同阈值。

## reject 与批量发布

reject 是 episode 级终态，不是批次失败。以下情况会被拒绝：

- 相机领先、落后或 state 内部空洞超过 500ms；
- 子任务区间重叠、越界或语义结构无效；
- 必需 Parquet、MP4 或声明的视频流缺失/损坏；
- 时间映射不可用；
- state/action schema 与目标 profile 不兼容。

每条拒绝必须写入 `rejected_episode_reports`：

```json
{
  "episode_id": "170958",
  "stage": "timeline_alignment",
  "error_code": "CAMERA_LEADS_STATE_OVER_LIMIT",
  "message": "camera frame 0 precedes state coverage by 0.721000s (limit 0.500000s)",
  "details": {
    "stream": "state",
    "frame_index": 0,
    "direction": "camera_leads_state",
    "observed_gap_s": 0.721,
    "allowed_gap_s": 0.5
  },
  "suggestion": "Check whether camera recording starts before the state stream."
}
```

只要至少一个 episode accepted 且 `failed_episode_count=0`，accepted 子集就会生成
合法 LeRobot v2.1 数据集。报告需要同时给出 discovered、accepted、rejected、
needs_review、failed、frame 和 video 统计，并逐条解释拒绝原因。

若全部 episode 都被拒绝，报告为 `processing_complete=true`、`published=false`。
这是终态质量结论，不应 resume；用户仍会拿到逐条 reject 报告，但没有可交付训练集。

## Runtime 交付

当前交付方式：固定版本 wheel 内置于 skill bundle（`edp/wheels/`），由
`edp_submit` 随冻结包自动暂存进 run 目录，profile exec 从 Storage 挂载路径
安装（第三方依赖由公网 PyPI 解析）；agent 侧零预置，bundle 缺 wheel 时提交
即报错。平台正式部署应演进为以下任一方式：

1. 在 Sandbox 基础镜像中预装固定版本 runtime；或
2. 将固定版本发布到平台内部 Python 包源，由 Sandbox 安装。

不能把用户手工上传 wheel 作为产品流程的前置条件。若平台没有提供固定 runtime，
Agent 应返回“部署配置缺失”，而不是向终端用户索要 wheel 路径。

## 最终输出

训练目录只包含：

```text
target/
├── meta/
├── data/
└── videos/
```

plan、episode 结果 JSON、reject 报告和中间 episode 输出保存在共享 work_root
（挂载持久），不发布到训练目录。完成条件是 merge 阶段 `complete=true`、
`published=true`、最终 validation 有效，且目标目录结构与统计一致。
