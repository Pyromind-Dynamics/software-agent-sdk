# 具身智能数据清洗架构与流程

## 目标

具身数据清洗是 `data-processing` 下的一个环境型 case。Agent 负责路由、确认和
报告；通用 Sandbox 工具负责容器生命周期；Python 3.10 的
`openhands-embodied-runtime` 负责确定性数据处理。

该流程不通过 Studio 工作流提交具身清洗任务，也没有具身专用的提交 Tool。

## 分层架构

```text
data-processing（统一入口）
├── data-cleaning（字段映射、格式转换、简单过滤）
├── data-preparation（DataFlow / LLM 数据准备）
└── environment-data-processing（需要特定运行环境）
    ├── embodied-data-cleaning（Python 3.10、LeRobot v2.1）
    └── 其他 case（JDK、编译器、GUI 等）

通用执行层
├── sandbox_create / sandbox_delete
├── sandbox_terminal
├── sandbox_read_file / sandbox_write_file / sandbox_delete_file
└── sandbox_upload / sandbox_download

具身确定性 runtime
├── source adapter 与格式探测
├── 多模态时间对齐
├── 静止段检测与可逆 plan
├── episode 级质量隔离与 checkpoint
├── LeRobot v2.1 生成、合并与校验
└── 结构化 batch/reject report
```

## 平台执行顺序

1. `data-processing` 根据数据结构和任务语义路由到
   `embodied-data-cleaning`。
2. Agent 使用 `pyrominddynamics/jupyter-lab-with-ssh:v0.9` 创建 Python 3.10
   CUSTOM Sandbox，并将平台 Storage 的 `/workspace` 挂载到容器
   `/target-workspace`。具身 case 不使用示例或临时 registry 镜像。
3. Sandbox 验证 `openhands-embodied-runtime==1.29.5`；若标准镜像未预装，
   则从部署包源或部署挂载的 wheel 安装。普通用户不需要上传 wheel。
4. 运行 `mode=plan`，检查代表 episode、源结构、流状态和可逆清洗计划。
5. Agent 一次性向用户确认全数据集 task text、子任务区间、next-state 动作约定和
   target path。
6. 新建 Sandbox，使用相同 run ID 和参数执行 `mode=full`。长任务用后台进程、PID
   和日志轮询判断进度。
7. runtime 逐 episode 处理并在每条完成后写 checkpoint。一个 episode 出错不会
   否定整个批次。
8. 没有 runtime failure 时，所有 accepted episode 被合并、校验并发布；rejected
   episode 留在审计报告，不进入训练目录。
9. 仅当存在 `failed`（运行时异常）时使用 `mode=resume`。resume 只重试 failed，
   不重新处理 accepted 或 rejected。
10. Agent 读取最终报告、校验目标 `meta/data/videos`，并删除 Sandbox。

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

开发时 wheel 可用于本地或预发布验证；平台正式部署应采用以下任一方式：

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

plan、checkpoint、日志、reject 报告和中间 episode 输出保存在 audit 目录，不发布到
训练目录。完成条件是 `report.complete=true`、`published=true`、最终 validation
有效，且目标目录结构与统计一致。
