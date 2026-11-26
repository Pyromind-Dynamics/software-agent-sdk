# 编排式场景数据处理(environment-processing)

启用 SDK 时，先读取[SDK 提交与回调](../../../../workflow-execution/references/sdk-execution.md)，
渲染、分片执行和聚合均由普通准备函数构图，通过服务端校验后调用 SDK。
SDK Main 不再注入 `edp_render`、`edp_submit`、`edp_aggregate`。
本页示例展示场景业务参数，不需要统一计划外壳。
运行依赖未就绪时先修复环境，不重新加入提交 Tool。
保留每片回执，部分成功后不整批重投。

- **定位**:编排类的具体场景数据处理——单条数据的处理要在数据自带的
  特定环境(镜像)中执行复杂流程(跑命令/跑测试/判定),因此编排为
  多阶段链路:EDP 渲染任务 渲染分片 → EDP 分片任务 逐条执行 →
  EDP 聚合任务 聚合训练文件(CustomCommandCPUNode 平台任务,agent 只走
  控制面,不持有全量数据);冻结运行时 = `sandbox_runner.py` +
  ProcessingProfile(`scripts/edp/profiles/`),支持断点续跑与强制清理
- **适用**:编排复杂度超出单算子/单次转换——批量逐条判定/验证/筛选,
  且执行需要特定环境(JDK 版本、特定工具链、数据自带镜像)
- **不适用**:纯格式转换/字段映射/简单过滤 → format-conversion 处理范式;
  抽样/清洗/生成/评分等平台 DataFlow 处理 → llm-pipeline 处理范式;
  仅创建/管理单个沙箱容器 → sandbox skill

## 场景 case 路由(先做这一步)

本范式按"处理什么任务"细分场景 case;接到任务先查下表,命中即读对应
case 文档(case 文档描述数据形态、字段映射与解题链路细节,流程与门禁
仍以本 playbook 为准):

| 数据形态 / 任务 | case 文档 | 执行模式 |
| --- | --- | --- |
| 成批记录,每条含镜像引用 + 题面 + verifier(tmax 类终端任务可用性/可行性验证) | [tmax 批量验证](cases/tmax.md)(profile: `scripts/edp/profiles/tmax-validation.json`) | 模式 A |
| S2 自采集或 LeRobot v2.1 数据的多模态对齐、静止帧清理、转换和校验 | [具身数据清洗](cases/embodied-data-cleaning.md)(profiles: `scripts/edp/profiles/embodied-cleaning.json` + `embodied-cleaning-merge.json`) | 模式 A |
| 无匹配 case | 无;按本 playbook 通用流程(模式 B 面向一次性/探索性任务) | 见"执行模式选择" |

## 执行模式选择

再看 `scripts/edp/profiles/` 目录下是否有匹配当前任务的 ProcessingProfile:

| 条件 | 执行模式 |
| --- | --- |
| **有匹配 profile** | **模式 A:冻结运行时 runner**,见下文"模式 A"(沙箱策略由 profile 的 `execution.sandbox_reuse` 声明:`per_record` 逐条隔离,`per_shard` 同镜像分片共享) |
| 无匹配 profile(一次性/探索性任务) | 模式 B:对话组装沙箱工具,见下文"模式 B" |

Profile 是声明式的(steps + verdict + output),控制流(逐条循环、镜像去重、
断点续跑、沙箱清理)全部冻结在 runner 里;没有匹配 profile 时**不要**现场改
runner,改用模式 B 对话组装。

## 前置条件

- 目标环境可用镜像已就绪:可直接拉取的容器镜像引用,如 `eclipse-temurin:8-jdk`
- 当前 SDK 执行环境已提供 `PYROMIND_API_KEY`、`PYROMIND_BASE_URL` 与集群。
  Worker 使用 `--api-key` 创建原生 SDK client，不传当前请求 Cookie/JWT。
- 解题 profile 按需使用当前环境的 `LLM_BASE_URL` / `LLM_AUTH_TOKEN` / `LLM_MODEL`，
  runner 通过 `--set` 接收，与 profile 的 `{secret:...}` 声明对应；纯环境 profile
  不要求 LLM。不要把明文密钥写进对话、manifest、脚本或冻结图。
- 含 runtime 环境声明的图必须通过普通 `submit_workflow` / `submit_workflow_batch`：
  helper 只在 SDK POST 前复制图并绑定必要环境值，最终 Studio 任务定义可能保存这些
  command 环境。这不是 Secret 服务，不保存、打印或回传绑定后的请求图。

---

## 模式 A:冻结运行时(profile 匹配时)

适用:成批记录(每条 = 镜像 + 题面 + verifier)、逐条独立判定、可断点续跑的
验证/处理任务。manifest 通用契约:`task_id` / `image`(docker 镜像引用)/
`workdir` / `prompt`(题面)/ `test_sh`(verifier 全文);各 case 的字段映射
与数据源结构见对应 case 文档(如 [tmax](cases/tmax.md))。

### 全链路三段平台化(agent 只走控制面)

manifest 的构造、逐条验证、最终聚合**全部在平台节点执行**,agent 全程
不出现本地 manifest.jsonl / verdicts.jsonl / 训练文件:

```text
看字段(直读 /workspace/) → 写/确认 render 模板 JSON(几 KB)
  ① EDP 渲染    → 平台节点:读 parquet → 分片写 batch-XXX/manifest.jsonl + shards.json
  ② EDP 分片    → 平台节点:逐条起镜像验证 → verdicts.jsonl + traces/
  ③ EDP 聚合 → 平台节点:跨片合并去重 + 转 slime/SFT → 训练文件落 storage
每段:增量 append+flush、progress.json、节点日志逐条一行、同 run/out 目录断点续跑
```

### 0. 提交后简报与等待纪律(硬约束,贯穿三段)

每次渲染、验证或聚合计划提交成功后,必须**立即**
向用户发一条简报(提交了什么 / task_id / 预计量级 / 终态回调后自动汇报
verdict 或 report),然后**结束当前回复**,把控制权交回平台回调。
**禁止**用 sleep 终端命令或反复查看输出目录来消磨等待
(对话 b32487cc 教训:agent 等待期间 23 次目录轮询 + 11 次 sleep 轮询
约 6 分钟,用户全程零信息)。中途确需看进度按
[挂载进度参考](../../storage-progress.md) 读取已知输出目录一次，不反复列目录；终态回调会自动唤醒会话，回调之前
agent 不需要做任何事。

### 1. 渲染分片(manifest 由平台节点构造)

**数据源定位(先做这一步)**:用户给出的数据路径(如 `/workspace/datasets/...`)
一律是 Storage 路径,**第一步先探索目录结构**。Storage 已挂载为 `/workspace`,
直接用 `read`/`terminal` 查看(`/workspace` 绝对路径)。数据源的目录结构与字段映射
见对应 case 文档(如 [tmax](cases/tmax.md))。

**调研纪律(硬约束)**:写模板前的调研只允许两类动作——查看数据(用
`read`/`terminal` 直读 `/workspace/...`)与读本 skill 文档/源码。单文件超限时
**不要**把全量数据拷进会话工作区查 schema——那会把全量数据拉回 agent 侧,
违背三段平台化的控制面原则;正确动作是先向用户提问,说明缺口
与候选方案,由用户决策。字段来源表达不了时先查下方 spec 能力表
(`message`/`storage_file` kind 与点号嵌套下钻覆盖 chat 格式、逐任务目录
与 struct 嵌套形态),仍表达不了再问用户,不要绕开平台自行构造 manifest,
也不要另起 pipeline 先把 parquet 展平再渲染。

**探查预算(硬约束)**:写模板前的探查控制在 **8 次查看以内**(1 次列数据集
根目录、1-2 次定向看数据文件列名/样本,其余按 case 文档补看关键路径);
一次查看拿到的列名与嵌套结构直接记下来写模板,**禁止反复查看同一文件调 n**;
探查预算按数据集一次性用完,不要对同一数据集反复重看。

**大规模探查委派**:当数据集**无 case 文档**且预计需要 **>10 次
查看** 才能摸清结构时,改用 `subagent(type=data_explorer)` 委派
批量探查——主 agent 只接收四段式交接(数据集结构/逐字 schema/一条
原始样本/注意点),探查噪声不进入主链路;任务描述写清要摸清哪些路径、
回答哪些问题。有 case 文档或 8 次以内能完成的验证型探查**不委派**,主
agent 直接查更快。拿到的 schema 用于写模板,列名以交接报告逐字内容
为准,渲染预检报错时再定向回看。

**出渲染模板 JSON**(字段映射 + 分片大小,几 KB;**shard_size 需与用户
确认**)。最小示例:

```json
{
  "fields": {
    "task_id": "task_id",
    "image": {"fixed": "<docker 镜像引用>"},
    "workdir": {"fixed": "/workspace"},
    "prompt": "description",
    "test_sh": {"kind": "storage_file",
                 "path_template": "datasets/<name>/task-data/{task_id}/tests/test.sh"}
  },
  "shard_size": 500
}
```

字段 spec 通用形态(`fields.*` 每项都支持):

| spec | 语义 |
| --- | --- |
| `"col"` 或 `{"field": "col"}` | parquet 列原值;支持点号下钻 struct 嵌套列(如 `env_config.task_id`,open-instruct 形态 task_id/image 嵌在 struct 里时直接写,不需要先展平)(缺列整批报错,不猜) |
| `{"fixed": 值}` | 常量 |
| `{"join": {"source", "on", "column"}}` | 从 storage 映射表(JSONL/CSV)按 task_id 查值;未命中该行进 `render_failures.jsonl` |
| `{"kind": "message", "source_field": "messages", "role": "user", "index": "first"\|"last"}` | 从 chat 格式 list 列抽指定角色消息文本(open-instruct 形态的 prompt 抽取) |
| `{"kind": "storage_file", "path_template": "datasets/.../task-data/{task_id}/tests/test.sh"}` | 逐行读 storage 文件全文作字段值;**path_template 相对 storage 根解析**(即完整 storage 路径,如 `datasets/allenai/tmax/task-data/...`,不是相对数据集目录);占位符引用已渲染字段(如 `{task_id}`) |
| `{"kind": "pytest_wrapper", "source": <上述任意 spec>, "target_path": ...}` | 把 source 解析出的 python 源码包装成 pytest verifier 脚本;兼容旧写法 `source_field: <列名>` |

行级问题(join 未命中/文件缺失/消息缺失)只跳过该行并逐条记
`render_failures.jsonl`,不中断整批渲染;模板级错误(缺列/spec 非法)
fail fast。完整示例(join、pytest_wrapper、chat 格式)见
[tmax case](cases/tmax.md)。

EDP 渲染任务的业务参数示例：

```json
{
  "template_path": "public_data/render_template.json",
  "data_source": "/workspace/datasets/source/train-*.parquet",
  "shard_size": 500
}
```

- 准备前取得数据源 parquet schema，作为 `prepare_edp_render` 的 `source_schema`
  做**模板预检**：对照 schema 校验模板
  引用的列名(含 struct 嵌套下钻),列名写错秒级报错并给出可用列清单,
  不再浪费一轮平台任务(glob 数据源/预检自身失败时跳过预检,节点端
  渲染仍会 fail fast)

- 模板用 `file_editor`/`apply_patch` 写入会话工作区(如
  `public_data/render_template.json`),`template_path` 直接传同一相对路径
  （`LocalFiles` 按指定工作目录解析）；**不要**先上传 storage 再传
  storage 路径,那多绕一步

- 节点分段读 parquet(pyarrow iter_batches,内存≈单片),分片写 storage
  (`<output_root>/batch-XXX/manifest.jsonl`),输出 `shards.json` 索引 +
  `render_failures.jsonl`(join 缺失任务)+ `progress.json`/`report.json`,
  节点日志逐条一行(`[i/N] task_id=... -> batch-XXX`)
- `data_source` 支持 glob(`train-*.parquet`)与目录(取其下 `*.parquet`)
- 渲染内存 ≈ 单片,10W+ 级数据源同样适用;agent 不下载 parquet

**门禁(硬约束)**:渲染完成后**立即用第一片提交平台 smoke**——禁止本地
核验/手工逐条编排/反复查看目录。已有产物复用规则:同一批数据未完成 →
直接消费旧产物或同 run 续跑;数据变了 → 重新渲染后新 run 提交。

### 2. 提交验证(smoke → 用户决策 → 分批/全量)

验证以 **CustomCommandCPUNode 平台任务**提交执行（EDP 分片任务 SDK 提交）:
调用 `pyromind_common.platform.environment` 的 `EdpSubmitOptions`、
`select_edp_manifests` 与 `prepare_edp_shard`，冻结 `sandbox_runner.py` + profile +
`pod_runtime/`（节点 shim）及所需 wheels。每片一个独立 workflow；保留准备结果中的
`manifest`、`run_id`、`output_dir`、`resumed`，对应 SDK 响应的真实 `task_id`。

**2.1 smoke(第一片,limit=3)**

EDP 分片任务的业务参数：

```json
{"manifest": "/workspace/edp/batch-001/manifest.jsonl", "limit": 3}
```

单片 `manifest` 模式直接调用 `select_edp_manifests(options)`。为该片选定 run UUID，
使用 `StagingDirectory(local, storage=<manifest父目录>/<run UUID>)`，再调用
`prepare_edp_shard(options, manifest=..., staging=..., runtime_dir=..., run_id=...,
platform_env=..., cluster=...)`。`runtime_dir` 指向 `scripts/edp`。
冻结文件发布到对应 Storage 位置后，将图转 DSL 交给保留的 `validate_workflow_dsl`，
读取实际校验图，再用普通 `submit_workflow` 创建一次；不要把含环境声明的图直接交给
`studio.create`。

- Kafka 终态回调自动唤醒会话(Succeeded/Failed),随后查看
  `<output_dir>/run/verdicts.jsonl`(task_id / verdict / exit_code /
  error_category / reward / note)与 `<output_dir>/run/traces/`(解题轨迹):
  用 `read`/`terminal` 读 `/workspace/<output_dir>/run/...`
- 运行中按[挂载进度参考](../../storage-progress.md)，从实际 `<output_dir>/run` 调用
  普通 `read_progress(..., tail_filename="verdicts.jsonl")`，查看计数、ETA 和最近 verdict；节点日志逐条一行
  (`[i/N] task_id=... verdict=... reward=...`)
- 按 verdict 分布汇报:usable 数、error 分类计数(error_category 枚举以
  profile 的 verdict 定义为准,如 [tmax](cases/tmax.md));**镜像缺件类错误
  (如 verifier_env_missing)单独分桶**——镜像缺件可修,非数据不可用
- smoke 本身就是一个批(单片 3 条):终态后同样自动聚合(见第 3 节),
  sft/slime 即时产出——用户要看"构建出的数据"零等待,转换层信号
  (如 reward<1.0 的 usable 记录不进 SFT)在烧全量前就暴露

**2.2 用户决策门禁(硬约束,须向用户提问确认)**

smoke 终态回调并汇报 verdict 分布 + 单条均时后,**必须向用户提问确认,
让用户三选一,禁止 agent 默认走全量**:

- ① **串行全量**(默认;每次提交一片,终态回调后检查再提交下一片)
- ② **显式并发批**(仅当用户明确要求;N=2-5。agent 给推荐值与推算依据:
  单条均时 × 片内条数 vs 12h 任务配额、同镜像拉取风暴约束——3 条样本
  不符合预期的全量成本很高,并发容量由用户拍板)
- ③ **先停**(修模板/镜像/join 源后再来)

**2.3 多片提交(分批与全量统一形态)**

EDP 分片任务的业务参数（片数按本次用户授权填写）：

```json
{"shards": "/workspace/edp/shards.json", "shard_offset": 0, "shard_count": 1}
```

- 从 Storage 取得 `shards.json` 的小型索引，传给
  `select_edp_manifests(options, shard_index=bytes)`，按选定路径逐片准备和校验。
  每片独立 workflow/output_dir/终态回调；`shard_count` 省略 = 1，
  只有用户明确要求并发时才能传 2–5，输入模型拒绝超过 5 片。
- 可以使用 `submit_workflow_batch` 依次提交实际校验图。首次提交失败后立即停止，
  `WorkflowBatchSubmissionError.accepted` 保留已受理的响应；按顺序对应准备结果，
  保存每片真实 task ID、run ID、manifest、输出目录。后续未提交的片不要记为运行中，
  部分成功后不整批重投。
- 片 run 目录嵌套在片 manifest 目录下(`<root>/batch-XXX/<run_id>/run/`),
  checkpoint 与 manifest 同片共存;某片中断,同 `run_id` 重提交该片即续跑
  （`EdpSubmitOptions.run_id` 传原 UUID，保留同一 staging 及冻结文件；
  verdicts.jsonl 为 checkpoint，自动跳过已判条目）。恢复时核验冻结资产 hash，
  不用当前部署的新 runner 覆盖原运行库。
- 每个回调会唤醒会话。串行模式(`shard_count=1`)检查该片并提交下一片;
  并发模式先判断"**本批是否全部片终态**"(对照本次提交的 task_ids),
  未到齐就只汇报进度继续等,到齐才进 2.4

**2.4 批间确认(硬约束)**

并发批全部片终态后:汇报本批统计(usable/error 分布、与前批对比、单条均时)
→ **再次向用户提问确认:继续下一批 / 调整批容量 / 停止**。不自动推进
下一批。与既有 `Terminated` 回调约定一致:终止/失败不得自动重提交,
交用户决策；需介入时先停止平台任务，启用 SDK 时按
[停止参考](../../sdk-stop-task.md) 执行，其他会话使用 `df_stop_task(task_id)`，
确认终态后再改再提交。

冻结脚本应兼容平台节点 Python 3.10，使用随包运行时。平台与模型配置从当前授权执行
环境取值，提交前在内存绑定，不写入本地冻结资产或计划。沙箱内解题与网关适配按对应 case，
如 [tmax](cases/tmax.md)。

### 3. 聚合训练文件(EDP 聚合,平台节点)

**每批片全部终态后 agent 自动聚合**(非用户触发;smoke 单片即首批):
对本批全部 run_dirs 跑一次增量 EDP 聚合任务 到同一 out_dir,训练文件随批增长;
聚合幂等(同 out_dir 断点续跑 + task_id 去重),重跑无副作用。
用户只在**换参数重派生**时才手动指定新 out_dir 重跑——验证(逐条起沙箱)
贵、转换便宜,`min_reward`/`system_prompt` 等训练侧参数应看完 verdict
分布后再定,换参数重派生不需要重跑昂贵的沙箱验证:

EDP 聚合任务的业务参数：

```json
{
  "run_dirs": ["/workspace/edp/batch-001/run-1"],
  "out_dir": "/workspace/edp/merged",
  "min_reward": 1.0
}
```

- 节点逐条合并各片 verdicts(去重 task_id)并同趟转换:
  `<out_dir>/verdicts.jsonl`(合并 checkpoint)、`slime.jsonl`(usable →
  slime RL)、`sft.jsonl`(reward ≥ min_reward 的轨迹 → SFT)、
  `progress.json`（从已知挂载目录有界读取）、`report.json`
  (verdict 分布、error_category 计数、verifier_env_missing 单独分桶、
  转换/跳过计数、重复 task_id 数)
- **断点续跑**:`<out_dir>/verdicts.jsonl` 已有 task_id 即 checkpoint,
  中断后同 out_dir 重提交(可追加新 run_dirs)自动续,不重复不丢
- **slime RL 三键**:只转 `verdict=usable` 记录;reward 不嵌入,rollout 时
  由 `metadata.test_sh` 实时判定(usable 但 reward 0 的记录保留——RL 需要
  0 分信号);字段映射见 slime-conversion.md
- **SFT messages 格式**:默认只转 `reward >= 1.0` 的解题成功轨迹
  (`min_reward` 可调);轨迹事件流解析与题面前置规则见对应 case 文档
  (如 [tmax](cases/tmax.md))
- **到此为止,无需 format-conversion 处理范式**:`<out_dir>/slime.jsonl` 与
  `sft.jsonl` 即交付物,直接对接训练;仅当需要去重/PII/语言过滤等清洗算子时
  才衔接 format-conversion 处理范式(多数 case 用不上)
- 聚合节点纯标准库、无凭据注入;判定规则与批次策略见
  batch-orchestration.md

### 4. 本地复现（仅工程师调试，Agent 路径提交平台任务）

工程师需要本地复现单条行为时可直接跑冻结运行时(agent 不要走此通道):

```bash
python scripts/edp/sandbox_runner.py \
    --profile scripts/edp/profiles/<case>.json \
    --manifest /path/to/manifest.jsonl \
    --output-dir /path/to/run-dir \
    --env pre --cluster us-west-1 \
    --api-key "$PYROMIND_API_KEY" \
    --set LLM_BASE_URL="$LLM_BASE_URL" \
    --set LLM_AUTH_TOKEN="$LLM_AUTH_TOKEN" \
    --set LLM_MODEL="$LLM_MODEL" \
    [--limit 3]
```

- `--set KEY=VALUE` 注入 LLM 凭据(profile 中以 `{secret:KEY}` 占位),
  值不会写进 verdicts;缺失时该条记 error(exec_failed)
- 解题 agent 等长任务以后台 + 退出码文件轮询执行(exec 单命令 600s 上限
  扛不住),`timeout` 参数为解题总预算(默认 1800s)
- **清理硬约束**:runner 对每条记录在 finally 中强制删除沙箱(Running 态
  会先 pause 再删),profile 写错只会让该条记为 error,不会泄漏沙箱

---

## 模式 B:对话组装沙箱工具(无匹配 profile 时)

系统消息启用 SDK 时，按 sandbox Skill 的
[生命周期](../../../../sandbox/references/sdk-lifecycle.md)、
[写入](../../../../sandbox/references/sdk-file-write.md) 和
[命令](../../../../sandbox/references/sdk-commands.md) 参考完成同样步骤。
下列 Tool 示例用于对应工具仍在注入的会话；均为独立工具，无 `operation` 字段。

### 1. 明确需求与环境要求

- 确定任务需要的运行时(JDK 版本、Python 版本、GPU 等)
- 确认镜像引用;不确定时先询问用户,不要猜测

### 2. 启动沙箱

```
sandbox_create(image="eclipse-temurin:8-jdk", cpu=4, memory="8Gi", wait_timeout=300)
```

- 可选:`name`、`volume_mounts`(如
  `[{"host_path": "/workspace", "mount_path": "/data"}]`,之后读写文件用
  容器内路径 `/data/...`)、`port_mappings`(如 `[{"container_port": 8080}]`)
- 记录返回的 `sandbox_id`(后续所有操作都要用到)
- 若返回失败,修正参数后重试,不要继续后续步骤

### 3. 探测并安装环境

```
sandbox_terminal(sandbox_id=<id>, command="java -version")
```

- 先探测:`java -version` / `python3 --version` / `node --version`
- 缺少依赖时用 `apt-get install` / `npm install -g` 安装(沙箱内网络可用)
- 每条命令在全新 shell 中执行(不保留 cd/环境变量);需要切目录用 `cwd` 参数

### 4. 写入处理脚本

```
sandbox_write_file(sandbox_id=<id>, path="/workspace/process.sh", content="#!/bin/bash\n...")
```

- 脚本一律写到工作目录(如 `/workspace/`),**不要写 `/tmp`**(挂载/清理语义
  不稳定,已有踩坑记录)
- 不再使用 `cat > ... <<'EOF'` heredoc 方式写大文件

### 5. 执行并观察

```
sandbox_terminal(sandbox_id=<id>, command="bash /workspace/process.sh", cwd="/workspace", timeout_seconds=600)
```

- 返回合并输出与 `returncode`;`timed_out=True` 表示命令超时
- 长任务改为后台 + 轮询日志:`nohup bash /workspace/process.sh > /workspace/run.log 2>&1 &`,
  之后 `sandbox_terminal(command="tail -n 50 /workspace/run.log")` 观察进度

### 6. 交付产物

Main 和目标沙箱共享同一 Storage 时，将结果写入已确认的挂载目录，再核对大小和 SHA-256。
Main 直接使用实际 `/workspace/...` 路径，不经过上传或下载 Tool 中转。
目标沙箱没有相同挂载时，使用 SDK `read_file` / `write_file` 传递必要文件，先读取 sandbox
Skill 的文件操作参考，核对目标内容后再继续；不能把不同容器内相同的路径字符串当成同一文件。
用户交付链接仍通过 `get_storage_url` 获取，不将容器路径作为浏览器链接。

### 7. 清理沙箱(硬约束)

```
sandbox_delete(sandbox_id=<id>)
```

> 无论任务成功还是失败，最后都释放沙箱；启用 SDK 时用 SDK，其余会话用 `sandbox_delete`。
> 删除失败时重试一次;确实无法删除时必须明确告知用户。

## 注意事项

- 每一步都检查上一步返回的 `status`/`returncode`,失败即修正,不要盲进
- 系统消息启用 SDK 时，文件操作按 sandbox Skill 的
  [读取参考](../../../../sandbox/references/sdk-file-read.md) 与
  [删除参考](../../../../sandbox/references/sdk-file-delete.md) 执行。
  其他上下文使用已启用的 `sandbox_read_file`、`sandbox_delete_file`；
  二进制按 base64 展示，删除非空目录须显式 `recursive=True`
- custom 镜像必须自带 bash(probe/脚本依赖它);镜像缺 bash 会在探测阶段暴露
- 沙箱内文件路径以沙箱视角为准;storage 路径与 upload 的目标目录分离
- 若需 VNC 图形交互,用创建返回的 `web_vnc_url`(供用户人工介入)
