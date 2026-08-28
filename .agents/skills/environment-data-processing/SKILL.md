---
name: environment-data-processing
description: >
  特定环境数据处理与数据可用性验证:当任务需要真实运行环境——按数据集
  自带的 docker 镜像逐条起沙箱、安装工具链、跑通任务并执行 verifier 判定
  出分——时使用。覆盖场景:tmax 类终端任务数据的逐条可用性/可行性验证
  (起镜像→安装 pi coding agent→pi headless 解题→同箱跑 test_sh→读 reward),
  JDK 等特定运行时下的代码筛选与处理。全链路三段平台化(edp_render 渲染分片
  → edp_submit 逐条验证 → edp_aggregate 聚合训练文件),agent 只走控制面,
  不持有全量数据;声明式 ProcessingProfile + 冻结运行时支持断点续跑与强制
  清理;无匹配 profile 时对话组装沙箱工具。纯格式转换/字段映射用
  data-cleaning; 不起沙箱的抽样/清洗/评分用 data-preparation;仅创建/管理
  单个沙箱容器用 sandbox。
triggers:
- tmax
- 可用性验证
- 可行性验证
- 终端任务
- 环境数据处理
license: MIT
---

# 特定环境数据处理(Environment Data Processing)

## 适用边界

| 场景 | 工具 |
| --- | --- |
| 仅格式转换/字段映射/简单过滤,无环境要求 | data-cleaning |
| 抽样/清洗/生成/评分,平台 DataFlow 处理 | data-preparation |
| **任务对运行环境有硬性要求(JDK 版本、特定工具链、批量逐条验证等)** | **本 skill** |

典型场景:
- 筛选仓库中仅使用 Java 1.8 特性的代码 —— 需要 JDK 8 环境才能准确编译/解析验证
- tmax 类终端任务数据可用性/可行性验证 —— 逐条按数据自带镜像起沙箱,安装
  pi coding agent 解题,同箱跑 verifier 判定 usable/error
  (模式 A,`profiles/tmax-validation.json`)

## 执行模式选择(先做这一步)

先判断数据形态:成批记录且每条含 `image`(docker 镜像引用)+ 题面 +
verifier/test_sh → 属于"逐条环境验证"场景,匹配
`profiles/tmax-validation.json`,**无需用户点名数据集类型**。

再看 `profiles/` 目录下是否有匹配当前任务的 ProcessingProfile:

| 条件 | 执行模式 |
| --- | --- |
| **有匹配 profile**(如 tmax 批量验证 → `profiles/tmax-validation.json`) | **模式 A:冻结运行时 runner**,见下文"模式 A" |
| 无匹配 profile(一次性/探索性任务) | 模式 B:对话组装沙箱工具,见下文"模式 B" |

Profile 是声明式的(steps + verdict + output),控制流(逐条循环、镜像去重、
断点续跑、沙箱清理)全部冻结在 runner 里;没有匹配 profile 时**不要**现场改
runner,改用模式 B 对话组装。

## 前置条件

- 目标环境可用镜像已就绪:可直接拉取的容器镜像引用,如 `eclipse-temurin:8-jdk`
- 当前会话已配置 Pyromind 认证(`auth_token` + env/cluster;runner 走
  `--auth-token` 或环境变量 `PYROMIND_AUTH_TOKEN`)
- LLM 端点约定(pi 链路):网关需提供 OpenAI chat-completions 协议;
  `LLM_BASE_URL` 带不带尾部 `/v1` 均可(runner 注册 provider 时统一
  规范化为 `/v1` 结尾);`LLM_MODEL` **必须显式传**——不传时 pi 用
  provider 默认模型,请求打到非预期模型(模型名原样透传给网关,如
  `openai/deepseek-v4-flash-0731`)
- LLM 凭据三件套 fallback:会话 secret 中没有 `LLM_BASE_URL` /
  `LLM_AUTH_TOKEN` / `LLM_MODEL` 时,从会话环境
  `DF_API_BASE_URL`(去尾部 `/v1`)/ `DF_API_KEY` / `DF_MODEL_NAME` 对应取值,
  再回退 legacy `ANTHROPIC_*`,经 `--set` 传给 runner(profile 的 run_pi
  env 声明了三个 `{secret:...}` 占位符,缺一项该条记 error);不要把明文密钥
  写进对话或 manifest

---

## 模式 A:冻结运行时(profile 匹配时)

适用:成批记录(每条 = 镜像 + 题面 + verifier)、逐条独立判定、可断点续跑的
验证/处理任务。当前 seed:`profiles/tmax-validation.json`,链路:起镜像沙箱
→ 装 pi → pi headless 解题 → 同箱写入并执行
test_sh → 按 reward 文件/退出码判定。manifest 字段契约:
`task_id` / `image` / `workdir`(tmax 默认 `/home/user`)/ `prompt`(题面)/
`test_sh`(verifier 全文)。

### 全链路三段平台化(agent 只走控制面)

manifest 的构造、逐条验证、最终聚合**全部在平台节点执行**,agent 全程
不出现本地 manifest.jsonl / verdicts.jsonl / 训练文件:

```text
preview_dataset 看字段 → 写/确认 render 模板 JSON(几 KB)
  ① edp_render    → 平台节点:读 parquet → 分片写 batch-XXX/manifest.jsonl + shards.json
  ② edp_submit    → 平台节点:逐条起镜像验证 → verdicts.jsonl + traces/
  ③ edp_aggregate → 平台节点:跨片合并去重 + 转 slime/SFT → 训练文件落 storage
每段:增量 append+flush、progress.json、节点日志逐条一行、同 run/out 目录断点续跑
```

### 1. 渲染分片(manifest 由平台节点构造)

**数据源定位(先做这一步)**:用户给出的数据路径(如 `/workspace/datasets/tmax/`)
一律是 storage 路径,**第一步就直接 `preview_dataset` 探索目录结构**,不要先在
本地 terminal 找(storage 路径本地不可见,白绕一步)。storage 上的 tmax 目录是
HF 数据集镜像:

| storage 路径 | 内容与用途 |
| --- | --- |
| `data/train-*.parquet` | **渲染主源**(任务索引,每行一个 task_id 对应 `tasks/` 同名目录);parquet 是二进制,preview 按文本读会乱码,用 `mode='sample'` 物化后以 pandas 解析字段 |
| `tasks/<task_id>/` | 每条任务的原始素材,仅供交叉核对:`container.def`(Apptainer 构建定义,**不是**可直接传 sandbox_create 的镜像引用,manifest 的 image 用 join 源映射)/`setup.sh`/`test_initial_state.py`/`test_final_state.py`(与 test_sh 交叉核对)/`task_summary.txt`(题面摘要)/`solutions/`(**参考解,严禁写入 manifest 或 prompt**——RL rollout 需要 CC 真实解题,泄漏参考解会污染训练数据) |
| `tasks.zip` | tasks/ 的原始压缩包,勿重复解压 |

**调研纪律(硬约束)**:写模板前的调研只允许两类动作——`preview_dataset`
(含 `mode='sample'` 物化)与读本 skill 文档/源码。预览单文件超限时
**不要**自建沙箱下载全量数据查 schema——那会把全量数据拉回 agent 侧,
违背三段平台化的控制面原则;正确动作是 AskUserQuestion 向用户说明缺口
与候选方案,由用户决策。字段来源表达不了时先查下方 spec 能力表
(`message`/`storage_file` kind 与点号嵌套下钻覆盖 chat 格式、逐任务目录
与 struct 嵌套形态),仍表达不了再问用户,不要绕开平台自行构造 manifest,
也不要另起 pipeline 先把 parquet 展平再渲染。

**出渲染模板 JSON**(字段映射 + 分片大小,几 KB;**shard_size 需与用户确认**):

```json
{
  "fields": {
    "task_id": "task_id",
    "image": {"join": {"source": "datasets/tmax/processed/manifest.jsonl",
                         "on": "task_id", "column": "image"},
              "on_missing": "fail"},
    "workdir": {"fixed": "/home/user"},
    "prompt": "description",
    "test_sh": {"kind": "pytest_wrapper", "source_field": "test_final_state",
                 "target_path": "/workspace/test_final_state.py"}
  },
  "shard_size": 500
}
```

chat 格式数据集(open-instruct 形态:prompt 在 `messages` 列、test 资产在
逐任务目录)模板示例:

```json
{
  "fields": {
    "task_id": "task_id",
    "image": {"join": {"source": "datasets/tmax/processed/manifest.jsonl",
                        "on": "task_id", "column": "image"}},
    "workdir": {"fixed": "/home/user"},
    "prompt": {"kind": "message", "source_field": "messages", "role": "user"},
    "test_sh": {"kind": "pytest_wrapper",
                 "source": {"kind": "storage_file",
                             "path_template": "datasets/allenai/tmax/task-data/{task_id}/tests/test_final_state.py"},
                 "target_path": "/workspace/test_final_state.py"}
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
fail fast。

字段映射(**以实际 parquet schema 为准,HF 发布版字段名可能不同**;若字段对
不上先 `preview_dataset mode='sample'` 确认再调整模板):

| manifest 字段 | 来源(实测 tmax parquet 13 列) | 说明 |
| --- | --- | --- |
| `task_id` | `task_id` | 原样 |
| `image` | **不在 parquet**(无 swerl 镜像引用,`container_def` 只是 Apptainer 定义) | 模板用 `join` 从 storage 上的映射表(JSONL/CSV,含 task_id+image 列)查得;缺映射任务记入 `render_failures.jsonl` 并逐条报告,**不猜 tag** |
| `workdir` | 固定 `/home/user` | 数据集约定 |
| `prompt` | `description` | 原样(题面),不做裁剪 |
| `test_sh` | `test_final_state` 列 | 渲染时自动包装:heredoc 写入 `/workspace/test_final_state.py` → `pytest` → 按 rc 写 `/logs/verifier/reward.txt`(1.0/0.0) |

提交渲染:

```text
edp_render(template_path=<file_editor/apply_patch 写模板时用的同一相对路径>,
            data_source=<storage parquet/glob/目录>,
            shard_size=500, [limit=N])
```

- 模板用 `file_editor`/`apply_patch` 写入会话工作区(如
  `public_data/render_template.json`),`template_path` 直接传同一相对路径
  (工具自动按会话工作区解析,也接受绝对路径);**不要**先上传 storage 再传
  storage 路径,那多绕一步

- 节点分段读 parquet(pyarrow iter_batches,内存≈单片),分片写 storage
  (`<output_root>/batch-XXX/manifest.jsonl`),输出 `shards.json` 索引 +
  `render_failures.jsonl`(join 缺失任务)+ `progress.json`/`report.json`,
  节点日志逐条一行(`[i/N] task_id=... -> batch-XXX`)
- `data_source` 支持 glob(`train-*.parquet`)与目录(取其下 `*.parquet`)
- 渲染内存 ≈ 单片,10W+ 级数据源同样适用;agent 不下载 parquet

**门禁(硬约束)**:渲染完成后**立即用第一片提交平台 smoke**——禁止本地
核验/手工逐条编排/反复 preview。已有产物复用规则:同一批数据未完成 →
直接消费旧产物或同 run 续跑;数据变了 → 重新渲染后新 run 提交。

### 2. 提交验证(smoke → 用户决策 → 分批/全量)

验证以 **CustomCommandCPUNode 平台任务**提交执行(`edp_submit` 工具):
工具冻结 `sandbox_runner.py` + profile + `pod_runtime/`(节点 shim)进每片
run 的 storage 目录,提交含凭据注入的节点工作流;每片一个独立 workflow,
返回 `task_ids` / `run_ids` / `output_dirs`。

**2.1 smoke(第一片,limit=3)**

```text
edp_submit(manifest=<render 输出的 batch-001/manifest.jsonl>, limit=3)
```

- Kafka 终态回调自动唤醒会话(Succeeded/Failed),随后用 `preview_dataset`
  查看 `<output_dir>/run/verdicts.jsonl`(task_id / verdict / exit_code /
  error_category / reward / note)与 `<output_dir>/run/traces/`(pi 轨迹)
- 运行中可 `df_check_progress(output_dir=<run 目录>, tail_filename="verdicts.jsonl")`
  看实时进度(total/processed/ETA + 最近若干条 verdict);节点日志逐条一行
  (`[i/N] task_id=... verdict=... reward=...`)
- 按 verdict 分布汇报:usable 数、error 分类计数(
  create_failed / probe_failed / pi_install_failed / pi_run_failed /
  verifier_failed / verifier_env_missing),**verifier_env_missing 单独
  分桶**(镜像缺件可修,非数据不可用)
- smoke 本身就是一个批(单片 3 条):终态后同样自动聚合(见第 3 节),
  sft/slime 即时产出——用户要看"构建出的数据"零等待,转换层信号
  (如 reward<1.0 的 usable 记录不进 SFT)在烧全量前就暴露

**2.2 用户决策门禁(硬约束,AskUserQuestion)**

smoke 终态回调并汇报 verdict 分布 + 单条均时后,**必须 AskUserQuestion
让用户三选一,禁止 agent 默认走全量**:

- ① **直接全量**(单批提交剩余全部片)
- ② **分批**(本次提交 N 片 = 批容量;agent 给推荐值与推算依据:单条均时 ×
  片内条数 vs 12h 任务配额、同镜像拉取风暴约束——3 条样本不符合预期的
  全量成本很高,批容量由用户拍板)
- ③ **先停**(修模板/镜像/join 源后再来)

**2.3 多片提交(分批与全量统一形态)**

```text
edp_submit(shards=<shards.json 路径>, shard_offset=<起始片>, shard_count=<本批片数>)
```

- 一次提交 N 片:每片独立 workflow/output_dir/终态回调,注册 N 个
  ActiveLongTask;`shard_count` 省略 = 从 `shard_offset` 提交到末尾(全量)
- 片 run 目录嵌套在片 manifest 目录下(`<root>/batch-XXX/<run_id>/run/`),
  checkpoint 与 manifest 同片共存;某片中断,同 `run_id` 重提交该片即续跑
  (verdicts.jsonl 为 checkpoint,自动跳过已判条目)
- N 个回调会逐个唤醒会话;agent 每次被唤醒先判断"**本批是否全部片终态**"
  (对照本次提交的 task_ids),未到齐就只汇报进度继续等,到齐才进 2.4

**2.4 批间确认(硬约束)**

本批全部片终态后:汇报本批统计(usable/error 分布、与前批对比、单条均时)
→ **再次 AskUserQuestion:继续下一批 / 调整批容量 / 停止**。不自动推进
下一批。与既有 `Terminated` 回调约定一致:终止/失败不得自动重提交,
交用户决策;需介入时先 `df_stop_task(task_id)` 停平台任务,再改再提交。

**平台约束(已实测)**:节点固定 Python 3.10(conda),openhands 系包(≥3.12)
装不上,`pod_runtime/` 提供 `processing_profile.py` 原样拷贝 +
`create_sandbox_api_client` 等价 shim,runner 零改动;**skill 脚本必须
兼容 Python 3.10**(如 `datetime.UTC` 是 3.11+,用 `datetime.timezone.utc`;
对话 ebda2d49 因 render_manifest.py 的 `from datetime import UTC` 在节点
import 阶段崩溃);storage 上传需带 `x-cluster` 头;节点命令的首段 `export`
可能被丢弃,命令以 `true;` 开头规避,凭据同时经 `--auth-token` / `--set`
显式引用(防丢 key);
**OpenAI 协议直连(实测)**:install_pi 把网关注册进沙箱内
`$HOME/.pi/agent/models.json`(provider `mygw`,openai-completions 协议,
baseUrl 规范化为 `/v1` 结尾),run_pi 以 `--mode json --provider mygw
--model <LLM_MODEL>` headless 解题;凭据只进 models.json,不进启动
脚本(pre/us-west-1 实测 deepseek 自定义网关,任务 8243 一次跑通
usable + reward 1.0);pi 要求 node ≥ 22.19:install_pi 无条件安装官方
Node 22.19(`.tar.gz` 发行包,不依赖镜像内 xz)到 /opt 并链接
/usr/local/bin(tmax 镜像自带 node12 也不受影响);LLM 凭据三件套
从会话 secret 取,缺则 `DF_API_BASE_URL`(去尾部 `/v1`) / `DF_API_KEY` /
`DF_MODEL_NAME` fallback,再回退 legacy `ANTHROPIC_*`;值经节点命令注入,
不落 verdicts

### 3. 聚合训练文件(edp_aggregate,平台节点)

**每批片全部终态后 agent 自动聚合**(非用户触发;smoke 单片即首批):
对本批全部 run_dirs 跑一次增量 `edp_aggregate` 到同一 out_dir,训练文件随批增长;
聚合幂等(同 out_dir 断点续跑 + task_id 去重),重跑无副作用。
用户只在**换参数重派生**时才手动指定新 out_dir 重跑——验证(逐条起沙箱)
贵、转换便宜,`min_reward`/`system_prompt` 等训练侧参数应看完 verdict
分布后再定,换参数重派生不需要重跑昂贵的沙箱验证:

```text
edp_aggregate(run_dirs=[<各片 output_dir>...], out_dir=<聚合输出目录>,
              [min_reward=1.0] [system_prompt=...] [limit=N])
```

- 节点逐条合并各片 verdicts(去重 task_id)并同趟转换:
  `<out_dir>/verdicts.jsonl`(合并 checkpoint)、`slime.jsonl`(usable →
  slime RL)、`sft.jsonl`(reward ≥ min_reward 的轨迹 → SFT)、
  `progress.json`(可与 `df_check_progress` 复用)、`report.json`
  (verdict 分布、error_category 计数、verifier_env_missing 单独分桶、
  转换/跳过计数、重复 task_id 数)
- **断点续跑**:`<out_dir>/verdicts.jsonl` 已有 task_id 即 checkpoint,
  中断后同 out_dir 重提交(可追加新 run_dirs)自动续,不重复不丢
- **slime RL 三键**:只转 `verdict=usable` 记录;reward 不嵌入,rollout 时
  由 `metadata.test_sh` 实时判定(usable 但 reward 0 的记录保留——RL 需要
  0 分信号);字段映射见 references/slime-conversion.md
- **SFT messages 格式**:默认只转 `reward >= 1.0` 的解题成功轨迹
  (`min_reward` 可调),输入为 pi 轨迹 `traces/<task_id>.pi_trace.jsonl`
  (`--mode json` 事件流:message_end 的 user/assistant/toolResult 事件,
  thinking 块丢弃,text/tool_calls/tool 结果映射为 OpenAI 风格 messages);
  pi 轨迹**首条 user 消息即启动题面**,不重复前置;legacy `.cc_trace.jsonl`
  (CC 时代轨迹)仍兼容,该格式不含题面,由片 manifest 的 `prompt` 作为
  首条 `user` 消息前置
- **到此为止,无需 data-cleaning**:`<out_dir>/slime.jsonl` 与 `sft.jsonl`
  即交付物,直接对接训练;仅当需要去重/PII/语言过滤等清洗算子时才衔接
  data-cleaning(tmax 通常用不上)
- 聚合节点纯标准库、无凭据注入;判定规则与批次策略见
  references/batch-orchestration.md

### 4. 本地复现(仅工程师调试,agent 路径一律平台工具)

工程师需要本地复现单条行为时可直接跑冻结运行时(agent 不要走此通道):

```bash
python scripts/sandbox_runner.py \
    --profile profiles/tmax-validation.json \
    --manifest /path/to/manifest.jsonl \
    --output-dir /path/to/run-dir \
    --env pre --cluster us-west-1 \
    --set LLM_BASE_URL=<OpenAI 兼容端点> \
    --set LLM_AUTH_TOKEN=<token> \
    --set LLM_MODEL=<模型名> \
    [--auth-token <token>] [--limit 3]
```

- `--set KEY=VALUE` 注入 LLM 凭据(profile 中以 `{secret:KEY}` 占位),
  值不会写进 verdicts;缺失时该条记 error(exec_failed)
- `run_pi` 以 nohup 后台 + 退出码文件轮询执行(exec 单命令 600s 上限
  扛不住 pi 长任务),`timeout` 参数为 pi 总预算(默认 1800s)
- **清理硬约束**:runner 对每条记录在 finally 中强制删除沙箱(Running 态
  会先 pause 再删),profile 写错只会让该条记为 error,不会泄漏沙箱

---

## 模式 B:对话组装沙箱工具(无匹配 profile 时)

严格按序执行;工具均为独立工具,无 `operation` 字段。

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

### 6. 回传产物

```
sandbox_upload(sandbox_id=<id>, sandbox_path="/workspace/result.json", storage_path="datasets/java8-filtered")
```

- `storage_path` 为 storage 目标目录(可省略,默认
  `/.pyromind-agent/<conversation_id>/uploads`)
- 反向:把 storage 文件取进沙箱用
  `sandbox_download(sandbox_id=<id>, storage_path=..., sandbox_path=...)`

### 7. 清理沙箱(硬约束)

```
sandbox_delete(sandbox_id=<id>)
```

> **无论任务成功还是失败,最后一步必须 `sandbox_delete`**,避免资源泄漏。
> 删除失败时重试一次;确实无法删除时必须明确告知用户。

## 注意事项

- 每一步都检查上一步返回的 `status`/`returncode`,失败即修正,不要盲进
- 辅助文件工具:`sandbox_read_file(sandbox_id, path)` 读文件(二进制返回
  base64);`sandbox_delete_file(sandbox_id, path, recursive=?)` 删除沙箱内
  文件/目录(删非空目录需 `recursive=True`)
- custom 镜像必须自带 bash(probe/脚本依赖它);镜像缺 bash 会在探测阶段暴露
- 沙箱内文件路径以沙箱视角为准;storage 路径与 upload 的目标目录分离
- 若需 VNC 图形交互,用创建返回的 `web_vnc_url`(供用户人工介入)
