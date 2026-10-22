# 历史经验 v2 接入与验证

包依赖由 AgentGenome Python/npm 提供（当前 0.6.29），通过 vendor 安装，继续由 Pi 标准加载器加载默认入口。工具定义、节点回执、修订规则、冻结和修订记录属于 AgentGenome；Runtime 使用通用 `execute_stage` 能力关联当前会话。上层没有 Pi 类型，公开 API 与原有 SSE 事件类型不变。SQLite 仅增加表，原资产及运行记录保留。

0.6.29 修掉沉淀计划预检的几类确定性误判：

- 最终产物被写成描述文本时，机械候选不再退化成 0 步；计划会回退推断产物，并把
  可用产物路径一次列全；
- `limit` 等显式标量类型不再按路径解析，上游步骤写出的中间产物也不再被判成缺失输入；
- `datasets/...` 和 `storage/datasets/...` 等别名按同一路径处理，宿主本地不存在时
  继续查询沙箱 metadata；
- `genome_*` 管理调用不进入候选业务图。

0.6.28 把首次沉淀改成确定性的机械候选：

- 第一次 `genome_distill` 只需要 `label` 和 `source_traces`，不再让模型先手写完整
  `steps`；插件从 trace 提取 run/tool/human 节点、根输入、最终产物和依赖闭包；
- 未被最终产物消费的探测、失败尝试和本地试跑进入 excluded branches。写后立即删除的
  临时文件和只读 `get/list/read/search` 工具不进入链路；
- `import/upload/submit/label_studio_project` 等有外部副作用的工具会保留；没有结果
  路径时生成稳定回执 artifact，并在 candidate ambiguities 里明确提示；
- 模型只展示候选计划和少量 ambiguity，用户确认后才进入原有的一次 preflight 与有界
  验证回放。

0.6.27 修掉两处确定性错误：

- `input_contract` 的类型不再缺省猜成 `path`。`limit` 这类标量必须有显式 `type`，或带可推导的 `default`；旧的文件契约仍通过 `format` / `fields` 等提示兼容。无类型标量在计划阶段直接阻塞，不再被路径规范化后变成错误值；
- `df_submit_pipeline` 的参数不再统一走通用路径改写。`input_path` 按 Storage 映射，`script_path` 和 `support_file_path` 按包内 workspace 路径映射，`model_profile` 等普通字符串保持原值，避免重复改写和错误 base。

0.6.26 把沉淀拆成「计划」和「确认后 preflight」两个阶段：

- `genome_distill` 首次调用只读取一次 trace，返回结构化计划：步骤类型、来源、输入输出、旁支、人工门禁、最终产物和预期入图步骤都在计划里。此时不读取脚本内容、不物化脚本、不创建 package、不编译 graph，也不执行或回放；
- 用户确认的是这份可读计划，不是一次尚未展示的完整构建。确认后再次调用时才做一次确定性 preflight；如果结果改变了用户看过的步骤范围、节点类型或最终产物，插件拒绝继续，不能靠“补齐步骤”在后台静默改写计划；
- preflight 失败会一次性聚合返回，不创建 capture；计划必须携带 `validation_params` 和 `execution_scope.mode=sample|subset`。用户确认计划即同时确认这次有界回放，preflight 通过后自动排队，回放成功才发布；缺少最小回放范围时不能确认，`full` / `unspecified` 也不能代替验证范围；
- 这个边界让长会话的沉淀收敛为固定流程：一次读 trace、一次出计划、用户确认、一次校验、创建 capture，避免把大量脚本分析和试错塞进模型思考阶段。

0.6.22 把链路范围的提示词改成一条数据流判据，消掉沉淀前的反复权衡（真实会话 `675361cc9ae84e3ab0c4c829d5127834`：`沉淀一下` 之后模型独白 92k 字符、4 分钟没有提交）：

- 复现：会话里先本地跑小样、再提交平台执行；旧提示词既写「与链路不相连的旁支不挡沉淀」又写「不要自己判定成旁支就缩范围」，还要求「拿不准先把步骤表拿给用户确认」，与另一条「列完立刻提交、范围问题由插件校验带回来」互相矛盾，模型在两条规则间反复推导，始终不敢提交；
- 提示词：链路范围只留一条判据——这一步的产物有没有被链路里后面的步骤读取、或它有没有读取链路里某一步的产物；有则必须进 steps，没有就是旁支，不进 steps 也不用问用户。本地试跑小样（哪怕与正式执行共用同一个脚本或同一份输入）明确算旁支，脚本文件不参与该判据；
- 提示词：删掉无条件的「拿不准先问用户再提交」——判据能定的直接提交；判据定不了时（用户意图有歧义，或不同范围会让长链路、跨人工门禁的链路明显不同），提交前用一句具体问题问用户，带步骤表和两个可选范围，不靠提交试错、不反复追问；「沉淀一下」的流程补上「步骤表只列一遍、列完就提交」；
- 机制未变：插件侧 `_scope_split` 早已按同一判据实现（脚本不算数据相连，本地试跑落旁支段、不触发范围复核），本次只改模型看到的措辞；
- 版本更新为 0.6.22。回归覆盖 guidance 断言：旁支判据、共用脚本的试跑、判据定不了时的单次提问、去掉无条件「先拿给用户确认」的旧写法。

0.6.20 修掉「清单离开图片目录」导致的平台提交失败（真实会话 `675361cc9ae84e3ab0c4c829d5127834`，失败 run `01cb945df4614e19baafa34e7bf480d2`）：

- 复现：同一条轨迹两次蒸馏选中不同步骤。v0.1.0（capture `15746052…`）里取子集的那一步落在图外，清单留在 `datasets/test_100/test_100/` 与图片同目录，平台任务 9146 成功；v0.2.2（capture `44deca2e…`）里它成了图节点，`emit` 把产物改写为 `{artifact}`，引擎按 `artifacts/<节点名><后缀>` 落位，清单离开图片目录，平台任务 9148 报 `ValueError: record 0: missing image 1-0805Y/1/B0.bmp`——清单里的图片是相对路径，平台按清单所在目录解析，产物一旦搬走就必然找不到；
- Genome（方案 2）：`artifact` 新增 `at` 槽位（形如 `{input.<键>.dir}/<文件名>`），run 叶子可以把产物钉在某个输入的目录里。编译期校验形状、槽位必须是本节点已声明的输入、只允许 run 叶子；引擎按 `at` 落位并把 run 目录外的绝对路径记进 ledger；`emit` 在产物与输入同目录时自动派生 `at`，产物名取绑定后的真实文件名（计划用 `{token}` 写法时也如此）；
- 宿主预检（方案 1）：`df_submit_pipeline` 提交前，bridge 在沙箱里按同样的有界头部（50 条记录 / 200 张图）检查清单里的相对图片是否存在，缺图当场失败并给出改法，不再等平台跑几分钟才报错；
- 顺带修掉两处路径拼写：`platform_input_path` 把挂载绝对路径改写成平台认的 Storage 相对路径（`/datasets/...`，不带 `storage/` 前缀——任务 9149 证明带前缀必失败）；bridge 的读路径统一先走 `storage/...` 别名，钉在 Storage 的产物在采样、预检和参数解析里都能解析；
- 版本更新为 0.6.20。回归覆盖 at 槽位编译与引擎落位、同目录/异目录产物、`{token}` 产物命名、提交前预检拦截与放行、挂载路径改写。

0.6.19 打通人工门禁的唤醒接续（真实会话 `675361cc9ae84e3ab0c4c829d5127834`，失败 run `8617f3cacd2445f48f749aa4baf58965`）：

- 复现：回放 A 跑到 `pcb-labelstudio-sft.ls-export` 人工门禁停下（`waiting`），用户完成复核导出后唤醒，run 直接 `interrupted`，错误是 `FileExistsError: '.../public_data/workflow-runs/<run-id>/artifacts'`——宿主每次唤醒都在同一个 run 目录上重新执行 `mkdir()`；
- 宿主修复（P0-1）：建 artifact 目录改成 `parents=True, exist_ok=True`，唤醒不再被「目录已存在」打断；
- 人工产物交接（P0-2）：`genome_resume` / `genome_replay_requeue` 支持 `artifact` 参数。用户在自己的位置导出（例如 `storage/.pyromind-agent/label-studio/<project>/export/label_studio_export.json`）时把该路径交给宿主，宿主重新进入那个人工门禁时把它落到图声明的 artifact 上再放行，用户不必知道运行目录；
- 中断续跑（P0-3）：`resume` 现在也接受 `interrupted` 且检查点里停着人工节点的 run；`execute` 优先用 run 上留存的 waiting 回执，回执丢失时回退到检查点里层级最深的 `waiting` 节点（park 会把该节点和所有祖先都标 waiting，最浅的是图根），唤醒的仍是用户已经回答过的那个门禁，不会再问一遍；
- 重排通道（P0-4）：新增 `genome_replay_requeue`——与图里脚本无关的宿主/基础设施失败（沙箱或 run 目录被回收、连接中断）重新排队同一次回放：检查点里停着人工节点的从检查点继续，其余用同一份冻结草稿从头跑；`captures.finish` 不再把 `interrupted` 当成终态 `draft`，重排后的成功仍能发布；
- 人工节点的用户入口（P0-5）：等待通知与插件 guidance 都要求把上游 tool 产物里的 `open_url` / 项目链接给用户当操作入口，图内部的 artifact 路径只是回执位置；失败通知按三类分流（宿主/基础设施→重排、脚本缺陷→修好重新沉淀、waiting/queued/running 不算失败）；
- 版本更新为 0.6.19。回归覆盖唤醒二次进入、人工产物落位、interrupted + 检查点续跑、重排（新 run / 检查点 / 已发布拒绝）。

0.6.16 把轨迹导出改成「完整导出、截断拒收」（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.15）：

- 背景：compact 导出（240 KiB 上限）为了一次宿主读取读得下会裁剪步骤，被裁的步数记在 `dropped_steps`；真实长会话的 compact 导出已经顶到上限。一旦被裁剪，蒸馏就会从残缺证据里静默丢步骤——这是「链路阉割」反复出现的机械来源之一；
- 导出：capture / distill 改用 `complete: true`：不裁剪字段、不丢步骤；完整导出超过 4 MiB 时在写文件前直接失败，报错给出「把链路拆到更短会话」的出路，不悄悄降级回 compact；
- 拒收：`capture_trace` 拒绝 `dropped_steps > 0` 和命令/工具参数里的 `clipped_fields`；`pipeline.py` / `distill.py` 在切片之前就报「轨迹导出不完整（dropped_steps）」，`review_confirmed` 也不能越过——导出里不存在的步骤，人工确认无从核对；
- 附带：`agentgenome/service.py` 修掉 resume 的检查点判定（`resume()` 把 waiting 持久化成 queued 时保留 waiting 回执，带回执或已有检查点的 queued run 必须续跑而不是重跑）；插件 guidance 同步。

0.6.17 补齐导出通道与文案（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.15）：

- 宿主适配：`execution.read` 的单文件上限从 256 KiB 提到 512 KiB（仍在 1 MiB 帧上限内），`read_many` 按 base64 字节预算（768 KiB）分批、超预算路径分页返回，一次响应不再可能顶穿帧上限；bridge 侧统一 `_READ_LIMIT`；
- 验证脚本：`scripts/validate_agentgenome.py` 补上 `local_root`，与产品接线一致，宿主侧导出（trace 写在会话目录）直读、不再绕 RPC；
- 文案：截断/裁剪类错误统一中文（`pipeline.py` / `distill.py` / `capture/trace.py` 与 `session-trace.js` 的 4 MiB 上限提示）；
- 防御：`emit._verify_outputs` 的 `assert` 改成显式 `EmitError`；`versions.py` 的失败消息（stage + version id）补上测试断言。

0.6.18 补齐「Genome artifact → Label Studio JSONL」交接并验证最终 SFT 产物（真实会话 `675361cc9ae84e3ab0c4c829d5127834`）：

- 新失败证据：回放 A 已成功跑完 `df_submit_pipeline`、异步等待与 `add-verdict`，但 `label_studio_project` 将工作区 artifact 当 Storage key 查找，报 `Pyromind storage object not found: /public_data/workflow-runs/...`；这不是等待脚本或数据内容的问题，而是 tool 的 `dataset_path` 执行域不匹配。
- 宿主修复：Label Studio 的 `jsonl` adapter 识别当前 Genome `public_data/workflow-runs/<run-id>/artifacts/*.out` 输入，先从会话 workspace 以 64 MiB 上限读取，按 SHA-256 暂存到当前 conversation 的 `/.pyromind-agent/<conversation>/genome-inputs/<run-id>/...jsonl` Storage key，再用真实 Storage 路径完成项目创建。普通用户 Storage 输入保持不变；缺文件、越界或超限时失败关闭。
- Genome 图验证：distill 有 `output_contract` 时必须显式声明 `final_artifact`，且它必须对应编译图的真实输出；JSONL/JSON 产物在发布前逐行/整体解析，核验非空、对象行和必需字段，不能再只用 `test -s` 把不完整 SFT 包发布。
- 插件提示同步：说明 `label_studio_project.dataset_path` 可直接引用先前节点的 JSONL artifact，由 SDK 自动暂存；需表达输出格式与字段时必须声明 output contract 和最终产物。
- 版本更新为 0.6.18。回归覆盖 artifact 上传路径、普通 Storage 路径兼容、超限拒绝，以及 malformed / missing-field SFT 不发布。

0.6.15 修掉「预打标 → Label Studio 复核 → 清洗 → SFT」链路沉淀失败暴露的四个问题（真实会话 `675361cc9ae84e3ab0c4c829d5127834`）：

- 问题一（异步工具回执）：`df_submit_pipeline` 是异步提交，工具节点落盘的 artifact 是 356 字节的受理单（`{status, task_id, output_dir}`），真实产物在 `{output_dir}/processed.jsonl`。原链路把受理单直接喂给下游 `make_review.py`，回放报 `JSONDecodeError`。现在异步工具沉淀成两步：tool 步骤（受理单 artifact）+ run 步骤跑 `.agents/skills/data-processing/scripts/preparation/wait_pipeline.py`（随包搬运），脚本等 `report.json` 落定、把产物复制到下游要读的 artifact；`report.json.status=failed` 时节点直接失败并指向 `failure.json` / `validation.json` / `llm_calls.jsonl`。这一步的产物路径是位置参数，所以 run 步骤用 `output` 显式声明产物；
- 问题一配套（Runtime）：图的 shell 步骤此前吃 Pi runtime 的 300s 默认上限，等不完平台任务。SDK bridge 现在显式传超时（`PYROMIND_AGENTGENOME_SHELL_TIMEOUT_SECONDS`，默认 1800s）；
- 问题二（目录输入采样）：根输入是目录（`images_root`）时采样只报 `Only base64 data is allowed`，回放也拿不到 `storage/` 别名。现在 `capture/sample.py` 记 `format: dir` 并跳过契约校验；bridge 的 `head` 先 `[ -d ]` 探测、目录直接报 `IsADirectoryError`，并对 `PATH_SCOPE_ERROR` 自动重试 `storage/` 拼写；
- 问题三（引用闭包与 keep）：反向切片只保留最终产物的依赖闭包，「副作用就是目的」的步骤（建 Label Studio 项目、提交平台任务）会被当辅助丢掉，而本地试跑小样（`head -10`、`sandbox_sample.py`）又容易被塞进 steps。现在 distill 的步骤可以声明 `keep: true` 强制保留；没被任何后续步骤消费、又没声明 keep 的步骤会被明确报错（加 keep / 从 steps 删掉改成根输入 / 改成 human 步骤三选一），插件 guidance 同步写明「小样试跑不进 steps」；
- 问题四（一次报全 + 预提交确认）：`plan_steps` 过去遇到第一个错误就抛，真实会话里连着提交了 5 轮才收敛。现在一次列出全部问题（编号清单），guidance 也要求对范围或归类拿不准时先把步骤表给用户确认。
- 附带（0.6.14 已改）：run 步骤可以显式声明 `output`（脚本把产物写成位置参数时），平台回放未通过的同一版本可以重新排队（`_replay_restartable`）。

0.6.13 让「本地试跑小样 → 提交平台执行」这类会话只沉淀提交那一步（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.14）：

- 背景：用户先跑小样确认脚本、再提交平台执行时，沉淀被范围复核挡住——插件把 4 次小样试跑判成与链路相连，而用户要的只是「提交平台 → 人工复核 → 格式转换」这三步；
- 原因一（机械）：`capture/trace.py` 的 `_tool_step` 对未知宿主工具只记 `<tool:名字>`，reads/writes 全空，宿主工具调用在数据流上不可见。现在 args 里的路径进该步 reads（`arg_paths`），`df_submit_pipeline`、`label_studio_project` 这类 tool 步骤能参与链路连通性判断；
- 原因二（机械）：小样试跑读的 `--pipeline <脚本>` 与计划里 tool 步骤的 `script_path` 指向同一个脚本，路径交集把试跑判成与链路相连。脚本是资产不是数据：`_scope_split` 现在只看数据路径（`_data_paths` 过滤 `ROLE_SCRIPT`），试跑落进旁支段；
- 原因三（机械）：`_handover_steps` 的第二条判据（被丢掉的步骤落在计划第一步之前且其后仍有用户轮）会把普通指令（「本地先验证 3 条小样」「提交平台执行 10 条」）误判成人工交接，删掉后只认用户点名交回产物的那一轮；
- 配套（机械）：脚本由被丢掉的固定命令产出（`cp` 模板到工作区这类）不再算「链路不完整」，tool 步骤 args 里的 `{脚本参数}` 也会物化——脚本随包搬运、args 改写成包内路径、参数从 params 去掉；
- 验证：真实会话 `675361cc9ae84e3ab0c4c829d5127834` 的轨迹（77 步）现在能沉淀成「`tool: df_submit_pipeline` → `human` 复核导出 → `run: ls_to_sft.py`」三步，5 个试跑步骤进旁支、`pcb_prelabel_pipeline.py` 随包搬运、不触发范围复核；AgentGenome 新增用例 `test_distill_drops_a_local_trial_that_shares_the_submitted_script`，全量 293 个测试通过。

0.6.12 补上「人工交接」这条沉淀范围证据（prompt 与机械校验两侧，设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.13）：

- 背景：0.6.11 的「先枚举、再提交」生效了——真实会话里模型确实先列了完整步骤表，随后仍把链路截成最后一步。这次它走的是一条合法后门：把用户中途人工导出的产物（「我已经人工复核完打完标成功导出了 …/label_studio_export.json」）声明成链路根输入。原规则只允许「编辑工具、人工或纯 agent 产物」写成 `{参数名}` 输入边界，人工产物被并进同一类，于是「复核导出 → SFT」从字面上看是干净的，插件放行；
- 机械侧原本看不见这件事：`_uncovered_steps` 以文件为证，而人工产物不是任何会话步骤写出来的，被丢掉的前半段与计划的根输入没有任何路径交集，`_scope_split` 判成旁支（旁支不挡沉淀），复核静默通过；
- 修复一（prompt）：输入边界收窄为「链路开始前就存在的输入 / 编辑工具写出的脚本 / 纯 agent 产物」，并写明「会话中途用户自己做完并交回的产物不是输入边界，它是 `human` 节点的产出，必须把人工门禁写成 human 步骤、上游固定步骤一并补进 steps」，`genome_distill` 描述、`source_traces` 字段描述与 guidance 同步；另加一条「沉淀的范围不能靠找一个干净的根输入来缩」的规则，并把「链路长、依赖平台配置（DF_*、模型 profile、Label Studio 项目）」列为不能缩范围的理由；
- 修复二（机械）：`capture/trace.py` 的 user 步骤开始保留该轮文本（此前是空串），`capture/distill.py` 新增 `_names_artifact` / `_handover_steps`——被丢掉的步骤若有后续用户轮点名了计划的根输入，或落在计划第一步之前且其后还有用户轮，就按「人工交接的前半段」报「链路范围需要用户复核」并逐条列出，只有用户确认后才允许带 `review_confirmed=true` + `review_reason` 通过；
- 验证：用真实会话 `675361cc9ae84e3ab0c4c829d5127834` 的轨迹（110 步）重建后，同一份「只留最后一步」的计划现在会被拦下并点出预打标那 5 个被丢掉的步骤（模板 `cp`、三次 `sandbox_sample.py` 小样）；AgentGenome `tests/capture/test_distill.py` 新增两个用例（人工交接前半段必须复核、人工门禁进图后链路完整通过），全量 292 个测试通过。

0.6.11 补上「先枚举、再提交」的沉淀决策程序（插件 prompt 侧）：

- 新增一条沉淀流程规则：用户说「沉淀一下」后，先按执行顺序列一张步骤表（一步一行：会话里的原始步骤 → 归类 `run` / `tool` / `llm` / `human` → 输入与产物路径），把真实跑过的固定命令、固定工具调用、人工门禁、agent 裁量都列进去，列完立刻按这张表提交；不要用散文反复权衡链路范围，不要因为某一步「上游已经做过」就把它排除在表外，也不要在提交前自己缩范围——范围问题由插件校验后带着具体步骤回来找用户确认。0.6.9 / 0.6.10 都没有这条决策程序，长链路下模型会在正文里反复推导范围而不调用工具（真实会话里出现过约 2 分钟、约 1 万字推理、只调了一次 `genome_trace` 就断掉的情况）；
- 内联代码那条补上「在会话里用落盘后的脚本重跑一次」：重跑才会在 trace 里留下可固化的命令，否则那段代码仍然只是编辑工具产物；同时写明内联代码不是缩小链路范围的理由，缺哪段脚本就补哪段；
- 恢复 0.6.9 的旁支规则（本地试跑小样、只写调试文件的探测、被后续步骤取代的尝试不挡沉淀、不必为它们问用户），并明确缩范围只在插件报「链路范围需要用户复核」且用户确认后才允许；
- 修掉 guidance 拼接：规则此前用 `join("")` 连成一整段，改成 `join("\n")`。

0.6.10 移除 `gaps` / `references`，历史经验只收完整固化的链路（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.7）：

- `manifest.py` 不再有 `references` / `gaps` 字段、`reference_path` / `reference_id` / `capture_references` / `capture_gaps` 等接口，`validate()` 丢掉 `root` 参数；`genome_list` 的紧凑视图不再带 `has_references` / `has_gaps` / `reference_assets`，`genome_get` 不再返回两层指针。旧包 manifest 里残留的同名键会被忽略，导入仍然成功；
- `capture/pipeline.py` 的 `_require_declared_gaps` 换成 `_require_solidified`：链路只要有固不下来的步骤就直接 `EmitError`，把每步的序号与原因列出来，并给出三条出路——换成 `run` / `llm` / `tool` / `human` 节点重新提交（`genome_distill`）、用 `final_artifact` 重新切片、或把本次特有的产物声明成 `{参数名}` 输入边界。不再有「声明成 gap 就能发布」这条后路；
- `determinism` 改由节点种类推导（`human` → `human-gated`，`llm` → `agent-assisted`，其余 → `script`），此前「有 gap 就是 agent-assisted」的推导会把含人工节点的图误标成 `script`；
- 编辑工具写出的产物不再能进 gaps：`_check_llm_promotions` 提示改成「改成脚本产出（`run`）或声明成输入边界」，`genome_distill` 的 `from` 校验同步；
- 插件侧：`genome_read_reference` 工具与 `references` / `gaps` 参数一并删除（工具数 14 → 13），system prompt 改成「固定命令→run，固定宿主工具调用→tool，人工门禁→human，真正的 agent 裁量→llm」，`genome_list` / `genome_get` / `genome_capture` / `genome_distill` 的描述同步；
- 影响：此前靠 gaps + references 半固化发布的经验不再能新建；已经发布的老包仍能 list/get/run，只是其中的 gap 步骤不会再被提示给 agent，需要按 graph 实际覆盖的范围使用。

0.6.9 新增人工门禁节点与沉淀范围复核（设计见 AgentGenome 仓库 `docs/v2-capture.md`）：

- graph schema 新增 `do: {human: <问题模板>}`：人的复核、打标这类环节不再降级成 `gaps`，而是成为引擎直接执行的叶子节点。引擎停在该节点（`NodeWaiting`），把节点路径、问题、必须写入的 artifact 一起写进 run 状态（`status=waiting`），checkpoint 记 `waiting`，不占 worker，也不写假的产物；
- 内核新增 `HumanPort` / `Ports.human`（`core/ports.py`）与 `Answer.pending`：宿主 `ask` 返回 `pending=true` 表示无人答复，引擎立即 park；唤醒时同一个节点重新进入，`prepare(..., resuming=<节点路径>)` 让宿主只批准那一个门禁，其后的门禁照旧 park；
- 服务层新增 `resume(scope, run_id)`，对应插件第 14 个工具 `genome_resume`：`status` 暴露 `phase=waiting_human`，capture 回放停在人工节点时状态记 `waiting`、报告写明「人工完成后调用 genome_resume，本条经验在回放成功前不会发布」，`cancel` 对 parked run 同样有效；
- SDK 宿主适配：`execution.capabilities` 增加 `human: true`，`ExecutionHost.ask` 没有 `resuming` 时一律 park；Runtime 把 `waiting` 当成终态之外的一档通知（有活动会话即时通知，没有活动会话则在下一次会话开始时补发），operation 保持打开直到 resume，通知里给出 node / question / artifact 并要求 agent 让用户完成后调 `genome_resume`；
- `genome_distill` 新增范围复核：计划若漏掉会话里产出过文件的步骤（`_uncovered_steps` 以文件本身为证，不看 agent 写的 `from`），先按 `_scope_split` 分成两类——**和链路相连**的（产物喂给链路，或接在链路后面）以「链路范围需要用户复核」报错并列出，只有用户确认「这次只要后半段」时才带 `review_confirmed=true` + `review_reason` 重提，确认过的边界写进沉淀报告；**不相连的旁支**（本地试跑小样、只写调试文件的探测、被后续步骤取代的尝试）不挡沉淀，连命令一起写进报告的「旁支步骤（未进链路，链路不依赖它）」段与 payload 的 `branches`，agent 不必为它们问用户；
- 人工判断本身无法机械验收：测试覆盖的是暂停/唤醒协议与 artifact 契约，人工节点的判断质量仍需真实会话验证。

0.6.8 新增 `do: {tool}` 节点与宿主工具端口（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.12）：

- graph schema 新增第四种节点形态 `do: {tool: {name, args}}`：平台提交（`df_submit_pipeline`）、Label Studio 导入建项目这类固定工具调用不再只能进 `gaps`，而是成为引擎直接执行的叶子节点，没有模型在环。`args` 里的字符串支持槽位替换，**精确等于槽位**的字符串保留 JSON 类型（路径参数仍是宿主路径），嵌在文本里的按字符串替换；
- 内核新增 `ToolPort` / `Ports.tool`（`core/ports.py`）：引擎把填充后的参数交给宿主，把返回 JSON 写成节点 artifact（`artifacts/<node>.out`，可用 `artifact: {ext: json}` 改后缀），随后照常走 `output` / `verify`；`is_error=true` 按 `do` 阶段失败处理并保留 artifact，`danger: true` 沿用既有 gate；
- 宿主能力位新增 `tool`：`execution.capabilities` 返回 `tool: true`，`GenomeService.check_capabilities` 按 `_requires_tool` 自动补依赖；没有工具端口的宿主在 handshake 阶段报 `E_NEEDS_PORT_ABSENT`；
- SDK 桥接侧：`BridgeTool` 走 `execution.tool` → Pi runtime → 适配器既有的 `tool.execute` 分支，图里的固定调用与 agent 手调是同一个业务工具（同一套会话上下文、权限与信号），不另写实现；`ExecutionHost` 补上 `write_text`，tool 节点的 artifact 由此落盘；
- `genome_distill` 的 `steps` 新增 tool 形态 `{tool, args, output, input?, from?}`：`output` 必填，`input` 必须出现在 `args` 里，`args` 里的路径自动成为该步 reads；scripts-first 政策对称化——会话里真实发生过的固定工具调用，声明成 `gaps` 或包装成 llm 节点都会被拒，只能用 tool 步骤提交。

0.6.7 之后的文件通道修复（SDK 宿主适配侧）：

- 宿主导出的 trace 不再镜像进沙箱：`ExecutionHost.read` / `read_many` 命中 `_local_export` 时直接读本地文件。此前每次 capture/distill 都把整份 trace（≤240 KiB）经桥接写进执行面再读回来，terminal 降级下等于 base64 后 48 行 staging、每 3584 字节一个 ACK 往返，一份 trace 就是几十秒，而且每次沉淀都重传一遍；
- `FallbackSandboxFileClient` 的 terminal 降级改为半开恢复：一次 file API 失败只把会话切到 terminal 一段时间（`retryFileApiMs`，默认 30 秒），cooldown 过后重新探测 file API，答复即恢复。此前一次抖动就把整个会话永久钉在慢通道上；
- terminal 写文件的解码不再依赖 `openssl`：`openssl base64 -d -A` 失败时回退到平台 `base64 -d` / `-D`。此前镜像里没有可用的 openssl 会让整条 staging 链以非零码结束（观察到的 126）。

0.6.7 修复「沉淀链路」反复失败的三处契约缺口（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.6）：

- 同一个文件在计划里允许两种写法：某一步写 `{filtered_jsonl}`、下一步读它绑定的字面路径（或反过来）。反向切片、产出配对（`emit._Plan.file_of`）与链路完整性校验（`distill._check_chain`）统一经 `trace.canonical_path` 归一后比较，不再把写法差异当成「缺上游步骤」或假根输入；`prepare_capture` 新增 `bindings` 参数，distill 复用 capture 时把同一份绑定传过去，否则第二次构建会漏掉切片里的上游节点、回放报「参数不匹配」；
- `final_artifact` 与步骤产出的写法不再要求逐字一致：字面路径与 `{参数名}` 指向同一个文件即可，`_check_final_artifact` 返回计划里实际写下的那个 token，`dispatch` 用这个 token 交给 capture 切片；
- `params` 不再要求与占位符集合完全一致：必须覆盖链路的根输入，链路中间产物的 `{参数名}` 可以一并给出（不是根输入、不进 graph 参数）；
- 运行期同目录辅助文件探测把「读不到」当成答案而不是失败：`emit.read_many` 把任何读取失败折叠成 `None`；SDK 桥接侧 `ExecutionHost.read` 把沙箱文件接口的 `FILE_READ_ERROR`/`No such file or directory` 转成 `FileNotFoundError`，其它 runner 错误照旧抛出。此前脚本注释或字符串里提到的 `serving/api_llm_serving_request.py` 之类不存在的 sibling 会让整个 distill 以 `PiRunnerError` 崩溃；
- 沉淀不再卡死在读取上：冻结一条链路要搬运脚本 + 同目录辅助文件，且 distill 与 capture 各构建一次包，逐文件读取把慢沙箱放大成几十次往返。AgentGenome 侧新增 `BatchSource.read_many` 批量读协议（`capture/emit.py` 的 `read_many` 帮助函数，`LocalSource` 直接实现）：脚本搬运、runtime sibling 探测按波次批量读，`distill._materialize_scripts` 的候选脚本一次批量读；SDK 桥接 `ExecutionHost.read_many` 转发 `execution.read_many` 并缓存已读文件，同一会话第二次构建零往返；Node 侧 `workflow-execution.ts` 实现 `execution.read_many`（单批上限 64 条，剩余路径分页返回）。`ExecutionHost._call` 同时改为有界等待：`PYROMIND_AGENTGENOME_CALL_TIMEOUT_SECONDS`（默认 180 秒）超时即报错，单次读取打滑不再让整个 distill 无限挂起；`execution.run` 不设上限，命令本身仍可长时间执行；
- 插件 prompt（系统提示词与 `genome_distill` 描述）同步：产出必须显式声明（`>`、`--out`、`--output` 之类），位置参数 `python3 脚本.py 输入 产出` 一律按输入识别；`params` 覆盖根输入即可，中间占位符可一并给出。

0.6.6 把「固定命令伪装成 llm 节点」的后门机械堵上，并给图产物补上类型后缀（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.11）：

- 固化策略 `policy` 新增到 `genome_distill` / `genome_capture`，默认 `scripts-first`：llm 节点必须带 `from` 指向会话里真正来源的步骤，`from` 缺失、不在轨迹里、或来源是固定命令/人工门禁/编辑写入都会直接报错并列出可选的会话步骤；把固定命令声明成 `gaps` 同样被拒绝。`adaptive-first` 显式信任 agent 的判断，保留 0.6.5 及以前的行为；
- 蒸馏报告新增「固化策略」一行，payload 带回 `policy` 字段；
- graph 叶子节点可声明 `artifact: {ext: jsonl}`，引擎按声明后缀命名产物（此前统一命名 `.out`，DataFlow 的 `LazyFileStorage` 按扩展名推断类型时直接报 `Unsupported file type`）；`emit` 从会话里该步骤真实产出的路径推断后缀，agent 不声明也不会丢类型；
- 搬运脚本时，脚本按文件名调用的同目录辅助文件（`.py`/`.sh`）传递性一起进包、文件名不变（`capture/emit.py`，上限 32 个）。技能 CLI 不是单文件：`sandbox_sample.py` 依赖同目录的 `source_fingerprint.py`、`validate_prepared_data.py`、`generate_report.py`，并把自身目录放进 `PYTHONPATH` 供 pipeline 导入 `df_logging`；只冻结 CLI 会让包在复用时报缺 helper；
- 配套的 SDK 侧修复（不属于插件本体）：`sandbox_sample.py` 支持在 `public_data/workflow-runs/<run>/package` 这类嵌套 cwd 下执行，白名单锚点改为向上查找最近的 `public_data/`；未知扩展名的输入先嗅探类型（json/jsonl/csv/txt）并在状态目录建别名（symlink→hardlink→copy，原文件不动），DataFlow 按别名扩展名读取。
- 另一处 SDK 侧修复：`sandbox_sample.py` 的源文件指纹在缺少同目录 `source_fingerprint.py`（冻结包场景）时内联计算，不再让整条 pipeline 失败。

0.6.5 把「llm 节点必须保留原始执行方式」变成默认蒸馏行为（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.10）：

- llm 节点的提示词除原样包含声明路径外，还必须点名这一步原本由哪个 skill、以什么命令形态执行（脚本、`--pipeline`、模型 profile 等）。此前提示词只承载任务目标，回放的 agent 会绕开 skill 现场硬做，链路看似跑通但没沿原路径；
- 约束写在插件 prompt 的三处（系统提示词、`genome_distill` 工具描述、`steps` schema），沉淀时不需要用户额外交代；
- 只影响 llm 节点的提示词文本，`run` 节点的命令与宿主 `agent_task` 端口不变。

0.6.3 把「自由裁量的 LLM 步骤」变成图节点，而不是丢进 gaps（设计见 AgentGenome 仓库 `docs/v2-capture.md` §6）：

- `genome_distill` 的 `steps` 新增 llm 形态 `{llm, output, input?, verify?, from?}`。提示词里原样写出的输入/产出路径被替换成 `{input.*}` / `{artifact}`，节点编译成 `do: {llm}` 叶子，`verify` 默认 `test -s {output.<key>}`；
- llm 节点由宿主的 `agent_task` 端口执行：宿主提示 agent 产出声明的 artifact，agent 写文件后回报 `genome_step_result`；`GenomeService._requires_agent` 在回放前自动给 manifest 补 `agent_task` 能力要求，SDK 桥接侧无需新增分支；
- 硬边界由 `_check_llm_promotions` 把住：一条真实跑过的固定 shell 命令（哪怕脚本内部调用 LLM）写成 llm 节点、或把人工门禁/编辑工具产物写成 llm 节点，给了 `source_traces` 时会被直接拒绝——固定命令必须用 `run`，没有固定调用形式的步骤才进 `gaps`。

0.6.0 给「具体 case 资产」补上输入证据与迭代稳定性（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.9）：

- capture 冻结包时对**根输入**做有界采样（`agentgenome/capture/sample.py`）：CSV/TSV 取表头 + 前 20 行、JSONL 取前 20 条、文本取头部、二进制只留元数据，单文件 ≤ 32 KiB、总量 ≤ 512 KiB。采样走宿主读端口，落 `captures/<id>/samples/`，**不进包**（包字节是回放证据，事后改写会让发布失效）。`genome_get` 返回 `input_samples` 有界预览，`genome_list` 返回 `has_samples`；
- 复用已发布资产时先用采样的 `format` / `required_fields` 机械核对本次根输入，不符直接拒绝并说明差异；`genome_run` 的 `allow_mismatch=true` 表示明知不符仍强行复用，差异记进 run 结果而不是中断；
- 回放 A 成功后把采样输入 + 产物 `sha256` 固化成一条 `baseline-<capture_id>` 迭代基线（`agentgenome/capture/baseline.py`）。采样不完整（截断/二进制/读取失败）、参数与 manifest 不符或没有产物摘要时基线标记 `ready=false` 并写明原因；
- 改脚本的修订自动继承该基线：`revisions.prepare` 在无已冻结 suite、也无未发布 pending 时用 `captures.baseline_cases(...)` 填充 `baseline_cases`，因此「改脚本 + 新用例」不再需要手工重造旧场景用例。回归执行新增 `kind=hash`，直接比对 checkpoint 里的产物摘要，天然适用大文件；
- SDK 宿主适配新增 `ExecutionHost.head(path, limit)`：本地导出文件直接取头部，远端文件在执行面用 `head -c ... | base64` 截断后回传，避免把整份数据集拉过桥接。

0.5.1 修掉了沉淀链路上两个「链路看起来沉淀成功、复用却跑不动」的洞（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.8）：

- 脚本参数物化：`{参数名}` 指向 agent 写进工作区的脚本（`.py`/`.sh` 等）时，插件把脚本随包搬运、把命令改写成具体路径，并把它从 `params` 去掉（报告写「脚本参数已物化」）。此前这类脚本只能作为根输入留给复用方自带，包不自洽，复用必然要重写一遍；真实会话 `0e360d94d6704bbaab15437788e271b0` 沉淀出的 `question-filter-answer-sft@0.1.0` 就是这种窄口径资产；
- 复用侧执行边界写进插件 prompt：graph 覆盖的确定性步骤只能交给 `genome_run` 执行，agent 不得用文件工具重写 graph 里的脚本、也不得手工跑等价命令替代，脚本确实不适用时走 `revision_id` 通道；此前复用会话会把 graph 里的转换脚本重写一遍（2358 字符 vs 原 1731 字符）而不是调用 `genome_run`；
- 沉淀带根输入的链路时要求给 `new_params`（换一组真实输入或按 `input_contract` 造的小样本）跑回放 B，缺省时经验只能发布成未泛化的窄口径版本。

0.5.0 的沉淀链路补齐了四件事（设计见 AgentGenome 仓库 `docs/v2-capture.md` §5.6 / §5.7）：

- `genome_distill` 支持 `{参数名}` 占位符 + `params` 绑定，回放 A 用声明的真实值执行；
- 传 `source_traces` 时校验链路完整性：计划读取的产物如果由会话里更早的成功步骤产出而该步未提交，直接报错并指出缺哪一步；占位符输入作为显式边界写进报告；
- `references` 新增具体 case 资产指针 `{asset_id, version?, when?, input_mapping?}`：沉淀时解析到已发布版本并校验参数映射，不复制文件；
- 复用按两层匹配：先命中具体 case 资产直接 `genome_run`，都不命中才退到泛化资产并按 `gaps` / 文件 reference 补做 LLM、人工步骤。`genome_list` 的 `reference_assets` 直接带 `when` / `input_mapping`。

0.4.0 新增 `genome_distill`：会话里的步骤是内联代码、只读探测或脚本内部写出产物时，机械固化不会通过，agent 先把脚本落盘、把输入产出写到命令行上，再提交规范化链路；插件校验契约、编译 graph，并和 `genome_capture` 一样排队回放验证。`distill` 与 `capture` 共用同一道 origin turn 闸门和异步完成通知，SDK 侧不需要专属分支。

本版使用 `os-sandbox` 的受保护目录保存验收代码，文件工具与终端均不能修改。执行前校验摘要，模型请求通过独立文件 API 交接，不占用终端。业务数据仍留在执行环境，产物链接继续通过 SDK Storage 工具获取。

远程 `sandbox` 支持固定脚本及连续修订；验收约束由 AgentGenome 的工具说明和修订返回结果提示，不承诺运行目录只读。提交仅回收允许修改的业务脚本，验收文件继续使用冻结版本。Agent 节点（`do.llm`）由 Pi 会话执行，不依赖只读验收目录，因此 sandbox 下同样可用（`execution.capabilities` 的 `agent_task` 不再跟随 `protectedRoot`）；模型验收仍要求只读验收目录，sandbox 下保持不可用。

修订不设次数限制：Pi 使用最近的运行 ID，基于上次冻结副本继续修订，或自行判断交还 Skill。重复提交仍去重，旧版回归标准保持冻结；未通过验证的草稿用例可纠正后重新验证。失败运行可直接申请修订；成功但结果不符合需求的运行须提供 `reason`。修订副本返回 `verification_protection`（`prompt` 或 `read_only`）。原因持久化到 AgentGenome 的修订记录，旧记录兼容；Pi 能力发现仍在 Adapter 下层，Runtime 和产品协议不增加专属分支。

## 自动验证

在 SDK 根目录执行：

```sh
uv sync --frozen
npm --prefix harness-adapter/pi-runtime install
npm --prefix harness-adapter/pi-runtime run build
uv run pytest tests/pyromind_runtime
npm --prefix harness-adapter/pi-runtime test
```

`test_genome_v2_integration.py` 使用安装的插件包、真实 Pi 会话与 OS 沙箱，通过本地模型替身验证两批 CSV、阶段回执、独立模型验收、参数修订、脚本修订和失败接手。它不访问业务平台或真实模型服务，不证明模型在真实业务数据上的判断质量。

测试需要允许监听本地 HTTP 端口，并允许 OS 沙箱启动。AgentGenome 的原生服务测试还需要 Unix socket。测试只创建临时资产，不自动发布到共享业务资产目录。

## 人工测试准备

平台验证使用默认的远端执行面启动（不设置 `PYROMIND_PI_TERMINAL_BACKEND` 时即为 `sandbox`）：

```sh
./start_inference.sh
```

本地无平台凭据时改用支持的 OS 沙箱环境：

```sh
PYROMIND_PI_TERMINAL_BACKEND=os-sandbox ./start_inference.sh
```

任务包作者可参考 AgentGenome 的 `templates/data-cleaning-v2`。新包需通过宿主验证后显式发布，无策略的旧纯脚本包默认允许修订，由 Pi 声明业务脚本与验收资源；显式禁止的包仍拒绝。`genome_run` 返回后让出当前轮次；阶段任务和最终结果回到同一会话，不需要手动轮询。

本版不恢复重启前的节点任务、不自动修复或重放，不启动独立 Pi worker。取消节点只针对该请求对应的轮次；完成通知仍由 Runtime 发送一次。

## 固定脚本 revision 快速验证

在 AgentGenome 根目录使用临时资产与工作区验证，不影响正在使用的共享资产：

```sh
test_root=$(mktemp -d /tmp/genome-revision.XXXXXX)
AGENTGENOME_HOME="$test_root/registry" \
workspace_dir="$test_root/workspace" \
OH_CONVERSATIONS_PATH="$test_root/workspace/conversations" \
./start_sdk.sh test-revision
```

若有意在共享目录验证，须先停止使用同一资产目录的 SDK。

命令使用 SDK 的真实 Pi OS 沙箱，不调用模型；从旧 `data-cleaning@1.0.0` 验证两批 CSV、脚本修订、逗号与分号回归和重复提交。全部通过才自动发布候选并切换 latest。建议设置临时 `AGENTGENOME_HOME` 和 `workspace_dir` 测试。普通 `start` 不导入、发布或重放资产。

远程沙箱的提示约束修订由 Runtime 测试中的无只读能力执行宿主覆盖；真实远程平台仍需在连接的平台上做一次人工验证。测试用脚本识别逗号及分号，保持旧 `data_file` 调用可用，旧包内容不变。参数值调整不发布新版本；脚本或参数接口变更先冻结候选，再执行本次任务与全部回归。已有样例不可删除或降低断言；并发 latest 变化时保留候选，不覆盖新版本。

## terminal 数据处理

新会话不再注册 `df_run_pipeline`，历史消息类型继续兼容。普通 Python 直接通过 terminal 执行；DataFlow 使用技能的 `sandbox_sample.py`，优先 `--pipeline/--input/--output/--model-profile/--output-schema/--timeout` 直参，只有 flags 覆盖不到的字段才用 `--config` 读配置，用法见技能 `references/terminal-cli.md`。模型配置由宿主在内存中注入，远程命令脚本和日志不保存密钥；重连只恢复观察。平台提交工具不变。

远程终端按三段各自计时：端点解析（`sandbox.ensure`，含平台创建与工作区准备，默认 15 分钟）、准备（脚本落盘、TTY 握手、启动回执，默认 3 分钟）、命令执行（工具传入的 `timeout`，默认 300 秒）。落盘分三层：会话静态的 `watch.sh`、`launch.sh`、`env_runner.py` 和会话环境 `env.b64` + `env.sig` 每个会话只下一次，握手回执顺带报告它们在沙箱里是否齐全、环境签名是否匹配，命中时零额外往返，环境变化才重新落盘；每条命令只搬 `cmd.sh`，staging 行按字节预算打包（单行默认 3072 字符，落在 Linux PTY 4096 字节规范队列内，可用 `PYROMIND_PI_TERMINAL_SCRIPT_CHUNK_CHARS` 调整），每批一条确认，整包（脚本 + 启动 + 字节校验）放得下时合成一跳完成 stage 与 launch，放不下才分批或走文件 API，文件 API 缺失、失败或超出准备预算时自动降级回 TTY，校验失败的批次退回逐行确认（`batch-fallback`）。启动回执本身就证明启动行执行过，正常路径不再轮询 `pid`/`rc`，只有重连后的观察路径才先确认启动。执行证据的 `staging`、`round_trips`、`timings_ms` 字段记录本次实际走的路径、往返次数和分段耗时；`PYROMIND_PI_TERMINAL_TRACE=1` 把每次确认的耗时（每条命令的首跳含建连）和每条命令的分段耗时各写一行 JSON 到 stderr，用来把平台侧延迟与命令自身耗时分开，stderr 不在协议通道上，行为与未开启时一致。前两段预算可用 `PYROMIND_PI_TERMINAL_RESOLVE_TIMEOUT_MS`、`PYROMIND_PI_TERMINAL_STARTUP_TIMEOUT_MS` 覆盖（毫秒），`PYROMIND_PI_TERMINAL_FAST_PATH=0` 可整体退回逐行确认的慢路径。
