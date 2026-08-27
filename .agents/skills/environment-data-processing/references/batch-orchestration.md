# 批量编排(batch-orchestration)

## 输入清单 manifest.jsonl

每行一条待验证数据,字段:

```json
{
  "task_id": "task_000000_c19dda5b",
  "image": "hamishi740/swerl-tmax-v3:618e344e0172",
  "workdir": "/home/user",
  "prompt": "You are tasked with ...",
  "test_sh": "#!/bin/bash\nset -e\n...",
  "setup_sh": "可选,一般已 bake 进镜像"
}
```

清单本身由 `edp_render` 在平台节点分片写入 storage
(`<root>/batch-XXX/manifest.jsonl` + `shards.json` 索引),agent 不在本地构造;
沙箱内需要时用 `sandbox_download` 拉取,agent 侧可 `preview_dataset` 直读。

## verdict 结构(输出)

`verdicts.jsonl`,每行(runner 实际输出结构):

```json
{
  "task_id": "task_000000_c19dda5b",
  "image": "hamishi740/swerl-tmax-v3:618e344e0172",
  "verdict": "usable",
  "exit_code": 0,
  "error_category": null,
  "cached": false,
  "reward": 0.85,
  "note": null
}
```

- `verdict`: `usable` / `error`
- `error_category`: error 时的分类(`create_failed` / `probe_failed` /
  `exec_failed` / `pi_install_failed` / `pi_run_failed` / `verifier_failed` /
  `verifier_env_missing`),usable 时为 null
- `reward`: verifier 写入 `/logs/verifier/reward.txt` 的 0~1 浮点;
  文件缺失时回退按退出码判定,此时为 null
- `note`: 分桶原因说明(如 verifier_env_missing 时记录命中的缺失特征)
- `verifier_env_missing`:verifier 输出含 `No such file or directory` /
  `command not found`(镜像缺件)。**可修镜像回收,不算数据不可用**;
  统计可用率时单独分桶,勿与真正的 verifier_failed 混计
- **读不到结果 ≠ reward 0**:判定链路本身坏(脚本崩、超时、路径错)记
  error,不与"任务没做对"混淆
- 实时追加写:每条判定后立即落盘,不攒内存批量写
- 除 verdicts 外,profile 开启 `export_trace` 时同步产出
  `traces/<task_id>.pi_trace.jsonl`(pi `--mode json` 事件流,供 SFT 转换)

## 断点续跑(输出即 checkpoint)

1. 会话中断/超时后,先读取已有 `verdicts.jsonl`,提取已处理的 `task_id` 集合
2. 从 manifest 中跳过已处理项,继续验证剩余项
3. 已处理项的判定结果**不重跑、不覆盖**
4. 平台路径:`edp_submit` 同 `run_id` 重提交即自动续跑;聚合用
   `edp_aggregate` 同 `out_dir` 重提交(可追加新 run_dirs),已合并的
   task_id 自动跳过;本地 runner 场景同 `--output-dir` 重跑原命令即续跑

## 镜像复用语义(pi 链路)

- **默认逐条真实执行**:pi 解题结果依赖题面,同镜像不同记录不可复用判定
- 仅**纯环境预检**场景(不跑 pi、只验镜像可启动/依赖可用)可加
  `--dedup-by-image` 按镜像复用 verdict
- **不重试坏镜像**:同一镜像 create 失败即记 `create_failed`,换下一条,
  避免拉取风暴;仅当 manifest 中 `image` 完全一致(含 tag)才可能复用缓存

## 抽样渐进与分批决策(决策权在用户)

| 阶段 | 规模 | 动作 |
|------|------|------|
| smoke | 第一片 limit=3 | 全量链路,确认 verdict 分布与 error 主因 |
| **用户决策** | - | **AskUserQuestion 三选一:全量 / 分批(批容量由用户拍板) / 停**,禁止 agent 默认全量 |
| 分批扩大 | 每批 N 片 | `edp_submit(shards=..., shard_count=N)`,每片独立 workflow/回调 |
| 批间确认 | - | 本批全部片终态后汇报统计,再次 AskUserQuestion:继续 / 调容量 / 停 |
| 全量 | 剩余全部片 | 仅当用户明确选择 |

- agent 给推荐批容量时附推算依据:单条均时 × 片内条数 vs 12h 任务配额、
  同镜像拉取风暴约束;3 条样本不符合预期的全量成本很高,决策权在用户
- 运行中用 `df_check_progress(output_dir=<run 目录>,
  tail_filename="verdicts.jsonl")` 看进度(节点日志亦逐条一行);
  N 片的终态回调逐个唤醒会话,先判断"本批是否全部终态"再决定是否询问下一批

## 并发与资源

- runner 串行执行,逐条创建/删除,**不并行创建多条**(平台资源与镜像拉取额度限制;
  多片并行的片数即批容量,由用户确认,防同镜像拉取风暴)
- 单条默认配额:create wait ≤600s,probe 60s,pi 安装 ≤600s,pi 解题总预算
  ≤1800s(runner 内部 nohup 后台 + 退出码文件轮询,不受单命令 600s 上限
  约束),verifier ≤600s
- 进度观测:三段平台化后统一走 `progress.json`(edp_render/edp_submit/
  edp_aggregate 同一契约,`df_check_progress` 可读),节点日志逐条一行;
  需介入时先 `df_stop_task(task_id)` 停平台任务再改再提交

## 手工兜底(runner 不可用时)

按 SKILL.md 模式 B 组装沙箱工具,判定口径与上文一致:verifier 正常跑完且
reward 可读(或 exit 0)→ usable;verifier 本身报错(语法错误、缺依赖、命令
不存在)→ error。无论成败每条必须 `sandbox_delete`(Running 态先 pause 再删)。

手工装 pi 时已验证的环境适配方法(runner 的 install_pi 已内置同样逻辑,
此处供模式 B 参考):

- pi 要求 node ≥ 22.19(tmax 镜像自带 node12,apt 装的老 nodejs 不可用):
  下载官方 `node-v22.19.0-linux-x64.tar.gz`(用 .gz 发行包,镜像可能缺
  xz)解压到 /opt 并 PATH 前置,再 `npm install --prefix /opt/pi
  @earendil-works/pi-coding-agent`
- 网关注册:把 OpenAI 兼容端点写进 `$HOME/.pi/agent/models.json`
  (provider 任意名,`"api": "openai-completions"`,baseUrl 以 `/v1` 结尾,
  apiKey/id 填真实凭据),启动用 `--provider <名> --model <LLM_MODEL>`;
  HOME 必须与解题工作目录一致(pi 在 $HOME/.pi 下找配置)
- root shell 下建议用镜像内非 root 账号(tmax 为 `user`)经
  `su -s /bin/bash user -c '...'` 执行,与文件属主对齐
- pi 长任务必须后台化:平台 exec 单命令 600s 上限扛不住解题,用
  `nohup ... > trace.jsonl 2> err.log; echo $? > exit.txt &` 后
  sleep 轮询 exit 文件(runner 的 run_pi 已内置该模式)
- 采集 pi 轨迹:启动加 `--mode json` 重定向到文件,得到 message_end
  事件流(user/assistant/toolResult),可直接交 `scripts/convert_to_sft.py`
  转训练格式(首条 user 消息即题面,无需 --manifest 前置)
