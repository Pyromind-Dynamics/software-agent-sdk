# tmax 批量验证(case)

[environment-processing 范式](../playbook.md)下的具体场景 case。数据形态:成批
记录,每条含 `image`(docker 镜像引用)+ 题面 + verifier/test_sh——**无需用户
点名数据集类型**,数据形态匹配即走本 case。

- 执行定义(profile):`scripts/edp/profiles/tmax-validation.json`
- 链路:起镜像沙箱 → 装 pi coding agent → pi headless 解题 → 同箱写入并执行
  test_sh → 按 reward 文件/退出码判定
- 执行模式:模式 A(冻结运行时 runner);三段链路、门禁与批次策略见
  [playbook](../playbook.md)

## manifest 字段契约

`task_id` / `image` / `workdir`(tmax 默认 `/home/user`)/ `prompt`(题面)/
`test_sh`(verifier 全文)。

## 数据源定位

storage 上的 tmax 目录是 HF 数据集镜像:

| storage 路径 | 内容与用途 |
| --- | --- |
| `data/train-*.parquet` | **渲染主源**(任务索引,每行一个 task_id 对应 `tasks/` 同名目录);parquet 是二进制,preview 按文本读会乱码,用 `mode='sample'` 物化后以 pandas 解析字段 |
| `tasks/<task_id>/` | 每条任务的原始素材,仅供交叉核对:`container.def`(Apptainer 构建定义,**不是**可直接传 sandbox_create 的镜像引用,manifest 的 image 用 join 源映射)/`setup.sh`/`test_initial_state.py`/`test_final_state.py`(与 test_sh 交叉核对)/`task_summary.txt`(题面摘要)/`solutions/`(**参考解,严禁写入 manifest 或 prompt**——RL rollout 需要解题 agent 真实解题,泄漏参考解会污染训练数据) |
| `tasks.zip` | tasks/ 的原始压缩包,勿重复解压 |

## 字段映射

**以实际 parquet schema 为准,HF 发布版字段名可能不同**;若字段对不上先
`preview_dataset mode='sample'` 确认再调整模板:

| manifest 字段 | 来源(实测 tmax parquet 13 列) | 说明 |
| --- | --- | --- |
| `task_id` | `task_id` | 原样 |
| `image` | **不在 parquet**(无 swerl 镜像引用,`container_def` 只是 Apptainer 定义) | 模板用 `join` 从 storage 上的映射表(JSONL/CSV,含 task_id+image 列)查得;缺映射任务记入 `render_failures.jsonl` 并逐条报告,**不猜 tag** |
| `workdir` | 固定 `/home/user` | 数据集约定 |
| `prompt` | `description` | 原样(题面),不做裁剪 |
| `test_sh` | `test_final_state` 列 | 渲染时自动包装:heredoc 写入 `/workspace/test_final_state.py` → `pytest` → 按 rc 写 `/logs/verifier/reward.txt`(1.0/0.0) |

## 渲染模板示例

标准形态:

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
逐任务目录):

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

字段 spec 通用形态见 [playbook](../playbook.md)。

## pi 解题链路(OpenAI 协议直连,实测)

- install_pi 把网关注册进沙箱内 `$HOME/.pi/agent/models.json`(provider
  `mygw`,openai-completions 协议,baseUrl 规范化为 `/v1` 结尾),run_pi 以
  `--mode json --provider mygw --model <LLM_MODEL>` headless 解题
- 网关需提供 OpenAI chat-completions 协议;`LLM_BASE_URL` 带不带尾部
  `/v1` 均可(runner 注册 provider 时统一规范化为 `/v1` 结尾)
- `LLM_MODEL` **必须显式传**——不传时 pi 用 provider 默认模型,请求打到
  非预期模型(模型名原样透传给网关,如 `openai/deepseek-v4-flash-0731`)
- 凭据只进 models.json,不进启动脚本(pre/us-west-1 实测 deepseek 自定义
  网关,任务 8243 一次跑通 usable + reward 1.0)
- pi 要求 node ≥ 22.19:install_pi 无条件安装官方 Node 22.19(`.tar.gz`
  发行包,不依赖镜像内 xz)到 /opt 并链接 /usr/local/bin(tmax 镜像自带
  node12 也不受影响)
- LLM 凭据三件套从会话 secret 取,缺则 `DF_API_BASE_URL`(去尾部 `/v1`) /
  `DF_API_KEY` / `DF_MODEL_NAME` fallback,再回退 legacy `ANTHROPIC_*`;值经
  节点命令注入,不落 verdicts

## 错误分类枚举(verdicts 的 error_category)

`create_failed` / `probe_failed` / `pi_install_failed` / `pi_run_failed` /
`verifier_failed` / `verifier_env_missing`(镜像缺件可修,非数据不可用,
汇报时单独分桶)

## SFT 轨迹格式

- 输入为 pi 轨迹 `traces/<task_id>.pi_trace.jsonl`(`--mode json` 事件流:
  message_end 的 user/assistant/toolResult 事件,thinking 块丢弃,
  text/tool_calls/tool 结果映射为 OpenAI 风格 messages)
- pi 轨迹**首条 user 消息即启动题面**,不重复前置
- legacy `.cc_trace.jsonl`(CC 时代轨迹)仍兼容,该格式不含题面,由片
  manifest 的 `prompt` 作为首条 `user` 消息前置
- 默认只转 `reward >= 1.0` 的解题成功轨迹(`min_reward` 可调);slime 三键
  映射见 [slime 转换契约](../slime-conversion.md)
