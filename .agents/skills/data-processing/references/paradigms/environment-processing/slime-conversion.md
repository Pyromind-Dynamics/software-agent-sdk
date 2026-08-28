# slime 格式转换衔接(slime-conversion)

## 边界

- **本 skill 负责验证 + 轻量转换**:验证产出 `verdicts.jsonl`,经
  `scripts/convert_to_slime.py` 直接转 slime 三键(纯字段映射,本地跑,
  不进沙箱、无需平台执行)
- **复杂清洗仍交 format-conversion 处理范式**:去重/PII/语言过滤/内容改写等需要
  清洗算子的场景,以本转换产物为输入走 format-conversion 流程
  (预览 → 编写 clean_script → 平台执行 → 校验)

## 目标格式(slime jsonl 三键)

```json
{
  "prompt": [{"role": "user", "content": "<题面,剥 vanillux 尾部>"}],
  "label": "task_000000_c19dda5b",
  "metadata": {
    "protocol": "tmax",
    "instance_id": "task_000000_c19dda5b",
    "image": "hamishi740/swerl-tmax-v3:618e344e0172",
    "workdir": "/home/user",
    "problem_statement": "<题面全文>",
    "test_sh": "#!/bin/bash\nset -e\n..."
  }
}
```

## 字段映射(原始 HF → slime)

| 原始字段 | 转换后 | 规则 |
|----------|--------|------|
| `messages[user]` | `prompt` / `problem_statement` | 去掉 `Please solve this task:` 前缀与 vanillux「Recommended Workflow / 只许一次 bash」尾部 |
| `messages[system]` | 丢弃 | 面向 vanillux bash tool,slime 侧另配 coding agent 系统提示 |
| `env_config.image` | `metadata.image` | 原样 |
| `env_config.task_id` / `ground_truth` | `label` / `metadata.instance_id` | 原样 |
| 固定值 | `workdir=/home/user` | Tmax 约定(以 manifest 为准) |
| 固定值 | `protocol=tmax` | 必须显式写出,走同箱 grader |
| `task-data/.../tests/test.sh` | `metadata.test_sh` | deferred 注入,防偷看 |
| `task-data/.../setup.sh` | 默认不写 | 已 bake 进镜像 |

## 转换脚本(scripts/convert_to_slime.py)

```
python scripts/convert_to_slime.py \
    --manifest run-dir/manifest.jsonl \
    --verdicts run-dir/verdicts.jsonl \
    --out slime.jsonl [--protocol tmax]
```

1. 以 verdicts 过滤 manifest,只保留 `verdict=usable` 记录
2. 按 SKILL.md 字段映射写出 slime 三键(题面尾部剥离已在 manifest 构造时完成)
3. reward 不嵌入:rollout 时由 `metadata.test_sh` 实时判定;`usable` 但
   reward 0 的记录保留(RL 需要 0 分信号)

SFT 侧对应 `scripts/convert_to_sft.py`(输入 traces/ 轨迹 + verdicts,
默认只转 reward ≥ 1.0 的解题成功轨迹,输出 messages 格式)。

## 验证后过滤语义

| verdict | 转换阶段行为 |
|----------------|--------------|
| `usable` | 进入转换,写入 slime jsonl |
| `error` | 排除;error_category 汇总进清洗报告,便于溯源 |

## 验题约定(训练侧,不属于本 skill)

- 同箱 deferred 验题:agent 结束后把 `test_sh` 写入容器再执行
- 计分:优先读 `/logs/verifier/reward.txt`(0~1);否则 exit 0 → 1.0
- Tmax 不走 git_diff;不因空 patch 强制 solved=0
