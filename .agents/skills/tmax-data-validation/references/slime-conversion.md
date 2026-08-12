# slime 格式转换衔接(slime-conversion)

## 边界

- **本 skill 只负责验证**,产出 `usable` 的 task_id 清单(`verdicts.jsonl`)
- **格式转换交给 data-cleaning**:纯 Python 字段映射,无环境要求,不进沙箱
- 转换入口:以 `verdicts.jsonl` 过滤后的清单为输入,调用 data-cleaning 的
  清洗脚本流程(预览 → 编写 clean_script → 平台执行 → 校验)

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
| `messages[system]` | 丢弃 | 面向 vanillux bash tool,slime 用 Claude Code + SWE_PROMPT |
| `env_config.image` | `metadata.image` | 原样 |
| `env_config.task_id` / `ground_truth` | `label` / `metadata.instance_id` | 原样 |
| 固定值 | `workdir=/home/user` | Tmax 约定(以 manifest 为准) |
| 固定值 | `protocol=tmax` | 必须显式写出,走同箱 grader |
| `task-data/.../tests/test.sh` | `metadata.test_sh` | deferred 注入,防偷看 |
| `task-data/.../setup.sh` | 默认不写 | 已 bake 进镜像 |

## 转换脚本模式(参考 convert_tmax_to_slime.py)

1. 读取 HF `tmax-15k-open-instruct` 数据集(或已有 jsonl)
2. 按 `task_id` 关联 `tests/test.sh` 全文
3. 剥离 user 题面的 harness 尾部
4. 写出 slime 三键
5. **本流程差异**:只输出 `verdicts.jsonl` 中 `status=usable` 的 task_id

## 验证后过滤语义

| verdict.status | 转换阶段行为 |
|----------------|--------------|
| `usable` | 进入转换,写入 slime jsonl |
| `error` | 排除;reason 汇总进清洗报告,便于溯源 |

## 验题约定(训练侧,不属于本 skill)

- 同箱 deferred 验题:agent 结束后把 `test_sh` 写入容器再执行
- 计分:优先读 `/logs/verifier/reward.txt`(0~1);否则 exit 0 → 1.0
- Tmax 不走 git_diff;不因空 patch 强制 solved=0
