# 批量编排(batch-orchestration)

## 输入清单 manifest.jsonl

每行一条待验证数据,字段:

```json
{
  "task_id": "task_000000_c19dda5b",
  "image": "hamishi740/swerl-tmax-v3:618e344e0172",
  "workdir": "/home/user",
  "test_sh": "#!/bin/bash\nset -e\n...",
  "setup_sh": "可选,一般已 bake 进镜像"
}
```

清单本身放在 storage;沙箱内需要时用 `download` 拉取,agent 侧可直接读取。

## verdict 结构(输出)

`verdicts.jsonl`,每行:

```json
{
  "task_id": "task_000000_c19dda5b",
  "image": "hamishi740/swerl-tmax-v3:618e344e0172",
  "status": "usable",
  "reason": "",
  "test_exit_code": 1,
  "duration_s": 42
}
```

- `status`: `usable` / `error`
- `reason`: error 分类(见 verification-rules.md),usable 时为空字符串
- 实时追加写:每条判定后立即追加,不攒内存批量写

## 断点续跑

1. 会话中断/超时后,先读取已有 `verdicts.jsonl`,提取已处理的 `task_id` 集合
2. 从 manifest 中跳过已处理项,继续验证剩余项
3. 已处理项的判定结果**不重跑、不覆盖**
4. 全部完成后才回传最终 `verdicts.jsonl`(覆盖旧版)

## 镜像去重

- 同一 `image` 的多次 create 会重复拉镜像,成本高
- 会话内维护 `image -> 环境验证结果` 缓存:
  - 环境已验证 OK:同 image 的新条目**跳过 create/探测**,直接 file write + 跑 verifier
  - 环境验证失败:同 image 的新条目直接记 `error`(reason 复用),不重复 create
- 仅当 manifest 中 `image` 完全一致时才复用(含 tag)

## 抽样渐进策略

| 阶段 | 规模 | 动作 |
|------|------|------|
| smoke | 20 条 | 全量流程,确认 verdict 分布与 error 主因 |
| 用户确认 | - | 展示 usable 比例、error 分类统计、典型失败原因 |
| 分批扩大 | 每批 ≤200 条 | 按批次续跑,每批后汇报统计 |
| 全量 | 剩余 | 仅当用户明确要求 |

- 抽样用 manifest 前 N 条或随机种子采样,保持可复现
- 全量前先确认镜像去重收益(不同 tag 比例),评估耗时

## 并发与资源

- 沙箱逐条创建/删除,**不并行创建多条**(平台资源与镜像拉取额度限制)
- 单条验证默认配额:create wait ≤300s,探测 60s,verifier ≤300s
- 每处理 50 条汇报一次进度(已处理/总数、usable 数、error 数)
