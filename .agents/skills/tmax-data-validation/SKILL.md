---
name: tmax-data-validation
description: >
  Tmax 类终端任务数据可用性验证:对每条数据(镜像 + 题面 + verifier)启动
  custom 沙箱加载镜像,验证环境可运行、verifier 可执行,逐条判定 usable/error
  并产出 verdict 清单,过滤出可用于 slime 训练的数据。支持清单驱动批量验证、
  断点续跑、镜像去重与抽样渐进。格式转换衔接 data-cleaning。
license: MIT
---

# Tmax 数据可用性验证(Tmax Data Validation)

## 适用边界

| 场景 | 工具/skill |
| --- | --- |
| 数据格式转换(无环境要求,如 slime jsonl 字段映射) | data-cleaning |
| 内容级清洗/LLM 处理 | data-preparation |
| **逐条验证"镜像可启动 + verifier 可运行",过滤可用数据** | **本 skill** |
| 镜像内安装环境 + codex 做内容处理 | environment-data-processing |

本 skill 针对形如 [TMax-15K](https://huggingface.co/datasets/allenai/TMax-15K)
的终端 Agent 任务数据:每条数据 = Docker 镜像引用 + 题面 + `tests/test.sh`
verifier。目标是判定每条数据**可验证、可测试**(镜像可拉取启动、workdir
存在、verifier 能跑完给出确定结果),产出 `verdicts.jsonl` 供下游格式转换使用。

## 前置条件

- 数据清单已就绪:storage 中 `manifest.jsonl`,每行含 `task_id` / `image` /
  `test_sh`(verifier 全文,可选 `setup_sh`)
- 当前会话已配置 Pyromind 认证(secret `auth_token` + env/cluster)
- 依赖工具:
  - sandbox_create、sandbox_terminal（沙箱生命周期与命令执行）
  - sandbox_write_file、sandbox_read_file（沙箱文件读写）
  - sandbox_upload、sandbox_download（沙箱与存储间文件传输）
  - sandbox_delete（沙箱清理）

## 工作流(严格按序执行)

### 1. 确认清单与抽样规模

- 确认 `manifest.jsonl` 的 storage 路径与字段;抽样策略见
  `references/batch-orchestration.md`(先 smoke 20 条)
- 与用户确认抽样规模与判定口径(默认:exit 0 或非 0 均算可测)

### 2. 启动沙箱（sandbox_create）

```json
{"image": "hamishi740/swerl-tmax-v3:618e344e0172", "sandbox_type": "custom", "cpu": 4, "memory": "8Gi", "wait_timeout": 300}
```

- 平台自动拉取镜像;记录返回的 `sandbox_id`
- 拉取失败/启动超时 → 该条记为 `error`(reason=镜像拉取失败),**不要重试同镜像**,继续下一条
- 同一 `image` 已在本次会话验证过环境 → 跳过重复 create,直接复用经验(见 batch-orchestration)

### 3. 环境探测（sandbox_terminal）

```json
{"sandbox_id": "<id>", "command": "test -d /home/user && echo OK || echo NO_WORKDIR; command -v bash", "timeout_seconds": 60}
```

- workdir 缺失 / 基础命令不可用 → 记 `error`,跳转到步骤 6(清理)

### 4. 写入 verifier（sandbox_write_file）

```json
{"sandbox_id": "<id>", "path": "/workspace/__tmax_test__.sh", "content": "<test_sh 全文>"}
```

- 仅 custom 沙箱支持;内容长且含特殊字符时优先 sandbox_write_file,避免 terminal heredoc 转义错误

### 5. 执行 verifier（sandbox_terminal）

```json
{"sandbox_id": "<id>", "command": "bash /workspace/__tmax_test__.sh", "timeout_seconds": 300}
```

- 判定规则(完整版见 `references/verification-rules.md`):
  - verifier 正常跑完(exit 0 或非 0)→ `usable`(reward 0/1 都是有效训练信号)
  - verifier 本身报错(语法错误、缺依赖、命令不存在)→ `error`
- 立即把 verdict 追加写入本地清单(见 batch-orchestration 的 verdict 结构)

### 6. 删除沙箱（sandbox_delete,硬约束）

```json
{"sandbox_id": "<id>"}
```

- **无论成败必须 delete**,禁止跳过;批量验证时逐条清理,不累积沙箱

### 7. 循环与回传

- 逐条执行步骤 2-6,直到清单处理完毕或达到抽样规模
- 完成后回传 `verdicts.jsonl`:**最后一条数据的沙箱删除前**,用 sandbox_write_file
  写入沙箱并 sandbox_upload(沙箱有 workspace mount 时也同时落到 storage):

```json
{"sandbox_id": "<最后一条的沙箱>", "path": "/workspace/verdicts.jsonl", "content": "<verdicts 全文>"}
{"sandbox_id": "<最后一条的沙箱>", "sandbox_path": "/workspace/verdicts.jsonl", "storage_path": "datasets/tmax/verdicts"}
```

- 然后删除该沙箱,完成清理
- 中断恢复:已有 `verdicts.jsonl` 时从最后已处理 `task_id` 之后继续,不重复验证

## 结果衔接(格式转换)

`usable` 的 task_id 清单交给 data-cleaning 做 slime 格式转换(字段映射见
`references/slime-conversion.md`)。转换是纯 Python 处理,不需要沙箱。

## 硬性约束

1. **delete 不可省略**:每条数据验证完必须清理沙箱
2. **不重试坏镜像**:同一镜像失败即记 error,换下一条,避免拉取风暴
3. **verdict 实时落盘**:每条判定后立即追加写,不批量攒内存
4. **抽样渐进**:先小批确认 verdict 分布,再扩大规模,不直接全量
