# 实验聚合与查询复测

日期：2026-11-13。仅测试与记录，不修改业务实现，不提交或重跑平台任务。

## 2026-11-13 新建实验实测

本节为用户明确要求“新建一个实验试试”后的独立测试；本次确实提交了一个训练任务，覆盖前文“未提交任务”的范围说明。未修改业务代码，未修复用户暂缓的安全/性能问题。

- conversation：`1ce0973becf447bba54051b9c259af2a`。
- 新业务 Project：`2f0422aea5694bbeb6419cc2c5bd2cfc`，EchoMind / us-west-1。
- 计划：只读核验现有 train 3 / val 2 → 1 epoch 单卡 LoRA SFT → 使用训练已有验证预测 CPU 评分。不新增 GPU 基线推理，不覆盖原始数据。
- Dataset 实测 63/63 检查通过；业务 Project 通过页面“创建并启动”产生。
- 实时 DSL 校验 valid=true；首次平台训练 task `10345`，仅一次提交。
- 浏览器现有登录直达实验 API，返回 `artifact_project_id=p1`、`experiment_id=r1`、`version=v1`；业务 Project、conversation、taskId 均匹配。
- 终态任务 `failed`；现有版本账本维持 1 Project / 1 Run / 3 ArtifactVersion，平台任务数 1，没有因查询或归档重复提交。
- 自动导出同一 v1 的 pending/failed 观察清单；文件摘要校验通过，清单携带非空运行 archive_ref，评测为空，没有伪造模型或指标。
- 跨 Pyromind 路由查询该 EchoMind conversation 返回 Conversation not found；不存在的实验详情返回 Experiment was not found。
- 实验 API 普通查询中的 archive_ref 为 null，而导出清单含引用；本次未修复该查询能力缺口。
- 普通查询的 inputs 为模型引用、配置和代码快照，未单独登记 train/val 数据产物；原有严格输入冻结与结构化训练数据提取仍未实现。

### 平台阻塞

8 个 Config Builder 节点成功，ModelTrainSFTNode 在 `on_train_begin → wandb.init` 失败：

```text
WANDB_API_KEY invalid: API key must have 40+ characters, has 13.
```

本次仅传 Secret 名 `WANDB_API_KEY`，平台未做 Secret 解析，而按 13 字符字面量传给 W&B。失败发生在第 1 个优化步之前，输出目录为空，无 checkpoint、无验证预测，因此没有启动 Eval。原 Train 子会话通过平台回调恢复并给出失败诊断；没有使用明文 Key 绕过，没有重提训练。

最终页面为 waiting_for_input，等待用户选择如何处置 W&B 阻塞；未擅自选择、取消或重跑。截图：`/tmp/echomind-lineage-acceptance-2026-11-13/new-experiment-blocked.png`。

### 本次结论

真实“新 Project → 首次提交 → v1 登记 → failed 终态归档”链路通过。成功训练后的模型关联、CPU 评分挂接、非空指标及远端逐份读回仍未完成，不判定完整端到端通过。

重新运行完整 Runtime 测试：758 passed、6 skipped、1 个现有 Starlette 弃用警告。浏览器部分 API 标签导航出现 CDP 超时，未据此推断 API 错误；有效列表回包和本地导出证据已分别核对。

## 2026-11-13 10:50 当前环境复测

本节覆盖下文较早检查结果，仅描述此次复测实际观察：

- Chrome 原标签仍显示历史 conversation `ce867fa036284b8297f5b713bf121134`。刷新后页面为 Not connected、Project 0、对话 0；没有新建任务或发送消息。
- 监听 8000 的后端进程 PID 10867/12305，归档开关为 1；OpenAPI 返回 200，包含两套前缀的实验列表、详情与指标共 6 条路由。
- 使用服务已配置的测试 Cookie 发起只读实验查询，返回 401；调试身份请求也返回 401。未更改鉴权设置、未签发身份、未导出浏览器凭据。
- 当前 `workspace/conversations` 为空；只读查询 `workspace/product-versions/versions.sqlite3` 的 projects 表为空，snapshot-staging 也为空。本次无历史实验可核对，不能把此前的历史账本测试算作本次在线验证。
- 重新执行本文定向测试命令：66 passed，耗时 1.66 秒。
- 未重新触发运行归档，未验证新实验清单远端上传、真实评测挂接或非空指标；安全与性能问题继续按用户要求暂缓。

本次结论：自动化测试通过，当前在线环境不满足端到端验收条件。需要有效登录和一轮有真实凭证的数据再验收，不能宣称功能全通过。

## 结论

未达到完整功能验收标准。近期数据的只读投影和定向测试通过，但存在旧持久化数据兼容问题；当前历史数据不能验证真实评测挂接及指标输出。未重新测试用户暂缓的安全与性能问题。

## Chrome 与在线服务

- Chrome localhost:5173 的 EchoMind 页面可打开历史 Project，显示 Connected / Idle。
- 页面尚无实验版本列表或指标查询入口，此前第二步未实现该前端入口，不能把现有聊天页面可用等同于新功能已在前端可用。
- 后端进程已启用 `PYROMIND_RUN_ARCHIVES_ENABLED=1`；OpenAPI 可发现两套产品前缀下的实验列表、详情及指标路由。
- 服务环境中的测试 Cookie 已过期，单凭该 Cookie 请求实验列表返回 HTTP 401。浏览器现有登录可读历史对话，但本次没有导出浏览器凭据用于 API 请求。
- Session API Key 单独认证不会提供 Product owner 身份；携带该 Key 的只读请求返回 Conversation 404，不据此认定实验路由不存在。
- 尚无 `experiment.json` 新导出记录。未发起唤醒 Agent 的消息来强制导出，因此本次不声称验证了在线清单上传。

## 定向测试

命令：

```text
uv run pytest tests/pyromind_runtime/test_experiments.py tests/pyromind_runtime/test_version_integration.py tests/pyromind_runtime/test_package_boundaries.py -q -o addopts=''
```

结果：66 passed。包括列表/详情 API、鉴权隔离、实验关联、验证集选模、指标组、归档幂等与已有主链路回归。

## 真实账本只读检查

SQLite 用 mode=ro 读取，不调用平台，不修改会话与版本账本。

- 总计 21 个账本 Project。
- p1–p5 缺少本地 conversation snapshot，不参与投影。
- 6 个可解析 Project 投影稳定，重复结果相同；共生成 3 个 workflow 实验视图、18 个未关联评测、0 个指标行。
- 当前打开的 p19 对应 1 个实验、7 个未关联评测、0 个指标行。没有推断或补造模型来源。

### 发现：旧数据读取失败

10 个有快照的历史 Project 在读取现有模型时失败：

| Project | 被拒绝的旧字段 |
| --- | --- |
| p6、p7 | PipelineRun.task_run_id |
| p8、p10–p16 | WorkflowExecutionRequest.permission_id |

当前 ContractModel 使用 extra=forbid，相关加载器未兼容这些真实旧字段。新增实验查询调用这些加载器，因此会受到同样影响。该失败在实验聚合之前，不能归因于本次投影算法，但仍是全历史兼容验收的阻塞。

## 未完成的验收

- 使用有效身份请求在线实验列表、详情与指标，并与业务 Project 绑定核对。
- 真实训练/评分来源关联成功的评测及非空指标查询。
- 新实验清单的远端上传和读回。
- 旧数据迁移或兼容加载。

当前 66 项定向测试通过并不能覆盖以上真实环境缺口。
