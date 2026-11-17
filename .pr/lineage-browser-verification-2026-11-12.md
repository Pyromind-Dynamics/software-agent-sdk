# 运行血缘归档验证

日期：2026-11-12。未修改生产代码，未重启服务，未批准页面中的删除命令，未提交训练或评测任务。

## 结论

归档逻辑通过真实历史数据的隔离验证；当前 Chrome 环境的端到端验收被配置和会话状态阻塞，不能判定通过。

## Chrome 与服务现状

- 前端：localhost:5173，Vite `/api` 代理到 localhost:8000。
- 当前 conversation：`3f23ac7a48b34a7eb023ca53168b9724`。
- 页面和持久化快照均为 `waiting_for_confirmation`，有 1 项待确认操作，命令包含 `rm -rf`。
- 当前会话 external_tasks 为 0、workflow execution requests 为 0，版本账本尚无绑定项目；没有可触发运行归档的平台提交。
- 实际后端监听进程 PID 2342 / 2351 的 `PYROMIND_RUN_ARCHIVES_ENABLED` 均未设置。装配代码默认关闭归档。
- 未读取或转存浏览器 Cookie；连接面板检查后恢复折叠状态。

## 真实数据隔离验证

从与当前会话相同 owner/profile 的最近版本项目 `p21` 读取现有数据：

- conversation：`315bd1884667469e8e0e8108f9ebb09c`。
- 原版本数据库使用 SQLite 只读连接，并 backup 到临时数据库。
- 输入使用真实 `snapshot-staging` 缓存，读取配置/代码时走现有摘要校验。
- 调用本轮实现的 `Versions.archive_run()`，输出只写临时目录；无远程 Storage 请求。
- 不调用 submit、任务回调或现有会话对账，避免影响正在运行的服务。

| Run | 类型 | 文档数 | 关联产物数 | 验证 |
| --- | --- | --- | --- | --- |
| r79 | workflow | 2 | 3 | 通过 |
| r80 | pipeline | 5 | 10 | 通过 |
| r81 | pipeline | 5 | 10 | 通过 |
| r82 | pipeline | 5 | 10 | 通过 |
| r83 | pipeline | 5 | 10 | 通过 |
| r84 | pipeline | 5 | 10 | 通过 |
| r85 | pipeline | 5 | 8 | 通过 |

每次验证检查：

1. 归档快照完整性校验通过，可读回 JSON。
2. taskId、conversation、execution 与输入事实一致。
3. 包含配置/代码文本以及产物关系；不重复暴露 manifest.source。
4. 相同输入再次归档返回相同内容寻址身份。
5. 运行数和产物版本数不改变。

`r85` 在已有快照中没有匹配的 external task 状态，归档保留 null，未补造 succeeded。

## 尚未验证

- 开启开关后的在线自动触发、远程 Storage 上传和浏览器操作的全链路。
- 新实验 vN、严格输入冻结、checkpoint 主动发现不在本轮已实现范围。

下一步需要开启 `PYROMIND_RUN_ARCHIVES_ENABLED=1` 并重启后端。当前有待确认会话，因此未擅自中断服务；重启与真实任务执行需先确认影响。
