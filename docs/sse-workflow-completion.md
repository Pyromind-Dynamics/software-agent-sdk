# SSE 消息与 workflow 结束回调

## 分层和依赖

前端按 message_id 将消息事件应用到时间线，以 REST through_seq 为基线，
只重放更新的 SSE 事件。run_id 只关联执行轮次，不充当消息身份。

Product Runtime application 的 WorkflowCompletionHook 负责 dirty 判断、
输出去重、持久化恢复以及 workflow 输出先于最终状态的顺序。
内部 workflow.modified / run.finished 不进入公开 ProductEvent 协议。
HarnessAdapter.finalize_run 只保存已捕获的最终文件与原生 checkpoint，
返回统一 WorkflowState。依赖仍为 Adapter → Runtime ports/domain；
Runtime 不依赖 Harness 或 Server。

Pi 在原生 finishRun 中冻结 DSL 与 checkpoint，Adapter 归一化结束结果。
成功 write/edit 目标 workflow.py 才发修改事实。输入画布仅保存 in 快照。
OpenHands 通过 Adapter 将旧 dirty hook 的输出转为内部生命周期事件，
历史回放继续使用原有 ProductEvent 身份；不重复发布旧事件。

## 兼容与恢复

公开 API、ProductEvent、ConversationSnapshot 格式和版本保持不变。
新增 product/workflow-runs.json 保存产品收尾状态，旧会话缺失此文件视为空。
Pi 的 pi/run-completions.json 保留尚未交付 Product Runtime 的原生完成记录，
避免下一轮覆盖 inflight 后丢失上一轮结果；完成后清理原生待处理记录。
Product Store 先保存不可变完成记录，再调用 Adapter、保存输出，最后确认完成。
稳定事件 ID 和原生快照 ID 使重复执行幂等。请求凭据不进入这些记录。

前端 workflow 卡片使用 timeline.item_id 发起 fork/rollback；workflow.version
只是内容版本，不能当成 ProductEvent 的 event_id。

## 验证

前端覆盖同轮多消息、快照与增量乱序、重连和事件身份。
Runtime/Adapter 覆盖零次与多次编辑、停止/失败、内容复原、转换失败、
丢失内存队列、发布后重启、checkpoint 与最终状态顺序。
Pi 原生集成测试验证每条消息增量拼接等于完成正文，且结束帧保存最终 DSL。
