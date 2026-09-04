# 具身机器人数据清洗

本 case 处理 S2 自采集录制或挂载的 Hugging Face LeRobot v2.1 数据，执行
多模态时间对齐、静止帧清理、动作与标签关联、批量转换和 LeRobot v2.1 校验。

## 执行入口

读取 `../../../../../embodied-data-cleaning/SKILL.md` 并严格按其中的运行时契约
和批处理流程执行。该 case 使用通用 Sandbox 工具和固定的
`openhands-embodied-runtime==1.29.5`，不通过 Studio、`edp_submit` 或旧的
`run_embodied_cleaning_sandbox` 专用工具。

## 架构边界

- `data-processing` 负责统一路由和用户确认门禁。
- `embodied-data-cleaning` 描述本 case 的数据契约、质量门和执行顺序。
- `sandbox_create` / `sandbox_terminal` / `sandbox_read_file` /
  `sandbox_delete` 提供容器控制面。
- `openhands-embodied-runtime` 在 Python 3.10 沙箱内完成确定性处理。
- rejected episode 只输出结构化错误报告，不自动 repair；只有运行时失败的
  episode 可以通过 `resume` 重试一次。
