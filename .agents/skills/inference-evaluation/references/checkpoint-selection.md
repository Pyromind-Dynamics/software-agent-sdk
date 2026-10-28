# 持续 checkpoint 选模

训练前准备验证集字段映射、评价规则和资源配置；需要复杂业务指标时，按已有预测评分契约
准备 `parse_record`、`score_case` 及带 validation_examples 的 scoring_spec。
评分脚本的 `context.source` 为 `validation`，独立 test 为 `test`，样本 ID 与字段映射一致。

训练提交成功后注册，也可对仍在运行或已经结束的训练补挂：

```json
{
  "action": "register",
  "request_id": "生成一次并在重试时复用的 UUID",
  "arguments": {
    "training_task_id": "实际平台任务 ID",
    "training_node_id": "实际训练节点 ID",
    "template": {
      "input_path": "/datasets/validation.jsonl",
      "dataset_config_path": "public_data/inference_dataset_config.json",
      "evaluation_config_path": "public_data/inference_evaluation_config.json",
      "inference": {"gpu_product": "NVIDIA-L40S", "gpu_count": 1},
      "cpu": 4,
      "memory": 32
    },
    "selection": {"primary": {"metric": "pass_rate", "direction": "max"}}
  }
}
```

GPU 按实际模型选择，示例不代表默认选卡建议。复杂评分在 template 增加
`scoring_script_path` 和 `scoring_spec_path`；计划做独立 test 时，在 template.datasets
提供 train/test 的 `{"path":"Storage路径"}`，服务端冻结数据指纹。
所有 checkpoint 使用同一验证集、脚本、生成参数和工作流模板。

返回 run_id 后可结束当前轮次。`action="get"`、`arguments={"run_id":"..."}` 查询汇总；
`action="cancel"` 用同样参数取消选模及其评测任务，不停止训练。三种操作均传 request_id。
服务端每 60 秒查询平台、W&B 和文件元数据，每个选模任务串行评测；训练终止后补扫 10 分钟，
再等待子评测完成。关闭聊天不影响运行；服务重启后可能需要用户刷新平台登录凭据。

评测结果按 candidates 返回，最终汇总写入 output_dir 下的 checkpoint_selection.json 和
iteration_context.json。只能将成功、完整且同口径的评测用于排名；训练清理文件、评测失败和
未稳定产物会形成覆盖缺口。运行期间称“当前已评测候选中的最佳”。
提交状态 uncertain 时核对平台任务，不重复注册或重复提交。找到确切平台任务后，
使用 `action="reconcile"`，arguments 填 run_id、step 和 task_id；服务端验证任务名称与
工作流输出目录后恢复关联。无法确认则保留待处理状态。更新数据或评分规则时新建任务。
选模结束后，使用返回的 run_id 作为普通 inference 操作的 selection_run_id 执行独立 test。
