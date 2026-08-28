# Target Formats

每个 output.jsonl 只能使用一种格式，不允许混合。完整偏好对使用 DPO 格式，其他
SFT/GRPO/对话数据使用 messages 格式。

## Messages

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

顶层只允许 `messages`，不得保留业务字段或 metadata。

- `messages` 是非空数组，必须至少包含一条 user，并保留原始轮次顺序。
- assistant 不是通用格式的必需角色：SFT 需要 assistant 监督信号；GRPO prompt
  可以只有 system/user。后续工作流必须按实际角色选择训练阶段。
- `role` 只能是 `system`、`user`、`assistant` 或 `tool`。
- `content` 可以是非空字符串或多模态 content parts。文本 part 使用
  `{"type":"text","text":"..."}`；图片、视频、音频 part 使用相应 `type`
  并提供非空 `url` 或 `path`。
- assistant 只有在包含非空、合法的 `tool_calls` 时才允许空 `content`。
- 标准工具调用使用 `type: "function"`、非空 `id`、非空函数名和对象类型的
  `arguments`。tool 消息使用 `tool_call_id` 关联已有调用，并保留或推断 `name`。
- 一个固定 system prompt 可以应用到整个数据集；不要逐条生成不同 prompt。

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Check whether 91 is prime."},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {"name": "factor", "arguments": {"n": 91}}
        }
      ]
    },
    {
      "role": "tool",
      "name": "factor",
      "tool_call_id": "call_1",
      "content": "91 = 7 * 13"
    },
    {"role": "assistant", "content": "91 is not prime."}
  ]
}
```

## DPO preference

源记录同时包含非空 prompt、chosen 和 rejected 时优先识别为 DPO：

```json
{"prompt":"问题","chosen":"更好的回答","rejected":"较差的回答"}
```

- 顶层只允许 `prompt`、`chosen`、`rejected`，三者都是非空字符串。
- chosen 和 rejected 必须不同；不得丢弃或默认选择任一分支。
- 当前 Pyromind DPO Builder 只支持文本字段。对话数组或多模态偏好对无法无损转成
  三个字符串时，必须询问用户，不得静默展平。
- 只有用户明确要求把偏好数据改造成 SFT 时，才允许选择单分支；该结果属于有损
  messages 转换，不再是 DPO 数据。
