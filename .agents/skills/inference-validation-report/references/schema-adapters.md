# Dataset Schema Adapters

## Supported Input Shapes

### Flat text or multimodal rows

```json
{
  "id": "sample-1",
  "images": ["images/a.jpg", "images/b.jpg"],
  "system_prompt": "You are an inspector.",
  "user_prompt": "Judge this sample.",
  "gt": "<answer>pass</answer>"
}
```

The automatic field order is:

- ID: `id`, `sample_id`, `uid`, `name`, otherwise the row index.
- Target: `gt`, `ground_truth`, `expected`, `answer`, `label`, `target`.
- System: `system_prompt`, `system`.
- User: `user_prompt`, `prompt`, `instruction`, `question`, `input`.
- Media: `images`, `image_paths`, `image`.

Override ambiguous fields with the corresponding CLI option.

### Message rows

```json
{
  "id": "sample-2",
  "images": ["images/a.jpg"],
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Classify the image."},
    {"role": "assistant", "content": "cat"}
  ]
}
```

The final assistant message becomes ground truth and is removed from the request. Other
assistant messages are also treated as targets, so use flat fields for multi-turn histories
that contain assistant context before the final target.

Message content may contain OpenAI-style image parts:

```json
{
  "role": "user",
  "content": [
    {"type": "image_url", "image_url": {"url": "images/a.jpg"}},
    {"type": "text", "text": "What is shown?"}
  ]
}
```

Local relative paths resolve against the JSONL parent directory. Absolute paths, HTTP(S)
URLs, and existing data URIs are accepted.

## Target Extraction

For classification, the evaluator checks the last JSON object for one of these keys:
`result`, `label`, `answer`, `prediction`, `class`, `category`. It then checks
`<answer>...</answer>`, followed by the whole trimmed response.

Boxes are read from `boxes`, `bboxes`, or `bbox` in the last JSON object. Supported box
shape is `[x1, y1, x2, y2]` or a list of such boxes.

## Automatic Task Detection

- Structured JSON in at least 80% of targets plus a small repeated answer vocabulary:
  classification.
- Structured JSON in at least 80% without a small label vocabulary: structured.
- Short targets with a small repeated vocabulary: classification.
- Otherwise: generation.

Use explicit `--task-type` and `--labels` for authoritative control. Auto-detection is a
convenience, not a replacement for known dataset semantics.

## Offline Responses

To rebuild metrics or HTML without endpoint calls, provide JSONL rows containing `id` and
one of `raw_response`, `response`, or `output`:

```json
{"id": "sample-1", "raw_response": "pass"}
```

Every validation ID must exist in the offline response file.
