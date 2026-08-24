---
name: inference-validation-report
description: Call an OpenAI-compatible deployed inference endpoint on a validation JSONL, resume and retry predictions, calculate task-appropriate metrics, and generate a self-contained HTML review report with per-sample inputs, outputs, media, and optional boxes. Use when users ask to test a deployed model/API on a validation or eval set, assess fine-tuning results, compare predictions with ground truth, inspect misclassified samples, or produce a shareable validation report for text-only, structured JSON, single-image, multi-image, or visual inspection datasets.
---

# Inference Validation Report

Use `scripts/evaluate_inference.py` as the canonical implementation. Do not rewrite an
endpoint-specific evaluator unless the endpoint or dataset cannot be represented by its
arguments.

## Workflow

1. Locate the validation JSONL and inspect at most three rows. Read
   [schema-adapters.md](references/schema-adapters.md) when automatic fields are ambiguous.
2. Determine the endpoint URL, model name, API-key environment variable, output directory,
   and whether the user wants the original distribution or a derived balanced subset.
3. Keep secrets out of scripts, JSONL, HTML, shell output, and final responses. Export the
   key only through the named environment variable.
4. Run up to three rows first with a fresh temporary output directory. Check that media is
   found, requests succeed, targets parse correctly, and the inferred task type is sensible.
5. Run the full validation set. Reuse the same final output directory when resuming;
   `predictions.partial.jsonl` prevents repeated successful calls.
6. Verify `predictions.jsonl` has the same row count as the validation JSONL, API errors are
   accounted for, and `metrics.json` matches the headline HTML values.
7. Lead the handoff with the result, class-specific weaknesses, and clickable links to the
   HTML report, metrics, predictions, and evaluation script.

## Command

```bash
export INFERENCE_API_KEY="<secret>"

python3 scripts/evaluate_inference.py /path/to/validation-or-dataset-dir \
  --endpoint "https://host/v1/chat/completions" \
  --model "deployed-model-name" \
  --output /path/to/report-dir \
  --temperature 0 \
  --workers 8
```

Use `--api-key-env NAME` for a differently named environment variable. Use
`--responses-jsonl` for offline report regeneration without another model call.

## Adaptation Rules

- Preserve the validation set's existing distribution unless the user explicitly requests
  resampling. Never duplicate validation records to balance them.
- Prefer automatic field discovery. Supply explicit `--gt-field`, `--user-field`,
  `--system-field`, `--images-field`, or `--messages-field` only when inspection shows
  ambiguity.
- Use `--task-type classification|generation|structured` if auto-detection is wrong. For
  classification, pass `--labels pass,defect,fail` when the legal label set is known.
- Text-only rows require no special flags. Top-level `images`, message content image parts,
  single images, and arbitrary multi-image rows are supported.
- Use `--box-image-index N` only when boxes should be drawn on a particular image. Default
  to the first image. Use `--box-scale 1` for normalized boxes or `--box-scale 1000` for the
  common 0–1000 coordinate convention when auto-detection is unsuitable.
- The report must remain self-contained and shareable. It may become large because local
  media is embedded as data URIs.

## Interpretation

- Classification: report accuracy, Macro-F1, per-class precision/recall/F1, distributions,
  and confusion matrix. Do not substitute token accuracy for class accuracy.
- Generation: report exact match, normalized match, token-F1, and sequence similarity.
- Structured JSON: report valid JSON rate and canonical structural exact match in addition
  to overlap metrics.
- Localization: report box precision/recall at IoU 0.5 only when either target or prediction
  contains boxes.
- Always inspect per-class support. High overall accuracy can hide low recall on rare labels.
- Treat malformed responses and API failures as invalid predictions, not successful cases.

## Output Contract

The output directory contains:

- `dataset_profile.json`: detected fields, task type, labels, and media coverage.
- `predictions.partial.jsonl`: append-only checkpoint for resume.
- `predictions.jsonl`: ordered final predictions and per-sample scores.
- `metrics.json`: machine-readable aggregate metrics.
- `validation_report.html`: self-contained visual review report.

Do not include the API key in any artifact.
