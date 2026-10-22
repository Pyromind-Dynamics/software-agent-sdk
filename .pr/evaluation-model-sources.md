# Evaluation model sources

The existing `df_submit_pipeline` input fields remain unchanged. The referenced
evaluation JSON accepts `model_source.type` (`storage`, `clone`, `download`);
omission retains Storage semantics. Download also accepts `download_source`
(`huggingface` by default, or `modelscope`). The existing `inference.model_path`
holds the Storage path or model repository identifier.

The shared tool validates configuration and platform node contracts, generates
the two/three-node DSL, publishes through the workspace port, and stages the
existing evaluator/report assets. Harness adapters contain no source selection
or topology construction; the Pi prompt only routes to the evaluation Skill.

Product Runtime keeps execution identity, input versions and persistence. Its
internal preparation port returns both effective workflow and frozen DSL so the
canvas matches the submitted snapshot. Clone/Download identifiers are excluded
from existing-model snapshot selection. Dependency direction stays tools →
generic execution protocol, adapter → tools/runtime; Runtime imports neither.

No HTTP, ProductCommand, ProductEvent or ConversationSnapshot fields change.
Existing runs retain their definitions and fingerprints; reading an omitted
source uses Storage without inserting new defaults into persisted configuration.
Runtime scripts are unchanged, preserving existing evaluator fingerprints and
report/annotation protocols. Resume reuses frozen configuration and predictions;
mutable remote model names do not gain revision pinning.

Regression coverage includes source validation, bound ports, local/Sandbox
canvas writes, selected-model conflicts, versioned final DSL, old/resumed runs,
platform validation failures and uncertain submissions. The multi-turn behavior
case replays baseline Benchmark, GT comparison, post-training evaluation and
ordinary training without submitting real platform tasks.
