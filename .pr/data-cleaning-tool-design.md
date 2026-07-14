# Pyromind Data Cleaning Tool Design

## Goal

Add a `run_dataset_cleaning` tool that submits an agent-authored cleaning
script to Pyromind Studio as a one-node `CustomCommandNode` workflow. The
script reads and writes directly in the user's mounted storage, so long-running
jobs do not depend on the agent-server process.

LLM-backed transformation (`llm_map`) is intentionally out of scope for the
first version.

## Responsibilities

The agent:

1. Calls `preview_dataset` to inspect the source.
2. Writes a cleaning script that implements the agreed CLI contract.
3. Uploads the script with `upload_file_to_pyromind`.
4. Calls `run_dataset_cleaning` with the source and script storage paths.

The tool:

1. Validates and normalizes storage paths and resource limits.
2. Allocates a run ID and a fixed output directory for a new run, or validates
   an existing run ID for resume.
3. Builds a fixed one-node xyflow payload. The model never supplies arbitrary
   shell command text.
4. Submits the payload to `POST /std2/studio_api/api/prompt` with the same
   cookie and cluster context as the existing Pyromind storage tools.
5. Persists task association metadata after the platform returns a task ID.
6. Returns the task ID, run ID, output directory, and initial status.

The cleaning script:

1. Implements the CLI contract:

   ```text
   python clean_script.py --input <path> --output <path> --state-dir <dir>
                          [--resume] [--limit N]
   ```

2. Streams records, appends output, atomically checkpoints progress, records
   row-level errors, and writes final statistics.
3. Treats the run directory as its state directory.

## Tool Contract

Action fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `input_path` | string | Source path in user-visible Pyromind storage |
| `script_path` | optional string | Uploaded script for a new run; omitted on resume |
| `limit` | optional positive integer | Maximum source records for this run |
| `resume_run_id` | optional UUID | Resume an existing run when present |
| `cpu` | integer, default 4 | Custom command CPU request |
| `memory` | integer, default 32 | Custom command memory request in GiB |
| `gpu_count` | integer, default 0 | Custom command GPU count |
| `gpu_product` | enum | Studio-supported GPU product |

Observation fields:

| Field | Meaning |
| --- | --- |
| `status` | Initial platform status, normally `Pending` |
| `task_id` | Studio task ID returned by `/prompt` |
| `run_id` | Cleaning run UUID |
| `output_dir` | User-visible storage result directory |
| `resumed` | Whether this submission resumes an existing run |

The output root is tool configuration, not an action field. Its default is:

```text
/agentTest/data_cleaning
```

Each run uses:

```text
/agentTest/data_cleaning/<run_id>/
  clean_script.py
  output.jsonl
  errors.jsonl
  checkpoint.json
  stats.json
```

The new-run command copies the uploaded script to `clean_script.py` in the run
directory before execution. Resume always executes that frozen copy, preventing
an accidental script change from corrupting an existing output.

## Path Mapping

Tool callers use storage paths such as `/datasets/train.jsonl`. The command pod
sees the current user's storage at `/target-workspace`, so the tool maps paths
as follows:

```text
/datasets/train.jsonl
  -> /target-workspace/datasets/train.jsonl
```

Only normalized absolute storage paths are accepted. Traversal segments and
control characters are rejected. Every path inserted into the command is shell
quoted by the tool. Resume reloads the server-created output path from the task
association and verifies that its final segment matches the validated run UUID.

## Workflow Payload

The tool submits current xyflow format:

```json
{
  "id": "<run-id>",
  "name": "agent-data-clean-<short-run-id>",
  "nodes": [
    {
      "id": "1",
      "type": "default",
      "position": {"x": 0, "y": 0},
      "data": {
        "display_name": "Custom Command",
        "nodeType": "CustomCommandNode",
        "config": {
          "command": "<tool-generated-command>",
          "cpu": 4,
          "memory": 32,
          "gpu_count": 0,
          "gpu_product": "NVIDIA-H100-NVL"
        }
      }
    }
  ],
  "edges": []
}
```

Database IDs and timestamps from frontend node definitions are omitted. The
Studio backend resolves the authoritative system node definition by
`nodeType`.

## Task Association

`/prompt` returns a task ID but does not accept the `out_id` used by the generic
workflow API. The tool therefore persists one JSON record per task under a
hidden directory next to the conversation directories:

```text
<conversations-dir>/.pyromind_dataset_cleaning_tasks/<task-id-sha256>.json
```

Each record contains the task ID, conversation ID, run ID, output directory,
input path, script path, limit, resume flag, submission time, and latest known
status. Writes use a temporary file plus `os.replace`.

The dataset-cleaning callback adapter checks this store when no conversation ID
is supplied. A matching cleaning task resolves the conversation and supplies a
cleaning-specific terminal delivery to the generic workflow callback pipeline.
The generic callback remains unaware of task stores and cleaning artifacts; it
only owns status normalization, conversation lookup, deduplication, and delivery.
This keeps association valid across agent-server restarts and shared conversation
storage, without scanning every conversation.

## Submission And Completion Semantics

Submission is asynchronous. The tool returns after `/prompt` accepts the task.
On terminal callback:

1. Resolve the cleaning association by task ID.
2. Update its persisted status.
3. Resume the owning conversation with the run ID and output directory.
4. Tell the agent to preview `<output-dir>/stats.json` and report the result.

Platform `Succeeded` means the pod completed, but the agent should still verify
`stats.json`. A missing final statistics artifact is reported as incomplete.

To resume, the agent invokes the same tool with `resume_run_id` and the original
`input_path`. The tool resolves the persisted run association, rejects unknown
runs or a changed input, uses the frozen script and output directory, and appends
`--resume`; the script owns checkpoint and output consistency.

## Test Strategy

Unit tests cover:

1. New and resume command/workflow construction.
2. Path validation and shell quoting.
3. Cookie and cluster forwarding to the prompt endpoint.
4. Prompt response validation and task association persistence.
5. Callback lookup by task ID without an explicit conversation ID.
6. Pyromind agent registration and parameter wiring.

A live tool test then submits a tiny uploaded JSONL cleaning script with
`limit=3`, waits for the Studio task, and checks the result directory artifacts.
