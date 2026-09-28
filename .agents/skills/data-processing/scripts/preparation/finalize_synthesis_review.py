"""Merge managed reviews: pipeline.py review_job.json reviewed.jsonl."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from reference_synthesis import finalize_review


def main(job_path: str, output_path: str) -> None:
    job_file = Path(job_path).resolve()
    job = json.loads(job_file.read_text())
    binding_path = (job_file.parent / job["binding_path"]).resolve()
    binding = json.loads(binding_path.read_text())
    reviews_path = (job_file.parent / job["review_output_path"]).resolve()
    outputs = []
    if reviews_path.is_file():
        for line in reviews_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    outputs.append(row)
            except json.JSONDecodeError:
                continue
    output = Path(output_path)
    report = output.with_suffix(".quality.json")
    if output.exists() or report.exists():
        raise ValueError(
            "use new output paths; preserve generation and raw review files"
        )
    rows, summary = finalize_review(binding_path.parent, binding, outputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve() != binding_path.parent:
        copies = {}
        for row in rows:
            for artifact in row["artifacts"]:
                relative = Path(artifact["path"])
                source = (binding_path.parent / relative).resolve()
                target = (output.parent / relative).resolve()
                if (
                    relative.is_absolute()
                    or not source.is_relative_to(binding_path.parent)
                    or not target.is_relative_to(output.parent.resolve())
                    or target.exists()
                ):
                    raise ValueError(
                        "artifact destination must be new and within output"
                    )
                copies[target] = source
        for target, source in copies.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
