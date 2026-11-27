"""Read-only CPU probe shipped as source to the Project's Storage workspace."""

import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path


MAX_SPLIT_BYTES = 8 * 1024 * 1024


def main() -> None:
    from datasets import Dataset

    output = Path(sys.argv[1])
    proof = json.load(sys.stdin)
    for split in ("train", "val"):
        if not proof[f"{split}_rows"]:
            continue
        path = output / f"{split}_messages.jsonl"
        if not path.is_file() or path.resolve(strict=True) != path.absolute():
            sys.exit("reader_probe_source_changed")
        with path.open("rb") as stream:
            raw = stream.read(MAX_SPLIT_BYTES + 1)
        if len(raw) > MAX_SPLIT_BYTES:
            sys.exit("reader_probe_sample_too_large")
        if hashlib.sha256(raw).hexdigest() != proof[f"{split}_sha256"]:
            sys.exit("reader_probe_source_changed")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        try:
            loaded = Dataset.from_list(rows)
        except Exception:
            sys.exit(f"reader_probe_schema_{split}")
        if loaded.num_rows != proof[f"{split}_rows"]:
            sys.exit("reader_probe_source_changed")
    proof["train_reader_probe"] = {
        "method": "datasets.Dataset.from_list",
        "datasets_version": version("datasets"),
        "pyarrow_version": version("pyarrow"),
    }
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
