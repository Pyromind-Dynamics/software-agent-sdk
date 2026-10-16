"""Reproduce the metadata-only Pi compatibility update; preserve all runtime bytes."""
import gzip
import io
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "vendor"
source = root / "agentgenome-pi-extension-0.6.3.tgz"
target = root / "agentgenome-pi-extension-0.6.3-pi087.tgz"
with tarfile.open(source) as old, target.open("wb") as destination:
    with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as new:
            for member in old.getmembers():
                content = old.extractfile(member) if member.isfile() else None
                if member.name == "package/package.json":
                    metadata = json.load(content)
                    metadata["peerDependencies"]["@earendil-works/pi-coding-agent"] = "0.84.2 || 0.87.0"
                    raw = (json.dumps(metadata, indent=2) + "\n").encode()
                    member.size = len(raw)
                    content = io.BytesIO(raw)
                new.addfile(member, content)
