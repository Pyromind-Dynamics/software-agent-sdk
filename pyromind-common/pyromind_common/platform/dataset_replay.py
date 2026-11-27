"""Server-owned read-only Dataset verification on a mounted Storage volume."""

from dataclasses import dataclass
from pathlib import Path


VERIFIER = (
    Path(__file__).resolve().parents[3]
    / ".agents/skills/data-processing/scripts/preparation/verify_training_splits.py"
)
MAX_DECODE_IMAGES = 64
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_DECODE_BYTES = 12 * 1024 * 1024


@dataclass(frozen=True)
class DatasetReplayPaths:
    """Canonical Storage-relative paths selected by the replay compiler."""

    approved_source: str
    source_batch: str
    output_dir: str
    fixed_validation: str | None = None

    def verification_command(self, mount_root: str) -> list[str]:
        command = [
            "sh",
            "-c",
            'probe="$1"; output="$2"; shift 2; '
            "for py in /workspace/.conda/bin/python "
            "/opt/pyromind/dataflow-venv/bin/python "
            "/home/openhands/dataflow-venv/bin/python "
            "/tmp/pyromind-dataflow-venv-1.0.13-cpu/bin/python python3; do "
            '"$py" -I -c "from datasets import Dataset" >/dev/null 2>&1 || continue; '
            'receipt=$("$py" "$@") || exit $?; '
            'printf %s "$receipt" | "$py" -I -c "$probe" "$output"; exit $?; '
            "done; echo reader_probe_environment_missing >&2; exit 1",
            "verify-dataset",
            VERIFIER.with_name("probe_sft_train_reader.py").read_text(),
            mount_root + self.output_dir,
            "-I",
            "-c",
            VERIFIER.read_text(),
            "--read-only",
            "--defer-image-decode",
            str(MAX_DECODE_IMAGES),
            str(MAX_IMAGE_BYTES),
            str(MAX_DECODE_BYTES),
            "--mount-root",
            mount_root,
            "--approved-source",
            mount_root + self.approved_source,
            "--source-batch",
            mount_root + self.source_batch,
            "--output-dir",
            mount_root + self.output_dir,
        ]
        if self.fixed_validation:
            command.extend(["--fixed-validation", mount_root + self.fixed_validation])
        return command
