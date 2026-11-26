"""PCB Dataset stage input and output contracts, without a model or network."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".agents/skills/data-processing/scripts/preparation/pcb_dataset_inputs.py"
)
FILES = ("defect.jpg", "diff.jpg", "gt.jpg")
ROLES = ("待检实拍图", "差分图", "同位置参考图")


def _create_triplet(root: Path, identity: str) -> list[Path]:
    directory = root / identity
    directory.mkdir(parents=True)
    images = [directory / name for name in FILES]
    for image in images:
        image.write_bytes(f"image-fixture:{identity}:{image.name}".encode())
    return images


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_manifest_groups_exact_verified_triplets_in_role_order(tmp_path: Path) -> None:
    root = tmp_path / "images"
    _create_triplet(root, "0012")
    _create_triplet(root, "0007")
    rules = tmp_path / "rules.txt"
    rules.write_text("待检图上标注框", encoding="utf-8")
    manifest = tmp_path / "input.jsonl"

    result = _run("manifest", root, manifest, "--inspection-rules-file", rules)

    assert result.returncode == 0, result.stderr
    assert _read_jsonl(manifest) == [
        {
            "id": identity,
            "images": [f"images/{identity}/{filename}" for filename in FILES],
            "image_labels": list(ROLES),
            "inspection_rules": "待检图上标注框",
        }
        for identity in ("0007", "0012")
    ]


@pytest.mark.parametrize("problem", ["missing", "extra", "symlink"])
def test_manifest_rejects_incomplete_or_ambiguous_groups(
    tmp_path: Path, problem: str
) -> None:
    root = tmp_path / "images"
    images = _create_triplet(root, "one")
    if problem == "missing":
        images[2].unlink()
    elif problem == "extra":
        (root / "one" / "unmatched.png").write_bytes(b"other image")
    else:
        elsewhere = tmp_path / "elsewhere.jpg"
        elsewhere.write_bytes(b"unrelated image")
        images[2].unlink()
        images[2].symlink_to(elsewhere)
    manifest = tmp_path / "manifest.jsonl"

    result = _run("manifest", root, manifest)

    assert result.returncode == 1
    assert "one" in result.stderr
    assert not manifest.exists()


def _tracer_rows(storage: Path) -> list[dict]:
    rows = []
    for identity, label in zip("01234", ("真点", "假点", "假点", "真点", "假点")):
        images = _create_triplet(storage / "pictures", identity)
        # Conference/Control and platform jobs see different mount prefixes.
        mount = "/workspace" if int(identity) % 2 else "/target-workspace"
        rows.append(
            {
                "id": identity,
                "images": [f"{mount}/{path.relative_to(storage)}" for path in images],
                "system_prompt": "PCB AVI review",
                "user_prompt": f"inspect record {identity}",
                "gt": json.dumps({"result": label, "boxes": []}, ensure_ascii=False),
            }
        )
    return rows


def test_manifest_stages_images_outside_the_source_and_audits_raw_paths(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "storage"
    rows = _tracer_rows(storage)
    manifest = storage / "runs/001/incoming/input.jsonl"
    manifest.parent.mkdir(parents=True)
    image_root = storage / "pictures"

    original = _run("manifest", image_root, manifest)
    assert original.returncode == 1
    assert not manifest.exists()

    staged = _run(
        "manifest", image_root, manifest, "--stage-images", "--storage-root", storage
    )

    assert staged.returncode == 0, staged.stderr
    labels = []
    for entry, row in zip(_read_jsonl(manifest), rows, strict=True):
        identity = row["id"]
        assert entry["id"] == identity
        assert entry["images"] == [f"images/{identity}/{name}" for name in FILES]
        assert entry["original_source_images"] == [
            f"/workspace/pictures/{identity}/{filename}" for filename in FILES
        ]
        for ref, filename in zip(entry["images"], FILES, strict=True):
            copied = manifest.parent / ref
            assert not copied.is_symlink()
            original_image = image_root / identity / filename
            assert copied.read_bytes() == original_image.read_bytes()
        labels.append(
            {
                "id": identity,
                "source_images": dict(zip(ROLES, entry["images"], strict=True)),
                "gt": row["gt"],
                "confirmed": True,
            }
        )

    labels_file = manifest.parent / "reviewed.jsonl"
    _write_jsonl(labels_file, labels)
    prompt = manifest.parent / "prompt.txt"
    prompt.write_text("PCB AVI review", encoding="utf-8")
    output = manifest.parent.parent / "dataset"
    result = _run(
        "messages",
        labels_file,
        output,
        "--source-manifest",
        manifest,
        "--storage-root",
        storage,
        "--system-prompt-file",
        prompt,
        "--user-prompt-file",
        prompt,
    )

    assert result.returncode == 0, result.stderr
    audit = _read_jsonl(output / "source_manifest.jsonl")
    for record, entry in zip(audit, _read_jsonl(manifest), strict=True):
        assert record["source_images"] == dict(
            zip(ROLES, entry["original_source_images"], strict=True)
        )


@pytest.mark.parametrize("collision", ["input.jsonl", "images"])
def test_staging_rejects_collisions_without_overwriting(
    tmp_path: Path, collision: str
) -> None:
    image_root = tmp_path / "raw"
    _create_triplet(image_root, "one")
    run = tmp_path / "run"
    run.mkdir()
    existing = run / collision
    existing.write_text("keep me", encoding="utf-8")

    result = _run(
        "manifest",
        image_root,
        run / "input.jsonl",
        "--stage-images",
        "--storage-root",
        tmp_path,
    )

    assert result.returncode == 1
    assert list(run.iterdir()) == [existing]
    assert existing.read_text(encoding="utf-8") == "keep me"


def test_tracer_batch_uses_confirmed_gt_and_holds_out_only_batch_rows(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "storage"
    rows = _tracer_rows(storage)
    batch = storage / "project-batches/em-1/4"
    batch.mkdir(parents=True)
    _write_jsonl(batch / "train.jsonl", rows)
    # This sibling belongs to business Eval and is deliberately not readable.
    (batch / "val.jsonl").write_text("fixed holdout, not for Train")
    output = storage / "runs/001/dataset"

    result = _run("messages", batch, output, "--storage-root", storage)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"train": 3, "val": 2}
    audit = _read_jsonl(output / "source_manifest.jsonl")
    assert [row["id"] for row in audit] == [row["id"] for row in rows]
    for split in ("train", "val"):
        contents = _read_jsonl(output / f"{split}_messages.jsonl")
        expected = [row for row in audit if row["split"] == split]
        for row, source in zip(contents, expected, strict=True):
            assert set(row) == {"messages"}
            system, user, assistant = row["messages"]
            original = next(row for row in rows if row["id"] == source["id"])
            assert system == {
                "role": "system",
                "content": [{"type": "text", "value": original["system_prompt"]}],
            }
            assert assistant == {
                "role": "assistant",
                "content": [{"type": "text", "value": original["gt"]}],
            }
            assert user["content"][-1] == {
                "type": "text",
                "value": original["user_prompt"],
            }
            for item, role in zip(user["content"][:-1], ROLES, strict=True):
                assert item == {"type": "image_url", "value": source["images"][role]}
                original_uri = source["source_images"][role]
                original_image = storage / Path(original_uri).relative_to(
                    Path(original_uri).parents[-2]
                )
                assert (
                    output / item["value"]
                ).read_bytes() == original_image.read_bytes()


@pytest.mark.parametrize("problem", ["mixed-triplet", "duplicate-id", "unconfirmed"])
def test_messages_fails_before_publishing_incorrect_labels_or_images(
    tmp_path: Path, problem: str
) -> None:
    storage = tmp_path / "storage"
    rows = _tracer_rows(storage)
    if problem == "mixed-triplet":
        rows[0]["images"][1] = rows[1]["images"][1]
    elif problem == "duplicate-id":
        rows[0]["id"] = rows[1]["id"]
    else:
        rows[0]["confirmed"] = False
    source = storage / "train.jsonl"
    output = tmp_path / "dataset"
    _write_jsonl(source, rows)

    result = _run("messages", source, output, "--storage-root", storage)

    assert result.returncode == 1
    assert not output.exists()


def _prelabel_fixture(storage: Path) -> tuple[Path, list[dict], list[str | Path]]:
    manifest = storage / "source.jsonl"
    rows = _tracer_rows(storage)
    _write_jsonl(
        manifest,
        [
            {
                "id": row["id"],
                "images": [f"pictures/{row['id']}/{name}" for name in FILES],
                "image_labels": list(ROLES),
                "system_prompt": row["system_prompt"],
                "user_prompt": row["user_prompt"],
            }
            for row in rows
        ],
    )
    labels = [
        {
            "id": row["id"],
            "source_images": dict(zip(ROLES, row["images"], strict=True)),
            "result": json.loads(row["gt"])["result"],
            "category": "示例标签",
            "confidence": 0.8,
            "boxes": [[101, 102, 201, 202]],
            "unknown_model_field": "must not be promoted",
        }
        for row in rows
    ]
    return (
        storage / "prelabel.jsonl",
        labels,
        [
            "--source-manifest",
            manifest,
            "--storage-root",
            storage,
        ],
    )


def test_prelabels_require_explicit_approval_and_keep_label_provenance(
    tmp_path: Path,
) -> None:
    source, labels, cli = _prelabel_fixture(tmp_path / "storage")
    actual_gt = '{"result":"真点","reason":"已人工复核"}'
    labels[0].update(confirmed=True, gt=actual_gt)
    _write_jsonl(source, labels)
    output = tmp_path / "dataset"

    blocked = _run("messages", source, output, *cli)

    assert blocked.returncode == 1
    assert "needs confirmation" in blocked.stderr
    assert not output.exists()

    approved = _run("messages", source, output, *cli, "--allow-pseudo-labels")

    assert approved.returncode == 0, approved.stderr
    audit = _read_jsonl(output / "source_manifest.jsonl")
    assert audit[0]["label_source"] == "human_confirmed"
    assert {row["label_source"] for row in audit[1:]} == {"approved_pseudo_label"}
    answers = [
        row["messages"][2]["content"][0]["value"]
        for split in ("train", "val")
        for row in _read_jsonl(output / f"{split}_messages.jsonl")
    ]
    assert actual_gt in answers
    assert all("unknown_model_field" not in answer for answer in answers)


@pytest.mark.parametrize("problem", ["image-mismatch", "missing-id", "bad-box"])
def test_prelabel_join_rejects_inconsistent_model_output(
    tmp_path: Path, problem: str
) -> None:
    source, labels, cli = _prelabel_fixture(tmp_path / "storage")
    if problem == "image-mismatch":
        labels[0]["source_images"][ROLES[2]] = labels[1]["source_images"][ROLES[2]]
    elif problem == "missing-id":
        labels.pop()
    else:
        labels[0]["boxes"] = [[0, 0, 1001, 1001]]
    _write_jsonl(source, labels)
    output = tmp_path / "dataset"

    result = _run("messages", source, output, *cli, "--allow-pseudo-labels")

    assert result.returncode == 1
    assert not output.exists()
