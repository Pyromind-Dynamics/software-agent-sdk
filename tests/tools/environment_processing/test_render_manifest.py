"""Tests for the template-driven manifest renderer (render_manifest.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".agents"
    / "skills"
    / "environment-data-processing"
    / "scripts"
    / "render_manifest.py"
)


def _load() -> Any:
    cached = sys.modules.get("render_manifest")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("render_manifest", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_manifest"] = module
    spec.loader.exec_module(module)
    return module


render = _load()


def _parquet_rows(path: Path, rows: int) -> Path:
    df = pd.DataFrame(
        {
            "task_id": [f"t-{i:03d}" for i in range(rows)],
            "description": [f"Solve problem {i}" for i in range(rows)],
            "test_final_state": [f"# final {i}\nassert True" for i in range(rows)],
            "env_config_image": [f"img:{i}" for i in range(rows)],
        }
    )
    df.to_parquet(path)
    return path


def _parquet_tmp(tmp_path: Path, rows: int = 6, name: str = "train.parquet") -> Path:
    return _parquet_rows(tmp_path / name, rows)


def _template(tmp_path: Path) -> Path:
    t = tmp_path / "template.json"
    t.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "tmax-render",
                "fields": {
                    "task_id": "task_id",
                    "image": "env_config_image",
                    "workdir": {"fixed": "/home/user"},
                    "prompt": "description",
                    "test_sh": {
                        "kind": "pytest_wrapper",
                        "source_field": "test_final_state",
                        "target_path": "/workspace/test_final_state.py",
                    },
                },
                "shard_size": 4,
            }
        )
    )
    return t


def test_render_record_maps_template_fields() -> None:
    row = {
        "task_id": "t-1",
        "description": "Solve A",
        "test_final_state": "# final\nassert True",
        "env_config_image": "img:1",
    }
    fields = {
        "task_id": "task_id",
        "image": "env_config_image",
        "workdir": {"fixed": "/home/user"},
        "prompt": "description",
        "test_sh": {
            "kind": "pytest_wrapper",
            "source_field": "test_final_state",
            "target_path": "/workspace/test_final_state.py",
        },
    }
    rec = render.render_record(row, fields, 0)
    assert rec == {
        "task_id": "t-1",
        "image": "img:1",
        "workdir": "/home/user",
        "prompt": "Solve A",
        "test_sh": (
            "#!/bin/bash\n"
            "cat > /workspace/test_final_state.py <<'PYEOF'\n"
            "# final\nassert True\n"
            "PYEOF\n"
            "python3 -m pytest /workspace/test_final_state.py -q\n"
            "rc=$?\n"
            'if [ "$rc" -eq 0 ]; then\n'
            '  echo "1.0" > /logs/verifier/reward.txt\n'
            "else\n"
            '  echo "0.0" > /logs/verifier/reward.txt\n'
            "fi\n"
            "exit 0\n"
        ),
    }


def test_render_missing_column_loud(tmp_path: Path) -> None:
    parquet = _parquet_tmp(tmp_path)
    template = json.loads(_template(tmp_path).read_text())
    template["fields"]["image"] = "no_such_column"
    with pytest.raises(ValueError, match="missing column"):
        render.render_product(str(parquet), template, local_out=str(tmp_path / "out"))


def test_render_product_shards_and_shards_json(tmp_path: Path) -> None:
    parquet = _parquet_tmp(tmp_path, rows=6)
    out = tmp_path / "out"
    shards, rendered = render.render_product(
        str(parquet),
        json.loads(_template(tmp_path).read_text()),
        shard_size=4,
        local_out=str(out),
    )
    assert rendered == 6
    assert len(shards) == 2  # 4 + 2
    assert shards[0].endswith("batch-001/manifest.jsonl")
    assert shards[1].endswith("batch-002/manifest.jsonl")
    first = json.loads(
        (out / "batch-001" / "manifest.jsonl").read_text().splitlines()[0]
    )
    assert first["task_id"] == "t-000"
    assert first["image"] == "img:0"


def test_to_storage_path_strips_mount_prefix() -> None:
    assert (
        render._to_storage_path("/target-workspace/edp/r/batch-001/manifest.jsonl")
        == "/edp/r/batch-001/manifest.jsonl"
    )
    assert render._to_storage_path("/edp/r/batch-001/manifest.jsonl") == (
        "/edp/r/batch-001/manifest.jsonl"
    )


def test_render_record_resolves_join_field() -> None:
    row = {
        "task_id": "t-1",
        "description": "Solve A",
        "test_final_state": "# final\nassert True",
    }
    fields = {
        "task_id": "task_id",
        "image": {
            "join": {"source": "images.jsonl", "on": "task_id", "column": "image"}
        },
        "workdir": {"fixed": "/home/user"},
        "prompt": "description",
        "test_sh": {
            "kind": "pytest_wrapper",
            "source_field": "test_final_state",
            "target_path": "/workspace/test_final_state.py",
        },
    }
    rec = render.render_record(row, fields, 0, join_indexes={"image": {"t-1": "img:9"}})
    assert rec["image"] == "img:9"


def test_build_join_indexes_reads_jsonl_and_csv(tmp_path: Path) -> None:
    join_jsonl = tmp_path / "images.jsonl"
    join_jsonl.write_text(
        json.dumps({"task_id": "t-1", "image": "img:1"})
        + "\n"
        + json.dumps({"task_id": "t-2", "image": "img:2"})
        + "\n"
    )
    join_csv = tmp_path / "images.csv"
    join_csv.write_text("task_id,image\nt-3,img:3\n")
    template = {
        "fields": {
            "image": {
                "join": {
                    "source": str(join_jsonl),
                    "on": "task_id",
                    "column": "image",
                }
            },
            "extra": {
                "join": {"source": str(join_csv), "on": "task_id", "column": "image"}
            },
        }
    }
    indexes = render.build_join_indexes(template)
    assert indexes["image"] == {"t-1": "img:1", "t-2": "img:2"}
    assert indexes["extra"] == {"t-3": "img:3"}


def test_render_product_join_miss_goes_to_failures(tmp_path: Path) -> None:
    """Join misses are skipped into render_failures.jsonl, never guessed."""
    parquet = _parquet_tmp(tmp_path, rows=3)
    join = tmp_path / "images.jsonl"
    join.write_text(json.dumps({"task_id": "t-000", "image": "img:0"}) + "\n")
    template = json.loads(_template(tmp_path).read_text())
    template["fields"]["image"] = {
        "join": {"source": str(join), "on": "task_id", "column": "image"}
    }
    out = tmp_path / "out"
    shards, rendered = render.render_product(
        str(parquet),
        template,
        shard_size=10,
        local_out=str(out),
        join_indexes=render.build_join_indexes(template),
    )
    assert rendered == 1
    assert len(shards) == 1
    failures = [
        json.loads(line)
        for line in (out / "render_failures.jsonl").read_text().splitlines()
    ]
    assert {f["task_id"] for f in failures} == {"t-001", "t-002"}
    assert all(f["reason"] == "join miss" for f in failures)
    report = json.loads((out / "report.json").read_text())
    assert report["rendered"] == 1
    assert report["failed"] == 2
    progress = json.loads((out / "progress.json").read_text())
    assert progress["total"] == 3
    assert progress["succeeded"] == 1
    assert progress["failed"] == 2


def test_expand_parquet_files_glob_and_directory(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _parquet_rows(data / "train-0.parquet", rows=2)
    _parquet_rows(data / "train-1.parquet", rows=2)
    (data / "notes.txt").write_text("not parquet\n")

    globbed = render._expand_parquet_files(str(data / "train-*.parquet"))
    assert [Path(f).name for f in globbed] == ["train-0.parquet", "train-1.parquet"]

    listed = render._expand_parquet_files(str(data))
    assert [Path(f).name for f in listed] == ["train-0.parquet", "train-1.parquet"]


def test_render_product_limit(tmp_path: Path) -> None:
    parquet = _parquet_tmp(tmp_path, rows=6)
    shards, rendered = render.render_product(
        str(parquet),
        json.loads(_template(tmp_path).read_text()),
        shard_size=4,
        limit=5,
        local_out=str(tmp_path / "out2"),
    )
    assert rendered == 5
    assert len(shards) == 2  # 4 + 1 remainder


# ---------------------------------------------------------------------------
# message / storage_file specs (open-instruct shaped datasets)
# ---------------------------------------------------------------------------


def _open_instruct_parquet(path: Path, rows: int) -> Path:
    df = pd.DataFrame(
        {
            "task_id": [f"task_{i:06d}" for i in range(rows)],
            "messages": [
                [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": f"Solve problem {i}"},
                    {"role": "assistant", "content": "answer"},
                ]
                for i in range(rows)
            ],
        }
    )
    df.to_parquet(path)
    return path


def test_render_open_instruct_shape_end_to_end(tmp_path: Path) -> None:
    """open-instruct 形态:prompt 在 messages 列,test 资产在逐任务目录。

    这正是预览受限/契约不匹配的场景——render 契约能表达这种形态,agent
    才能只走控制面(模板 + edp_render),不自建沙箱下载数据查 schema。
    """
    parquet = _open_instruct_parquet(tmp_path / "train.parquet", rows=3)
    for i in range(3):
        task_dir = tmp_path / "task-data" / f"task_{i:06d}" / "tests"
        task_dir.mkdir(parents=True)
        (task_dir / "test_final_state.py").write_text(
            f"def test_{i}():\n    assert True\n"
        )
    join = tmp_path / "images.jsonl"
    join.write_text(
        "".join(
            json.dumps({"task_id": f"task_{i:06d}", "image": f"img:{i}"}) + "\n"
            for i in range(2)  # task_000002 has no mapping -> join miss
        )
    )
    template = {
        "fields": {
            "task_id": "task_id",
            "image": {
                "join": {"source": str(join), "on": "task_id", "column": "image"}
            },
            "workdir": {"fixed": "/home/user"},
            "prompt": {
                "kind": "message",
                "source_field": "messages",
                "role": "user",
            },
            "test_sh": {
                "kind": "pytest_wrapper",
                "source": {
                    "kind": "storage_file",
                    "path_template": str(
                        tmp_path
                        / "task-data"
                        / "{task_id}"
                        / "tests"
                        / "test_final_state.py"
                    ),
                },
                "target_path": "/workspace/test_final_state.py",
            },
        },
        "shard_size": 2,
    }
    out = tmp_path / "out"
    shards, rendered = render.render_product(
        str(parquet),
        template,
        shard_size=2,
        local_out=str(out),
        join_indexes=render.build_join_indexes(template),
    )
    assert rendered == 2
    assert len(shards) == 1
    first = json.loads(
        (out / "batch-001" / "manifest.jsonl").read_text().splitlines()[0]
    )
    assert first["task_id"] == "task_000000"
    assert first["prompt"] == "Solve problem 0"
    assert first["image"] == "img:0"
    assert "def test_0():" in first["test_sh"]
    assert "python3 -m pytest /workspace/test_final_state.py" in first["test_sh"]
    failures = [
        json.loads(line)
        for line in (out / "render_failures.jsonl").read_text().splitlines()
    ]
    assert [f["task_id"] for f in failures] == ["task_000002"]
    assert failures[0]["reason"] == "join miss"


def test_render_product_row_level_failures_do_not_abort(tmp_path: Path) -> None:
    """行级数据问题(文件缺失)只跳过该行,不中断整批。"""
    parquet = _open_instruct_parquet(tmp_path / "train.parquet", rows=3)
    task_dir = tmp_path / "task-data" / "task_000000" / "tests"
    task_dir.mkdir(parents=True)
    (task_dir / "test.sh").write_text("#!/bin/bash\necho ok\n")
    template = {
        "fields": {
            "task_id": "task_id",
            "prompt": {"kind": "message", "source_field": "messages", "role": "user"},
            "test_sh": {
                "kind": "storage_file",
                "path_template": str(
                    tmp_path / "task-data" / "{task_id}" / "tests" / "test.sh"
                ),
            },
        },
    }
    out = tmp_path / "out"
    shards, rendered = render.render_product(
        str(parquet), template, shard_size=10, local_out=str(out)
    )
    assert rendered == 1
    failures = [
        json.loads(line)
        for line in (out / "render_failures.jsonl").read_text().splitlines()
    ]
    assert {f["task_id"] for f in failures} == {"task_000001", "task_000002"}
    assert all(f["field"] == "test_sh" for f in failures)
    assert all("missing" in f["reason"] for f in failures)


def test_render_message_spec_index_last_and_missing_role() -> None:
    row = {
        "task_id": "t-1",
        "messages": [
            {"role": "user", "content": "first prompt"},
            {"role": "assistant", "content": "mid answer"},
            {"role": "user", "content": "follow-up"},
        ],
    }
    rec = render.render_record(
        row,
        {
            "task_id": "task_id",
            "prompt": {"kind": "message", "index": "last"},
            "test_sh": {"fixed": "#!/bin/bash\ntrue\n"},
        },
        0,
    )
    assert rec["prompt"] == "follow-up"

    with pytest.raises(render.RenderSkip, match="no message with role"):
        render.render_record(
            row,
            {
                "task_id": "task_id",
                "prompt": {"kind": "message", "role": "tool"},
                "test_sh": {"fixed": "#!/bin/bash\ntrue\n"},
            },
            0,
        )


def test_render_pytest_wrapper_requires_source() -> None:
    with pytest.raises(ValueError, match="pytest_wrapper requires"):
        render.render_record(
            {"task_id": "t-1", "description": "x"},
            {
                "task_id": "task_id",
                "prompt": "description",
                "test_sh": {"kind": "pytest_wrapper", "target_path": "/w/t.py"},
            },
            0,
        )


# ---------------------------------------------------------------------------
# nested struct columns (dotted-path descent)
# ---------------------------------------------------------------------------


def test_render_nested_struct_dotted_paths_end_to_end(tmp_path: Path) -> None:
    """env_config struct 形态:task_id/image 嵌在 struct 列里,点号下钻直接
    渲染——不再需要先另起 pipeline 展平 parquet。"""
    df = pd.DataFrame(
        {
            "messages": [
                [{"role": "user", "content": f"Solve problem {i}"}] for i in range(2)
            ],
            "env_config": [
                {"task_id": f"task_{i:06d}", "image": f"img:{i}"} for i in range(2)
            ],
        }
    )
    parquet = tmp_path / "train.parquet"
    df.to_parquet(parquet)
    template = {
        "fields": {
            "task_id": "env_config.task_id",
            "image": {"field": "env_config.image"},
            "prompt": {
                "kind": "message",
                "source_field": "messages",
                "role": "user",
            },
            "test_sh": {"fixed": "#!/bin/bash\ntrue\n"},
        }
    }
    out = tmp_path / "out"
    shards, rendered = render.render_product(
        str(parquet), template, shard_size=10, local_out=str(out)
    )
    assert rendered == 2
    assert len(shards) == 1
    records = [
        json.loads(line)
        for line in (out / "batch-001" / "manifest.jsonl").read_text().splitlines()
    ]
    assert records[0]["task_id"] == "task_000000"
    assert records[0]["image"] == "img:0"
    assert records[0]["prompt"] == "Solve problem 0"
    assert records[1]["image"] == "img:1"


def test_render_dotted_path_missing_fails_with_column_list() -> None:
    row = {"env_config": {"image": "img:1"}, "messages": []}
    with pytest.raises(ValueError, match="env_config.task_id"):
        render.render_record(
            row,
            {
                "task_id": "env_config.task_id",
                "prompt": {"kind": "message"},
                "test_sh": {"fixed": "#!/bin/bash\ntrue\n"},
            },
            0,
        )


def test_render_product_aborts_on_consecutive_failures(tmp_path: Path) -> None:
    """系统性模板错误(如 path_template 语义错)在几百行内中止,
    不是烧完全量 14601 条才暴露。"""
    parquet = _open_instruct_parquet(tmp_path / "train.parquet", rows=250)
    template = {
        "fields": {
            "task_id": "task_id",
            "prompt": {
                "kind": "message",
                "source_field": "messages",
                "role": "user",
            },
            "test_sh": {
                "kind": "storage_file",
                "path_template": "wrong-root/task-data/{task_id}/tests/test.sh",
            },
        }
    }
    with pytest.raises(RuntimeError, match="systematically"):
        render.render_product(
            str(parquet), template, shard_size=100, local_out=str(tmp_path / "out")
        )


def test_render_scattered_failures_do_not_abort(tmp_path: Path) -> None:
    """散布的行级失败(隔行成功)永不触发快速中止。"""
    parquet = _open_instruct_parquet(tmp_path / "train.parquet", rows=10)
    for i in range(10):
        if i % 2 == 0:  # 偶数行有文件,奇数行缺失
            task_dir = tmp_path / "task-data" / f"task_{i:06d}" / "tests"
            task_dir.mkdir(parents=True)
            (task_dir / "test.sh").write_text("#!/bin/bash\ntrue\n")
    template = {
        "fields": {
            "task_id": "task_id",
            "prompt": {
                "kind": "message",
                "source_field": "messages",
                "role": "user",
            },
            "test_sh": {
                "kind": "storage_file",
                "path_template": str(
                    tmp_path / "task-data" / "{task_id}" / "tests" / "test.sh"
                ),
            },
        }
    }
    shards, rendered = render.render_product(
        str(parquet), template, shard_size=100, local_out=str(tmp_path / "out")
    )
    assert rendered == 5


def test_storage_file_missing_reason_names_mount_prefix(tmp_path: Path) -> None:
    """失败消息自解释:带上尝试过的挂载路径,agent 一眼看出
    path_template 该相对 storage 根而不是数据集目录。"""
    parquet = _open_instruct_parquet(tmp_path / "train.parquet", rows=1)
    template = {
        "fields": {
            "task_id": "task_id",
            "prompt": {
                "kind": "message",
                "source_field": "messages",
                "role": "user",
            },
            "test_sh": {
                "kind": "storage_file",
                "path_template": "task-data/{task_id}/tests/test.sh",
            },
        }
    }
    out = tmp_path / "out"
    render.render_product(str(parquet), template, shard_size=10, local_out=str(out))
    failure = json.loads((out / "render_failures.jsonl").read_text().splitlines()[0])
    assert "storage root" in failure["reason"]
    assert "/target-workspace/task-data/" in failure["reason"]
