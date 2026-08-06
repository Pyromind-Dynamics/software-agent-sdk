"""Mock 冒烟测试:不访问真实网络,验证 wandb_analysis.py 全命令可跑通。

同进程运行:patch 掉 wandb SDK 与平台客户端,直接调用 CLI main()。
"""

import json
import sys
import types


SCRIPTS = (
    "/Users/zhangpeng/PycharmProjects/software-agent-sdk/"
    ".agents/skills/training-analysis/scripts"
)
sys.path.insert(0, SCRIPTS)

import data_sources.wandb as ds_wandb  # noqa: E402
import train_analysis  # noqa: E402


CREDS = "/tmp/smoke_creds.json"
REPORT = "/tmp/smoke_report.md"


class FakeRun:
    def __init__(self, run_id, display_name=None, config=None, summary=None):
        self.id = run_id
        self.display_name = display_name or f"run-{run_id}"
        self.name = self.display_name
        self.state = "finished"
        self.config = config or {
            "learning_rate": 3e-5,
            "job_type": "sft",
            "num_epochs": 3,
            "batch_size": 8,
        }
        self.summary = summary or {"loss": 1.2, "val_loss": 1.5}
        self.summary_metrics = dict(self.summary)
        self.lastHistoryStep = 2
        self._history = [
            {"_step": 0, "loss": 2.0, "val_loss": 2.2},
            {"_step": 1, "loss": 1.5, "val_loss": 1.8},
            {"_step": 2, "loss": 1.2, "val_loss": 1.5},
        ]

    def scan_history(self, keys=None, page_size=None):  # noqa: ARG002
        # 真实 wandb scan_history 无论传何 keys 都附带 _step
        for row in self._history:
            selected = {k: row[k] for k in (keys or list(row)) if k in row}
            selected["_step"] = row["_step"]
            yield selected


class FakeApi:
    def __init__(self, _api_key=None, _timeout=None):
        self._runs = {
            "acme/sft-proj/run1": FakeRun("run1"),
            "acme/sft-proj/run2": FakeRun(
                "run2",
                display_name="run2-dup",
                config={"learning_rate": 1e-5, "job_type": "sft"},
                summary={"loss": 1.8, "val_loss": 2.0},
            ),
        }

    def run(self, path):
        return self._runs[path]

    def runs(  # noqa: PLR0913
        self,
        project_path,  # noqa: ARG002
        filters=None,  # noqa: ARG002
        order=None,  # noqa: ARG002
        per_page=None,  # noqa: ARG002
    ):
        return [r for r in self._runs.values()]

    def viewer(self):
        return types.SimpleNamespace(username="acme")


NODES = [
    {
        "node_code": "n1",
        "data": {
            "nodeType": "TrainModelNode",
            "config": {
                "wandb_api_key": "test-key-123",
                "wandb_project": "sft-proj",
                "wandb_entity": "acme",
            },
        },
    },
]


class FakePlatformClient:
    def __init__(self, base_url="", cookie="", cluster="", authorization=""):
        pass

    def task_workflow_result(self, _task_id):
        return {"nodes": NODES}

    def node_log(self, _node_id, _task_id):
        return (
            "wandb: setting up run abc123\n"
            "View run at https://wandb.ai/acme/sft-proj/runs/abc123\n"
        )

    def node_output(self, _node_code, _task_id):
        return {}


def capture(args):
    """同进程执行 CLI,捕获 stdout;argparse --help 走 SystemExit 视为成功。"""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            code = train_analysis.main(args)
    except SystemExit as exc:  # argparse --help 等
        code = int(exc.code or 0)
    assert code == 0, f"exit={code} args={args}"
    return buf.getvalue()


def main():
    # 替换 SDK 连接:不访问真实网络,注入 FakeApi
    def fake_connect(self, creds: dict[str, str]) -> None:
        self._api = FakeApi(creds.get("WANDB_API_KEY") or "test-key")

    ds_wandb.WandbDataSource.connect = fake_connect
    # 替换平台客户端
    train_analysis.PlatformClient = FakePlatformClient

    print("== 1. --help ==")
    out = capture(["--help"])
    for cmd in (
        "resolve-target",
        "probe",
        "analyze-run",
        "compare-runs",
        "project-summary",
        "report",
    ):
        assert cmd in out, f"--help 缺少 {cmd}"
    print("ok")

    print("== 2. resolve-target ==")
    out = capture(
        ["--data-source", "wandb", "resolve-target", "T123", "--creds-out", CREDS]
    )
    result = json.loads(out)
    assert result["data_source"] == "wandb"
    assert result["run_id"] == "abc123"
    assert result["entity"] == "acme"
    assert result["project"] == "sft-proj"
    assert result["creds_file"] == CREDS
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("== 3. probe ==")
    out = capture(["--creds-file", CREDS, "probe", "acme/sft-proj"])
    result = json.loads(out)
    assert result["sample_metric_keys"] == ["loss", "val_loss"]
    assert result["has_step_history"] is True
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("== 4. analyze-run ==")
    out = capture(["--creds-file", CREDS, "analyze-run", "acme/sft-proj", "run1"])
    result = json.loads(out)
    assert result["metric"] == "loss"
    assert result["metric_stats"]["final"] == 1.2
    assert result["diagnostics"]["has_nan"] is False
    assert result["steps"] == [0, 2]
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("== 5. compare-runs ==")
    out = capture(
        ["--creds-file", CREDS, "compare-runs", "acme/sft-proj", "run1", "run2"]
    )
    result = json.loads(out)
    assert "learning_rate" in result["config_diff"]
    assert result["run_a_metric"] == "loss"
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("== 6. project-summary ==")
    out = capture(
        [
            "--creds-file",
            CREDS,
            "project-summary",
            "acme/sft-proj",
            "--axis",
            "job_type",
        ]
    )
    result = json.loads(out)
    assert result["total_runs"] == 2
    assert result["buckets"]["sft"]["count"] == 2
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("== 7. report ==")
    out = capture(
        ["--creds-file", CREDS, "report", "acme/sft-proj", "run1", "--out", REPORT]
    )
    assert "report written" in out
    report_text = open(REPORT, encoding="utf-8").read()
    assert "训练分析报告" in report_text
    assert "数据来源: wandb" in report_text
    print(report_text[:400])

    # 兼容性:creds 缺 data_source 字段时默认 wandb
    print("== 8. 旧 creds 兼容(无 data_source 字段)==")
    legacy = "/tmp/smoke_creds_legacy.json"
    with open(legacy, "w", encoding="utf-8") as f:
        json.dump({"WANDB_API_KEY": "test-key-123"}, f)
    out = capture(["--creds-file", legacy, "probe", "acme/sft-proj"])
    result = json.loads(out)
    assert result["sample_metric_keys"] == ["loss", "val_loss"]
    print("ok")

    # 数据源自动探测失败时显式 --data-source 可指定
    print("== 9. --data-source 显式指定 ==")
    out = capture(
        [
            "--creds-file",
            CREDS,
            "--data-source",
            "wandb",
            "analyze-run",
            "acme/sft-proj",
            "run1",
        ]
    )
    result = json.loads(out)
    assert result["metric"] == "loss"
    print("ok")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
