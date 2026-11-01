"""Local acceptance client: .env credentials become ordinary request headers.

Never imports credentials into the server's global auth configuration or sends
headers as model content. Run from the workspace with uv run python scripts/...
"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
API = "/api/v2/echomind/conversations"


def settings(path):
    values = dotenv_values(path, interpolate=False)
    for name in ("PYROMIND_TEST_COOKIE", "PYROMIND_TEST_X_CLUSTER"):
        if not values.get(name):
            raise ValueError(f"Fill {name} in the local test environment file")
    headers = {
        "cookie": values["PYROMIND_TEST_COOKIE"],
        "x-cluster": values["PYROMIND_TEST_X_CLUSTER"],
    }
    if values.get("PYROMIND_TEST_USER_ID"):
        headers["x-pyromind-debug-user-id"] = values["PYROMIND_TEST_USER_ID"]
    if values.get("PYROMIND_TEST_SESSION_API_KEY"):
        headers["x-session-api-key"] = values["PYROMIND_TEST_SESSION_API_KEY"]
    return values, headers


async def request(client, method, path, **kwargs):
    response = await client.request(method, path, **kwargs)
    if response.is_error:
        # Error responses may include application trace data; never dump them
        # with auth headers into terminal/model/transcript evidence.
        raise ValueError(
            f"HTTP {response.status_code} for {method} {path}; inspect server logs"
        )
    return response.json()


async def run(args):
    values, headers = settings(args.env_file)
    base = values.get("PYROMIND_TEST_AGENT_URL") or "http://127.0.0.1:8000"
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=120) as client:
        if args.action == "probe":
            conversations = await request(client, "GET", API)
            print(
                json.dumps(
                    {
                        "server_reachable": True,
                        "visible_conversations": len(conversations),
                        "cookie_configured": True,
                        "cluster_configured": True,
                    }
                )
            )
            return
        if args.action == "run":
            llm = {}
            for key, name in (
                ("model", "PYROMIND_TEST_LLM_MODEL"),
                ("base_url", "PYROMIND_TEST_LLM_BASE_URL"),
                ("api_key", "PYROMIND_TEST_LLM_API_KEY"),
            ):
                if values.get(name):
                    llm[key] = values[name]
            prompt = args.prompt
            if prompt is None:
                model, dataset = (
                    values.get("PYROMIND_TEST_MODEL_PATH"),
                    values.get("PYROMIND_TEST_DATA_PATH"),
                )
                if not model or not dataset:
                    raise ValueError(
                        "Provide model/data Storage paths or an explicit --prompt"
                    )
                count = int(values.get("PYROMIND_TEST_SAMPLE_LIMIT") or 4)
                if count < 1 or count > 16:
                    raise ValueError("Local acceptance supports 1–16 samples")
                prompt = (
                    "请实际完成一次小样 Init baseline，仅评测、不训练不部署。"
                    f"基模 Storage 路径：{model}；数据：{dataset}。"
                    f"最多 {count} 个样本，"
                    f"inference_evaluation_config 的 limit={count}。"
                    "Manager 只委派 Eval；Eval 按 inference-evaluation Skill 自主探查、"
                    "选评分口径并调用原 df_submit_evaluation。"
                    "无需确认 Graph 或逐阶段确认。"
                    "缺少无法探查的关键输入才向我提问。最终登记真实指标、覆盖率和报告。"
                )
            created = await request(client, "POST", API, json={"llm": llm})
            identity = created["conversation_id"]
            command_id = uuid4().hex
            print(
                json.dumps({"conversation_id": identity, "command_id": command_id}),
                flush=True,
            )
            await request(
                client,
                "POST",
                f"{API}/{identity}/commands",
                json={
                    "type": "user_message",
                    "command_id": command_id,
                    "content": [{"type": "text", "text": prompt}],
                },
            )
        else:
            identity = args.conversation
            if not identity:
                raise ValueError(
                    "--conversation is required to watch/re-authorize an existing run"
                )
        async with asyncio.timeout(args.timeout):
            previous = None
            while True:
                snapshot = await request(client, "GET", f"{API}/{identity}/snapshot")
                executions = snapshot.get("experiment_executions") or []
                summary = [
                    {
                        key: item.get(key)
                        for key in (
                            "run_id",
                            "task_name",
                            "state",
                            "report_state",
                            "platform_task_id",
                            "metrics",
                            "report_uri",
                            "waiting_reason",
                        )
                    }
                    for item in executions
                ]
                encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True)
                if encoded != previous:
                    print(encoded, flush=True)
                    previous = encoded
                if executions and all(
                    item["state"]
                    in {
                        "succeeded",
                        "failed",
                        "cancelled",
                        "abandoned",
                        "prepared",
                        "needs_input",
                        "submission_uncertain",
                    }
                    for item in executions
                ):
                    if any(
                        item["state"]
                        in {"needs_input", "submission_uncertain", "abandoned"}
                        for item in executions
                    ):
                        raise ValueError(
                            "Execution requires input or submission reconciliation; "
                            "inspect this conversation"
                        )
                    if all(
                        item["report_state"] in {"ready", "failed"}
                        for item in executions
                    ):
                        if all(
                            item["state"] == "succeeded"
                            and item["report_state"] == "ready"
                            and item.get("metrics")
                            for item in executions
                        ):
                            queue = snapshot.get("experiment_queue") or []
                            if not queue or any(
                                item["state"] not in {"completed", "cancelled"}
                                for item in queue
                            ):
                                await asyncio.sleep(3)
                                continue
                            for item in executions:
                                await request(
                                    client,
                                    "GET",
                                    f"{API}/{identity}/experiment-executions/{item['run_id']}/artifacts/agent_report",
                                )
                            print(
                                json.dumps(
                                    {
                                        "baseline_artifacts_verified": True,
                                        "conversation_id": identity,
                                    }
                                ),
                                flush=True,
                            )
                            return
                        raise ValueError(
                            "Baseline/report failed; "
                            "GPU inference was not automatically repeated"
                        )
                await asyncio.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "run", "watch"))
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.test.local")
    parser.add_argument("--conversation")
    parser.add_argument("--prompt")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (ValueError, httpx.HTTPError, TimeoutError) as exc:
        print(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
