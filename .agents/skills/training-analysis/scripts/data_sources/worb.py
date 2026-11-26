"""Read training runs from the Shanghai PyroMind W&B service."""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import UUID

import httpx
from analysis_helpers import TRAINING_STEP_KEYS

from pyromind_common.tools.utils.training_source import (
    PYROMIND_WANDB_BASE_URL,
    worb_history_point,
    worb_run_api_url,
    worb_run_lookup_url,
)

from .base import RunData
from .wandb import WandbDataSource


class WorbDataSource(WandbDataSource):
    def connect(self, creds: dict[str, str]) -> None:
        key = creds.get("WANDB_API_KEY", "")
        if not key:
            raise ValueError("PyroMind W&B credential missing from the training task")
        self._auth = httpx.BasicAuth("api", key)

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=PYROMIND_WANDB_BASE_URL,
            auth=self._auth,
            follow_redirects=False,
            timeout=float(os.environ.get("WANDB_API_TIMEOUT", "60")),
        )

    def resolve_entity(self, project: str, fallback: str = "") -> tuple[str, str]:
        if fallback:
            return fallback, project
        with self._client() as client:
            response = client.post("/graphql", json={"query": "{ viewer { entity } }"})
            response.raise_for_status()
            return response.json()["data"]["viewer"]["entity"], project

    @staticmethod
    def _run_url(client: httpx.Client, project_path: str, run_id: str) -> str:
        entity, project = project_path.split("/", maxsplit=1)
        response = client.get(worb_run_lookup_url(entity, project, run_id))
        if response.status_code != 302:
            response.raise_for_status()
            raise ValueError("PyroMind W&B returned no run location")
        return worb_run_api_url(response.headers.get("Location", ""))

    def probe(self, project_path: str, run_id: str | None = None) -> dict[str, Any]:
        with self._client() as client:
            if run_id:
                url = self._run_url(client, project_path, run_id)
                run = client.get(url).raise_for_status().json()
                history_keys = client.get(f"{url}/history/keys").raise_for_status()
                keys = set(history_keys.json() or [])
                runs = [run]
            else:
                entity, project = project_path.split("/", maxsplit=1)
                projects = client.get("/api/projects").raise_for_status().json() or []
                identifier = next(
                    (
                        p["ID"]
                        for p in projects
                        if (p["Entity"], p["Name"]) == (entity, project)
                    ),
                    None,
                )
                if not identifier:
                    raise ValueError("PyroMind W&B project not found")
                url = f"/api/projects/{UUID(identifier)}/runs"
                all_runs = client.get(url).raise_for_status().json() or []
                runs = [run for run in all_runs if run["State"] == "finished"][:3]
                keys = set()
        keys.update(k for run in runs for k in (run.get("Summary") or {}))
        config_keys = {
            k
            for run in runs
            for k in (run.get("Config") or {})
            if not k.startswith("_")
        }
        metric_keys = sorted(k for k in keys if not k.startswith("_"))
        result: dict[str, Any] = {
            "path": project_path,
            "warnings": [] if runs else ["No finished runs found"],
            "sample_metric_count": len(metric_keys),
            "sample_metric_keys": metric_keys[:50],
            "sample_config_keys": sorted(config_keys)[:50],
            "has_step_history": any(run.get("HistoryLineCount", 0) > 0 for run in runs),
        }
        if run_id:
            result["run_summary_keys"] = sorted(runs[0].get("Summary") or {})
            result["run_config_keys"] = sorted(config_keys)
        elif not runs:
            result["run_count_estimate"] = 0
        return result

    def get_run_data(self, project_path: str, run_id: str, keys: list[str]) -> RunData:
        rows: dict[int, dict[str, Any]] = {}
        with self._client() as client:
            url = self._run_url(client, project_path, run_id)
            run = client.get(url).raise_for_status().json()
            if keys:
                fields = dict.fromkeys([*keys, *TRAINING_STEP_KEYS])
                with client.stream(
                    "GET", f"{url}/history", params={"keys": ",".join(fields)}
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.strip():
                            key, index, step, value = worb_history_point(
                                json.loads(line)
                            )
                            if key in fields:
                                row = rows.setdefault(index, {"_step": index})
                                row[key] = value
                                if step != index:
                                    row["step"] = step
        return {
            "run_id": run_id,
            "display_name": run.get("DisplayName") or run_id,
            "state": run.get("State") or "",
            "config": {
                k: v["value"] if isinstance(v, dict) and "value" in v else v
                for k, v in (run.get("Config") or {}).items()
                if not k.startswith("_")
            },
            "summary": run.get("Summary") or {},
            "history": [rows[index] for index in sorted(rows)],
        }
