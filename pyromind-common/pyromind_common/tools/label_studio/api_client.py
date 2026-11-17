"""Thin HTTP wrapper over the Label Studio REST API."""

from __future__ import annotations

from typing import Any

import httpx


class LabelStudioAPIError(ValueError):
    """Raised when a Label Studio API call fails."""


class LabelStudioAPIClient:
    """Synchronous client for Label Studio project management endpoints."""

    def __init__(self, *, base_url: str, token: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self._token}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        try:
            response = httpx.request(
                method,
                self._url(path),
                headers=self._headers(),
                json=json_body,
                params=params,
                timeout=timeout or self._timeout,
            )
        except httpx.RequestError as exc:
            raise LabelStudioAPIError(
                f"Label Studio {method} {path} failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            body = response.text[:500]
            raise LabelStudioAPIError(
                f"Label Studio {method} {path} returned "
                f"HTTP {response.status_code}: {body}"
            )
        if not response.content:
            return {}
        return response.json()

    def create_project(self, *, title: str, label_config: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/projects",
            json_body={"title": title, "label_config": label_config},
        )

    def get_project(self, project_id: int) -> dict[str, Any]:
        return self._request("GET", f"/api/projects/{project_id}")

    def find_project_by_title(self, title: str) -> dict[str, Any] | None:
        """Return the project with this exact title, if it already exists.

        Retries adopt an existing project instead of creating a duplicate when a
        previous attempt died after Label Studio had already stored it.
        """
        result = self._request("GET", "/api/projects", params={"title": title})
        projects = result.get("results") if isinstance(result, dict) else result
        if not isinstance(projects, list):
            return None
        for project in projects:
            if isinstance(project, dict) and project.get("title") == title:
                return project
        return None

    def validate_config(self, *, project_id: int, label_config: str) -> None:
        # Label Studio exposes config validation at /api/projects/<pk>/validate/
        # (see label_studio/projects/urls.py). There is no /validate-config route;
        # calling one answers 404 and takes every update_config down with it.
        self._request(
            "POST",
            f"/api/projects/{project_id}/validate/",
            json_body={"label_config": label_config},
        )

    def validate_config_standalone(self, *, label_config: str) -> None:
        """Validate a config before any project exists.

        ``POST /api/projects/validate/`` is Label Studio's project-less validator
        (``LabelConfigValidateAPI``, ``permission_classes = (AllowAny,)``): a valid
        config answers 204, an invalid one answers 400 carrying the reason. Running
        it up front turns "this XML is broken" into an immediate failure instead of
        one that only surfaces after the dataset has been converted and uploaded.
        """
        self._request(
            "POST",
            "/api/projects/validate/",
            json_body={"label_config": label_config},
        )

    def update_project_config(
        self, *, project_id: int, label_config: str
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/api/projects/{project_id}",
            json_body={"label_config": label_config},
        )

    def update_project_description(
        self, *, project_id: int, description: str
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/api/projects/{project_id}",
            json_body={"description": description},
        )

    def import_tasks(
        self, *, project_id: int, tasks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Label Studio's import endpoint expects a bare JSON array of tasks;
        # a dict body is interpreted as a single task whose data is that dict.
        return self._request(
            "POST",
            f"/api/projects/{project_id}/import",
            json_body=tasks,
            timeout=max(self._timeout, 300.0),
        )

    def list_project_tasks(
        self, project_id: int, *, page_size: int = 500
    ) -> list[dict[str, Any]]:
        """Read every task of a project.

        Label Studio paginates this endpoint with a ``total`` count instead of a
        next-page link and caps how large a page it returns, so the loop stops on
        that count rather than on the page size it asked for.
        """
        tasks: list[dict[str, Any]] = []
        page = 1
        while True:
            result = self._request(
                "GET",
                "/api/tasks",
                params={
                    "project": str(project_id),
                    "page": str(page),
                    "page_size": str(page_size),
                },
            )
            if not isinstance(result, dict):
                raise LabelStudioAPIError(
                    f"Label Studio task list returned unexpected payload: {result}"
                )
            page_tasks = result.get("tasks")
            if not isinstance(page_tasks, list):
                raise LabelStudioAPIError(
                    f"Label Studio task list returned unexpected payload: {result}"
                )
            tasks.extend(page_tasks)
            total = result.get("total")
            if not page_tasks or not isinstance(total, int) or len(tasks) >= total:
                return tasks
            page += 1

    def update_task_data(self, *, task_id: int, data: dict[str, Any]) -> dict[str, Any]:
        """Replace one task's data, leaving its annotations untouched.

        Label Studio replaces the whole data object, so callers must send the
        complete dict rather than only the fields they changed.
        """
        return self._request("PATCH", f"/api/tasks/{task_id}", json_body={"data": data})

    def export_annotations(self, project_id: int) -> list[dict[str, Any]]:
        result = self._request(
            "GET",
            f"/api/projects/{project_id}/export",
            params={"exportType": "JSON"},
            timeout=max(self._timeout, 300.0),
        )
        if not isinstance(result, list):
            raise LabelStudioAPIError(
                f"Label Studio export returned unexpected type {type(result).__name__}"
            )
        return result

    def get_annotation_from_names(self, project_id: int) -> set[str]:
        """Collect all ``from_name`` values referenced by existing annotations."""
        names: set[str] = set()
        for task in self.export_annotations(project_id):
            for key in ("annotations", "predictions"):
                for item in task.get(key, []):
                    for result in item.get("result", []):
                        name = result.get("from_name", "")
                        if name:
                            names.add(name)
        return names
