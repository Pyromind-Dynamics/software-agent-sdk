from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from pyromind_runtime.ports.task_monitor import TaskCheck


class SQLiteTaskMonitorStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS task_checks ("
                "conversation_id TEXT NOT NULL, task_id TEXT NOT NULL, "
                "due_at REAL NOT NULL, failures INTEGER NOT NULL DEFAULT 0, "
                "lease TEXT, PRIMARY KEY(conversation_id, task_id))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS task_checks_due ON task_checks(due_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def register(self, conversation_id: str, task_id: str, due_at: float) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO task_checks "
                "(conversation_id, task_id, due_at) VALUES (?, ?, ?)",
                (conversation_id, task_id, due_at),
            )

    def claim(self, now: float, lease_seconds: float) -> TaskCheck | None:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT conversation_id, task_id, failures FROM task_checks "
                "WHERE due_at <= ? ORDER BY due_at LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            lease = uuid4().hex
            connection.execute(
                "UPDATE task_checks SET due_at = ?, lease = ? "
                "WHERE conversation_id = ? AND task_id = ?",
                (now + lease_seconds, lease, row[0], row[1]),
            )
            return TaskCheck(row[0], row[1], row[2], lease)

    def postpone(self, task: TaskCheck, due_at: float, failures: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE task_checks SET due_at = ?, failures = ?, lease = NULL "
                "WHERE conversation_id = ? AND task_id = ? AND lease = ?",
                (due_at, failures, task.conversation_id, task.task_id, task.lease),
            )

    def remove(self, task: TaskCheck) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM task_checks WHERE "
                "conversation_id = ? AND task_id = ? AND lease = ?",
                (task.conversation_id, task.task_id, task.lease),
            )
