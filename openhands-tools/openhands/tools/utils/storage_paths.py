"""The one place that decides how a Storage path is spelled.

The user's Storage is a single tree that every layer names differently:

    /datasets/a.csv                   Storage root -- what the Storage API and
                                      platform tasks read. Canonical form.
    /target-workspace/datasets/a.csv  the sandbox container's mount point
    public_data/a.csv                 the conversation workspace, which is
                                      itself a directory inside Storage

Only the first form is unambiguous, so a tool normalizes its inputs to it with
:func:`to_storage_path`.

``storage/<path>`` used to be a symlink inside the workspace pointing at the
mount. It was removed, because one file then had two spellings and every path
handed to a platform tool had to drop the prefix by hand. The spelling is
**rejected** now: a caller that still produces it gets an error naming the two
spellings that work, instead of a lookup that fails somewhere else.

Two things this module deliberately does not settle, because they belong to the
caller:

* ``/storage/x`` is a valid *Storage* path but not a resolvable *sandbox* path:
  the sandbox only admits what sits under its mount, so that file is read there
  as ``/target-workspace/storage/x``. Keep :func:`to_storage_path` at the
  Storage-API boundary; the sandbox boundary stays with ``SandboxPathPolicy`` /
  ``resolve_workspace_path``.
* A **bare** relative path (``datasets/x``) has a different root in each layer:
  platform tools read it against the Storage root, the sandbox against the
  conversation workspace. Both are right locally, so the string must not cross
  the boundary -- a tool input has to be explicit.
"""

from __future__ import annotations

import os


REMOVED_ALIAS = "storage"
WORKSPACE_DIRNAME = "public_data"
DEFAULT_MOUNT_PATH = "/target-workspace"
# The platform's own name for the Storage root, used by workflow DSL nodes and
# by the commands a task builds from them.
PLATFORM_ROOT = "/workspace"

_WORKSPACE_PATH = (
    f"'{WORKSPACE_DIRNAME}/...' is a conversation workspace path, not a Storage "
    "path. Copy it into Storage first, for example under '/<dir>/<name>', then "
    "pass '/<dir>/<name>'."
)


class StoragePathError(ValueError):
    """Raised when a value cannot name a location inside Storage."""


def configured_mount() -> str:
    """The mount a deployment puts the user Storage on.

    A platform task container mounts the same Storage at the same place, so a
    path the model read inside its sandbox is one prefix away from the Storage
    path the platform reads.
    """

    return os.getenv("PYROMIND_SANDBOX_MOUNT_PATH", "").strip() or DEFAULT_MOUNT_PATH


def to_tool_storage_path(value: str, *, field: str = "path") -> str:
    """Normalize a path a model handed to a platform tool.

    The model reads Storage under the sandbox mount, so the same file reaches a
    tool as ``/target-workspace/datasets/x.csv``. The platform addresses it as
    ``/datasets/x.csv``, so the mount prefix is dropped here rather than left to
    the model to remember.
    """

    return to_storage_path(value, field=field, mount=configured_mount())


def to_storage_path(
    value: str,
    *,
    field: str = "path",
    mount: str | None = None,
) -> str:
    """Normalize an accepted spelling to a Storage-root path.

    ``mount`` names the sandbox mount point, so a container-absolute path
    (``/target-workspace/a.csv``) is accepted wherever the caller knows it.
    Without it, a mount-absolute path stays what it looks like -- a directory at
    the Storage root -- because guessing would be worse than failing.

    Two spellings name the Storage root itself and are read as the root: the
    sandbox mount (``/target-workspace``) and the platform's own ``/workspace``.
    A *user directory* that happens to be called ``storage`` keeps its absolute
    spelling (``/storage/...``), because that is data rather than a mount name.
    The relative ``storage/...`` form is the removed alias and is rejected.
    """

    raw = value.strip()
    if not raw:
        raise StoragePathError(f"{field} must not be empty.")
    if any(ord(character) < 32 for character in raw):
        raise StoragePathError(f"{field} contains control characters.")
    if mount:
        mount_root = "/" + mount.strip("/")
        if raw == mount_root or raw.startswith(mount_root + "/"):
            raw = raw[len(mount_root) :] or "/"
    if raw == PLATFORM_ROOT or raw.startswith(PLATFORM_ROOT + "/"):
        raw = raw[len(PLATFORM_ROOT) :] or "/"

    absolute = raw.startswith("/")
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if ".." in parts:
        raise StoragePathError(f"{field} must not contain '..'.")
    if not parts:
        raise StoragePathError(f"{field} must not be the Storage root.")
    if not absolute and parts[0] == REMOVED_ALIAS:
        root = (mount or DEFAULT_MOUNT_PATH).rstrip("/")
        raise StoragePathError(
            f"{field}: the '{REMOVED_ALIAS}/' alias was removed. Address Storage "
            f"by its mount path ('{root}/<path>') or by its Storage path "
            "('/<path>')."
        )
    if not absolute and parts[0] == WORKSPACE_DIRNAME:
        raise StoragePathError(f"{field}: {_WORKSPACE_PATH}")
    return "/" + "/".join(parts)
