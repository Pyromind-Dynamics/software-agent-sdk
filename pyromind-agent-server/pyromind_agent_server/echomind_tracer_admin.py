from pathlib import Path

from pydantic import BaseModel, SecretStr

from openhands.agent_server.config import Config
from openhands.agent_server.echomind_storage import (
    canonical_uuid,
    project_directory,
    read_model,
    write_model,
)
from openhands.sdk.utils.cipher import Cipher


class _TracerAdminSecret(BaseModel):
    resource_id: str
    ciphertext: str


def build_tracer_admin_cipher(config: Config | None) -> Cipher | None:
    return (
        config.cipher if config is not None and config.secret_key is not None else None
    )


def _scope(project_id: str, resource_id: str) -> str:
    return f"{canonical_uuid(project_id, kind='project')}:{resource_id}:"


def save_tracer_admin_key(
    *,
    projects_root: Path,
    project_id: str,
    resource_id: str,
    cipher: Cipher,
    key: str,
) -> None:
    ciphertext = cipher.encrypt(SecretStr(_scope(project_id, resource_id) + key))
    assert ciphertext is not None
    write_model(
        project_directory(projects_root, project_id) / "tracer_admin.json",
        _TracerAdminSecret(resource_id=resource_id, ciphertext=ciphertext),
    )


def load_tracer_admin_key(
    *,
    projects_root: Path,
    project_id: str,
    resource_id: str,
    cipher: Cipher | None,
) -> str | None:
    if cipher is None:
        return None
    try:
        stored = read_model(
            project_directory(projects_root, project_id) / "tracer_admin.json",
            _TracerAdminSecret,
        )
    except FileNotFoundError:
        return None
    if stored.resource_id != resource_id:
        return None
    value = cipher.try_decrypt_str(stored.ciphertext)
    scope = _scope(project_id, resource_id)
    return value.removeprefix(scope) if value and value.startswith(scope) else None
