from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, SecretStr

from openhands.agent_server.echomind_storage import (
    canonical_sha256,
    read_model,
    write_model,
)
from openhands.agent_server.persistence.store import _file_lock
from openhands.sdk.utils.cipher import Cipher


class RuntimeCredentialStoreError(RuntimeError):
    pass


class RuntimeCredentialIdentityMismatchError(RuntimeCredentialStoreError):
    pass


class _EncryptedCredential(BaseModel):
    schema_version: Literal[1] = 1
    ciphertext: str


class _CredentialPayload(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    scope: str
    platform_user_id: str
    access_key: SecretStr
    updated_at: AwareDatetime


class RuntimeCredentialStore:
    """Store platform Access Keys encrypted by app owner and platform cluster."""

    def __init__(self, root: Path | str, *, cipher: Cipher | None) -> None:
        self._root = Path(root)
        self._cipher = cipher

    def load(self, user_id: str, cluster: str) -> str | None:
        credential = self._load(user_id, cluster)
        return credential.access_key.get_secret_value() if credential else None

    def needs_refresh(
        self,
        user_id: str,
        cluster: str,
        *,
        max_age_seconds: float,
        platform_user_id: str | None = None,
    ) -> bool:
        credential = self._load(user_id, cluster)
        if credential is None:
            return True
        if platform_user_id is not None:
            _require_identity(credential, platform_user_id)
        return (
            datetime.now(UTC) - credential.updated_at
        ).total_seconds() >= max_age_seconds

    def save(
        self,
        user_id: str,
        cluster: str,
        key: str,
        *,
        platform_user_id: str,
        observed_at: datetime | None = None,
    ) -> None:
        cipher = self._cipher
        if cipher is None:
            raise RuntimeCredentialStoreError(
                "Runtime credential encryption is not configured."
            )
        if not key.strip() or not platform_user_id.strip():
            raise ValueError("Runtime credentials require an Access Key and user ID.")

        updated_at = observed_at or datetime.now(UTC)
        path = self._entry_path(user_id, cluster)
        try:
            with _file_lock(path.with_suffix(".lock")):
                existing = _read_credential(path, cipher)
                if existing is not None:
                    _require_identity(existing, platform_user_id)
                    if existing.updated_at > updated_at:
                        return
                ciphertext = cipher.encrypt(
                    SecretStr(
                        json.dumps(
                            {
                                "scope": path.stem,
                                "platform_user_id": platform_user_id,
                                "access_key": key,
                                "updated_at": updated_at.isoformat(),
                            }
                        )
                    )
                )
                assert ciphertext is not None
                write_model(path, _EncryptedCredential(ciphertext=ciphertext))
        except OSError as exc:
            raise RuntimeCredentialStoreError(
                "Could not write encrypted runtime credentials."
            ) from exc

    def _load(self, user_id: str, cluster: str) -> _CredentialPayload | None:
        if self._cipher is None:
            return None
        return _read_credential(self._entry_path(user_id, cluster), self._cipher)

    def _entry_path(self, user_id: str, cluster: str) -> Path:
        if not user_id.strip() or not cluster.strip():
            raise ValueError("Runtime credentials require a user ID and cluster.")
        scope = canonical_sha256([user_id, cluster]).removeprefix("sha256:")
        return self._root / f"{scope}.json"


def _require_identity(credential: _CredentialPayload, platform_user_id: str) -> None:
    if credential.platform_user_id != platform_user_id:
        raise RuntimeCredentialIdentityMismatchError(
            "Runtime credentials are bound to a different Pyromind account."
        )


def _read_credential(path: Path, cipher: Cipher) -> _CredentialPayload | None:
    try:
        stored = read_model(path, _EncryptedCredential)
        decrypted = cipher.try_decrypt_str(stored.ciphertext)
        if decrypted is None:
            raise ValueError("Invalid runtime credential ciphertext")
        credential = _CredentialPayload.model_validate_json(decrypted)
        if credential.scope != path.stem:
            raise ValueError("Mismatched runtime credential scope")
        return credential
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise RuntimeCredentialStoreError(
            "Could not read encrypted runtime credentials."
        ) from None
