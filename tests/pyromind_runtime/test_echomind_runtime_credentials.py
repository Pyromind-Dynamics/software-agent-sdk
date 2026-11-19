from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from pyromind_agent_server.echomind_runtime_credentials import (
    RuntimeCredentialIdentityMismatchError,
    RuntimeCredentialStore,
    RuntimeCredentialStoreError,
)
from pyromind_common.sdk.utils.cipher import Cipher


def test_credential_upsert_is_encrypted_scoped_and_restorable(tmp_path: Path) -> None:
    root = tmp_path / "credentials"
    store = RuntimeCredentialStore(root, cipher=Cipher("test-server-key"))
    assert store.load("owner-a", "us-west-1#pre") is None
    assert store.needs_refresh("owner-a", "us-west-1#pre", max_age_seconds=600)

    store.save("owner-a", "us-west-1#pre", "old-key", platform_user_id="platform-a")
    store.save("owner-a", "us-west-1#pre", "rotated-key", platform_user_id="platform-a")
    store.save("owner-b", "us-west-1#pre", "another-key", platform_user_id="platform-b")
    store.save("owner-a", "us-west-1", "production-key", platform_user_id="platform-c")

    restarted = RuntimeCredentialStore(root, cipher=Cipher("test-server-key"))
    assert restarted.load("owner-a", "us-west-1#pre") == "rotated-key"
    assert restarted.load("owner-a", "us-west-1") == "production-key"
    assert restarted.load("owner-b", "us-west-1#pre") == "another-key"
    assert restarted.load("owner-b", "us-west-1") is None
    assert not restarted.needs_refresh(
        "owner-a",
        "us-west-1#pre",
        max_age_seconds=600,
        platform_user_id="platform-a",
    )
    assert restarted.needs_refresh("owner-a", "us-west-1#pre", max_age_seconds=0)

    records = list(root.glob("*.json"))
    assert len(records) == 3
    persisted = "".join(path.read_text() for path in records)
    for text in (
        "old-key",
        "rotated-key",
        "production-key",
        "another-key",
        "owner-a",
        "owner-b",
        "platform-a",
        "us-west-1",
    ):
        assert text not in persisted


def test_runtime_credential_cannot_be_rebound_to_another_platform_user(
    tmp_path: Path,
) -> None:
    store = RuntimeCredentialStore(tmp_path, cipher=Cipher("test-server-key"))
    store.save("owner-a", "us-west-1", "original-key", platform_user_id="platform-a")

    with pytest.raises(RuntimeCredentialIdentityMismatchError):
        store.needs_refresh(
            "owner-a",
            "us-west-1",
            max_age_seconds=600,
            platform_user_id="platform-b",
        )
    with pytest.raises(RuntimeCredentialIdentityMismatchError):
        store.save("owner-a", "us-west-1", "other-key", platform_user_id="platform-b")

    assert store.load("owner-a", "us-west-1") == "original-key"


def test_concurrent_first_save_binds_only_one_platform_account(tmp_path: Path) -> None:
    barrier = Barrier(2)

    def save(platform_user_id: str) -> bool:
        store = RuntimeCredentialStore(tmp_path, cipher=Cipher("test-server-key"))
        barrier.wait(timeout=5)
        try:
            store.save(
                "owner-a",
                "us-west-1",
                f"{platform_user_id}-key",
                platform_user_id=platform_user_id,
            )
        except RuntimeCredentialIdentityMismatchError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        identities = ("platform-a", "platform-b")
        stored = list(pool.map(save, identities))

    assert sum(stored) == 1
    winner = identities[stored.index(True)]
    store = RuntimeCredentialStore(tmp_path, cipher=Cipher("test-server-key"))
    assert store.load("owner-a", "us-west-1") == f"{winner}-key"


def test_ciphertext_cannot_be_moved_to_another_scope(tmp_path: Path) -> None:
    store = RuntimeCredentialStore(tmp_path, cipher=Cipher("test-server-key"))
    store.save("owner-a", "us-west-1", "first-key", platform_user_id="platform-a")
    first = next(tmp_path.glob("*.json"))
    store.save("owner-b", "us-west-1", "second-key", platform_user_id="platform-b")
    second = next(path for path in tmp_path.glob("*.json") if path != first)
    second.write_bytes(first.read_bytes())

    with pytest.raises(RuntimeCredentialStoreError, match="Could not read"):
        store.load("owner-b", "us-west-1")
    assert store.load("owner-a", "us-west-1") == "first-key"


def test_missing_cipher_does_not_write_credentials(tmp_path: Path) -> None:
    root = tmp_path / "credentials"
    store = RuntimeCredentialStore(root, cipher=None)
    assert store.load("owner-a", "us-west-1") is None
    with pytest.raises(
        RuntimeCredentialStoreError, match="encryption is not configured"
    ):
        store.save("owner-a", "us-west-1", "test-key", platform_user_id="platform-a")
    assert not root.exists()
