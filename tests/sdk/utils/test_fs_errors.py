import errno

import pytest

from openhands.sdk.io.local import LocalFileStore
from openhands.sdk.utils.fs_errors import (
    CONVERSATION_SPACE_FULL_MESSAGE,
    ConversationStorageFullError,
    quota_exceeded_reason,
    raise_if_quota_exceeded,
)


def test_quota_exceeded_reason_matches_errno():
    assert quota_exceeded_reason(OSError(errno.EDQUOT, "fake")) == (
        CONVERSATION_SPACE_FULL_MESSAGE
    )
    assert quota_exceeded_reason(OSError(errno.ENOSPC, "fake")) == (
        CONVERSATION_SPACE_FULL_MESSAGE
    )


def test_quota_exceeded_reason_matches_message_text():
    assert quota_exceeded_reason(OSError("Disk quota exceeded")) == (
        CONVERSATION_SPACE_FULL_MESSAGE
    )
    assert quota_exceeded_reason(RuntimeError("No space left on device")) == (
        CONVERSATION_SPACE_FULL_MESSAGE
    )


def test_quota_exceeded_reason_unrelated_returns_none():
    assert quota_exceeded_reason(OSError(errno.EACCES, "permission denied")) is None
    assert quota_exceeded_reason(OSError("read-only file system")) is None
    assert quota_exceeded_reason(Exception("boom")) is None


def test_raise_if_quota_exceeded_translates_space_full_errors():
    with pytest.raises(ConversationStorageFullError) as exc_info:
        raise_if_quota_exceeded(OSError(errno.ENOSPC, "fake"))
    assert str(exc_info.value) == CONVERSATION_SPACE_FULL_MESSAGE
    assert exc_info.value.errno == errno.ENOSPC


def test_raise_if_quota_exceeded_ignores_unrelated_errors():
    raise_if_quota_exceeded(OSError(errno.EACCES, "permission denied"))


def test_local_filestore_write_translates_quota_error(tmp_path, monkeypatch):
    store = LocalFileStore(root=str(tmp_path))

    def fail_open(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("builtins.open", fail_open)
    with pytest.raises(ConversationStorageFullError) as exc_info:
        store.write("events/event-00000-test.json", "{}")
    assert str(exc_info.value) == CONVERSATION_SPACE_FULL_MESSAGE
