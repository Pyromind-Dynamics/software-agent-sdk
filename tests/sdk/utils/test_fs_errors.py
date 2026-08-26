import errno

from openhands.sdk.utils.fs_errors import (
    CONVERSATION_SPACE_FULL_MESSAGE,
    quota_exceeded_reason,
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
