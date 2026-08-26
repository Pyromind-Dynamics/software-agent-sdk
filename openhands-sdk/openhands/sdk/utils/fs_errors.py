"""Helpers for translating filesystem write failures into readable messages.

The workspace storage quota (an XFS project hard limit) is enforced by the
filesystem: a write past the limit raises ``OSError(EDQUOT)`` (or ``ENOSPC``
when the underlying device itself is full). The same condition can instead
surface as command stderr text, for example a ``bash`` command that hits the
quota exits non-zero without raising a Python errno. Both tools and the
agent-server surface these conditions to the user, so the detection lives here
in the SDK to avoid a reverse dependency from ``openhands-tools`` back onto
``openhands-agent-server``.
"""

from __future__ import annotations

import errno


#: User-facing reason surfaced when a conversation's storage is full. Kept as a
#: single constant so every write path that hits the quota hard limit reports
#: the same actionable message instead of a raw OS error.
CONVERSATION_SPACE_FULL_MESSAGE = (
    "Conversation storage space is full. Please start a new conversation."
)

#: OS errnos that mean a write failed because the conversation's storage (an XFS
#: project quota hard limit) or the underlying device ran out of space.
_SPACE_FULL_ERRNOS = frozenset({errno.EDQUOT, errno.ENOSPC})

#: Canonical error text (used by Python strerror and by shell commands that hit
#: the quota) that identifies a disk-full condition; matched case-insensitively.
_SPACE_FULL_TEXT = ("disk quota exceeded", "no space left on device")


def quota_exceeded_reason(exc: BaseException) -> str | None:
    """Return the readable space-full message when ``exc`` reports a full disk.

    Checks both the ``errno`` carried by an ``OSError`` and the error message
    text (for failures that surface as command stderr rather than a Python
    errno). Returns ``None`` when the failure is unrelated, letting callers keep
    their existing error text.
    """
    if getattr(exc, "errno", None) in _SPACE_FULL_ERRNOS:
        return CONVERSATION_SPACE_FULL_MESSAGE
    text = str(exc).lower()
    if any(marker in text for marker in _SPACE_FULL_TEXT):
        return CONVERSATION_SPACE_FULL_MESSAGE
    return None
