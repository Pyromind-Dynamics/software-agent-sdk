"""Shared utility functions for Pyromind."""

from .command import sanitized_env
from .datetime import PyromindUUID, utc_now
from .deprecation import (
    deprecated,
    warn_deprecated,
)
from .paging import page_iterator
from .truncate import (
    DEFAULT_TEXT_CONTENT_LIMIT,
    DEFAULT_TRUNCATE_NOTICE,
    maybe_truncate,
)


__all__ = [
    "DEFAULT_TEXT_CONTENT_LIMIT",
    "DEFAULT_TRUNCATE_NOTICE",
    "PyromindUUID",
    "maybe_truncate",
    "deprecated",
    "utc_now",
    "warn_deprecated",
    "page_iterator",
    "sanitized_env",
]
