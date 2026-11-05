"""Resolve an image reference without interpreting the message's business meaning."""

from pathlib import PurePosixPath

from pyromind_runtime.domain.content import ImageContent, ImageUrlContent
from pyromind_runtime.domain.user_input import PendingUserInput


def input_media_path(
    pending: PendingUserInput, content_index: int, image_index: int
) -> str:
    presentation = pending.presentation
    if presentation is None or not 0 <= content_index < len(presentation.content):
        raise FileNotFoundError("message image not found")
    content = presentation.content[content_index]
    paths = (
        (content.url,)
        if isinstance(content, ImageUrlContent)
        else (content.image_urls if isinstance(content, ImageContent) else ())
    )
    if not 0 <= image_index < len(paths):
        raise FileNotFoundError("message image not found")
    path = paths[image_index]
    if (
        not path.startswith("/")
        or path.startswith("//")
        or any(c in path for c in ("\\", ":", "?", "#"))
        or ".." in PurePosixPath(path).parts
    ):
        raise ValueError("message image must reference an absolute Storage path")
    return path
