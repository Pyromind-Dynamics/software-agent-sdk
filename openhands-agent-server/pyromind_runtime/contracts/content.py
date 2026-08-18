from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from pyromind_runtime.contracts.base import ContractModel


class TextContentBlock(ContractModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentBlock(ContractModel):
    type: Literal["image"] = "image"
    mime_type: str
    data: str


class ResourceContentBlock(ContractModel):
    type: Literal["resource"] = "resource"
    uri: str
    mime_type: str | None = None
    text: str | None = None


type ContentBlock = Annotated[
    TextContentBlock | ImageContentBlock | ResourceContentBlock,
    Field(discriminator="type"),
]

type JsonObject = dict[str, JsonValue]
