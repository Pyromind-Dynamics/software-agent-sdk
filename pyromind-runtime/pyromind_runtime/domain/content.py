from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from pyromind_runtime.domain.base import ContractModel


type JsonObject = dict[str, JsonValue]


class TextContent(ContractModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(ContractModel):
    type: Literal["image"] = "image"
    image_urls: tuple[str, ...] = Field(min_length=1)


class ImageUrlContent(ContractModel):
    type: Literal["image_url"] = "image_url"
    url: str = Field(min_length=1)


type ContentBlock = Annotated[
    TextContent | ImageContent | ImageUrlContent,
    Field(discriminator="type"),
]
