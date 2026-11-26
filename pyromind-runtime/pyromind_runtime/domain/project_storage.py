from typing import Literal

from pydantic import Field, field_validator

from pyromind_runtime.domain.base import AwareDatetime, ContractModel
from pyromind_runtime.domain.storage_paths import to_storage_api_path


class StorageProxyConfig(ContractModel):
    sandbox_id: str = Field(pattern=r"^sb-[a-zA-Z0-9_-]+$")
    mount_root: str = "/workspace"

    @field_validator("mount_root")
    @classmethod
    def validate_mount_root(cls, value: str) -> str:
        if not value.startswith("/") or value == "/":
            raise ValueError("Storage mount must be an absolute directory below root")
        result = to_storage_api_path("/__mount__" + value).removeprefix("/__mount__")
        if not result:
            raise ValueError("Storage mount must stay below the container root")
        return result


class ProjectStorageBinding(ContractModel):
    project_id: str
    owner_user_id: str
    cluster: str
    config: StorageProxyConfig
    status: Literal["ready", "blocked"] = "blocked"
    last_error: str | None = None
    last_checked_at: AwareDatetime
