from dataclasses import dataclass
from pathlib import Path

from pyromind_agent_server.config import Config


@dataclass(frozen=True, slots=True)
class ProductServiceContext:
    conversations_dir: Path
    idle_eviction_timeout: float = 1800

    @classmethod
    def from_config(cls, config: Config):
        config.conversations_path.mkdir(parents=True, exist_ok=True)
        return cls(config.conversations_path, config.idle_conversation_eviction_seconds)
