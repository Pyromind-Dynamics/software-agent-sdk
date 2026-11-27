"""Recover the dataset reader and its input bindings from an archived SFT workflow."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.storage_paths import normalize_workspace_path


@dataclass(frozen=True)
class SFTReader:
    node_type: str
    config: JsonObject
    paths: dict[str, str]
    version: str | None

    def field_mapping(self) -> dict[str, str]:
        if self.node_type == "DatasetConfigBuilderMessageNode":
            if self.config.get("messages_field") == "messages":
                return {"messages_field": "messages"}
            return {}
        if self.node_type != "DatasetConfigBuilderVisionNode":
            return {}
        roles = ["image_field", "user_prompt_field", "assistant_response_field"]
        if self.config.get("system_prompt_field"):
            roles.append("system_prompt_field")
        fields = {
            role: value
            for role in roles
            if isinstance(value := self.config.get(role), str)
            and re.fullmatch(r"[\w.-]{1,128}", value)
        }
        return fields if len(set(fields.values())) == len(roles) else {}


def sft_reader(workflow: Mapping[str, object]) -> SFTReader | None:
    nodes, edges = workflow.get("nodes"), workflow.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None
    graph: dict[str, JsonObject] = {}
    for node in nodes:
        if not isinstance(node, dict):
            return None
        identity, data = node.get("id"), node.get("data")
        if (
            not isinstance(identity, str)
            or identity in graph
            or not isinstance(data, dict)
        ):
            return None
        graph[identity] = data

    def sources(target: str, port: str) -> list[JsonObject]:
        return [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and edge.get("target") == target
            and edge.get("targetHandle") == port
        ]

    def upstream(target: str, port: str, output: str) -> str | None:
        incoming = sources(target, port)
        if len(incoming) != 1 or incoming[0].get("sourceHandle") != output:
            return None
        source = incoming[0].get("source")
        return source if isinstance(source, str) and source in graph else None

    trainers = [i for i, d in graph.items() if d.get("nodeType") == "ModelTrainSFTNode"]
    if len(trainers) != 1:
        return None
    dataset = upstream(trainers[0], "dataset_config", "dataset_config")
    if not dataset or graph[dataset].get("nodeType") != "DatasetConfigBuilderNode":
        return None
    reader = upstream(dataset, "dataset_kind_config", "dataset_kind_config")
    if not reader:
        return None
    kind, config = graph[reader].get("nodeType"), graph[reader].get("config")
    if not isinstance(kind, str) or not isinstance(config, dict):
        return None
    paths = {}
    direct = graph[dataset].get("config")
    for port in ("train_data_path", "val_data_path"):
        value = direct.get(port) if isinstance(direct, dict) else None
        if sources(dataset, port):
            join = upstream(dataset, port, "joined_path")
            if value or not join or graph[join].get("nodeType") != "PathJoinNode":
                continue
            join_config = graph[join].get("config")
            if not isinstance(join_config, dict) or any(
                sources(join, p) for p in ("base_path", "subpath")
            ):
                continue
            base, sub = join_config.get("base_path"), join_config.get("subpath")
            if not isinstance(base, str) or not isinstance(sub, str) or not base:
                continue
            value = base.rstrip("/") + "/" + sub.lstrip("/")
        if (
            isinstance(value, str)
            and value
            and not any(marker in value for marker in ("{", "}", "$", "[redacted]"))
        ):
            try:
                validate_persisted_payload(value)
                paths[port] = normalize_workspace_path(value)
            except ValueError:
                continue
    version = graph[reader].get("version")
    return SFTReader(
        kind,
        config,
        paths,
        version
        if isinstance(version, str)
        and re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,63}", version)
        else None,
    )


def archived_sft_input(workflow: JsonObject) -> JsonObject | None:
    reader = sft_reader(workflow)
    if reader is None or "train_data_path" not in reader.paths:
        return None
    fields = reader.field_mapping()
    if not fields:
        return None
    return {
        "reader": reader.node_type,
        "reader_version": reader.version,
        "fields": dict(fields),
        "paths": dict(reader.paths),
    }


def train_reader_fields(runtime: Mapping[str, object]) -> dict[str, str]:
    workflow = runtime.get("workflow")
    reader = sft_reader(workflow) if isinstance(workflow, dict) else None
    return reader.field_mapping() if reader else {}
