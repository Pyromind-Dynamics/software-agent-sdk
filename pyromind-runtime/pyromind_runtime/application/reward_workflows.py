from __future__ import annotations

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.reward_bundle import RewardBundle


def training_workflow(workflow: JsonObject) -> bool:
    return any(
        isinstance(node_type, str) and node_type.startswith("ModelTrain")
        for node_type in (data.get("nodeType") for data in _nodes(workflow).values())
    )


def validate_reward_workflow(workflow: JsonObject, bundle: RewardBundle) -> None:
    nodes = _nodes(workflow)
    edges = workflow.get("edges")
    grpo = [
        key
        for key, node in nodes.items()
        if node.get("nodeType") == "ModelTrainGRPONode"
    ]
    if not grpo or not isinstance(edges, list):
        raise ValueError(
            "rubric_rl training requires GRPO with the Project's frozen Reward bundle"
        )

    def incoming(node: str) -> list[JsonObject]:
        return [
            edge
            for edge in edges
            if isinstance(edge, dict) and edge.get("target") == node
        ]

    for identity in grpo:
        config_edges = [
            edge
            for edge in incoming(identity)
            if edge.get("targetHandle") == "reward_config"
        ]
        config_id = _linked_node(config_edges, "reward_config")
        config = _config(nodes, config_id, "RewardConfigBuilderNode")
        if config.get("normalize", False) is not False or any(
            config.get(f"reward_item_{index}") not in (None, "")
            for index in range(1, 6)
        ):
            raise ValueError(
                "use the frozen Reward score without additional items or normalization"
            )

        item_edges = incoming(config_id)
        item_id = _linked_node(item_edges, "reward_item")
        if item_edges[0].get("targetHandle") not in {
            f"reward_item_{index}" for index in range(1, 6)
        }:
            raise ValueError("connect the frozen reward_item to a reward_item input")
        item = _config(nodes, item_id, "RewardItemBuilderCustomNode")
        if incoming(item_id) or item.get("entry") != bundle.studio_entry:
            raise ValueError(f"Reward Custom Entry must be {bundle.studio_entry}")
        weight = item.get("weight", 1)
        if (
            type(weight) not in (int, float)
            or weight != 1
            or item.get("kwargs", "") != ""
        ):
            raise ValueError(
                "use weight 1 and empty kwargs for the frozen Reward entry"
            )


def _linked_node(edges: list[JsonObject], source_handle: str) -> str:
    if len(edges) != 1 or edges[0].get("sourceHandle") != source_handle:
        raise ValueError(
            f"expected one {source_handle} connection for the frozen Reward"
        )
    node = edges[0].get("source")
    if not isinstance(node, str):
        raise ValueError("invalid Reward source node")
    return node


def _nodes(workflow: JsonObject) -> dict[str, JsonObject]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return {}
    result: dict[str, JsonObject] = {}
    for node in nodes:
        if isinstance(node, dict):
            identity, data = node.get("id"), node.get("data")
            if isinstance(identity, str) and isinstance(data, dict):
                result[identity] = data
    return result


def _config(nodes: dict[str, JsonObject], identity: str, node_type: str) -> JsonObject:
    node = nodes.get(identity, {})
    config = node.get("config", {})
    if node.get("nodeType") != node_type or not isinstance(config, dict):
        raise ValueError(f"frozen Reward connection requires {node_type}")
    return config
