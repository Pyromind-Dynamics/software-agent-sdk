"""Validate persistent output locations in platform workflow definitions."""

from pathlib import PurePosixPath


_PERSISTENT_OUTPUT_PATH_FIELDS = {
    "CloneAndCacheDataset": ("target_path",),
    "CloneAndCacheModel": ("target_path",),
    "DataPreprocessLabelNode": ("output_path",),
    "DataPreprocessMergeNode": ("output_path",),
    "DataPreprocessPromptExpandNode": ("output_path",),
    "DataPreprocessReasoningNode": ("output_path",),
    "DataPreprocessSplitNode": ("output_path",),
    "DownloadAndCacheDataset": ("cache_dir",),
    "DownloadAndCacheModel": ("cache_dir",),
    "DownloadURLNode": ("output_path",),
    "ModelEvalApiNode": ("output_path",),
    "ModelMergeLoraNode": ("output_path",),
    "ModelTrainDPONode": ("output_path",),
    "ModelTrainGRPONode": ("output_path",),
    "ModelTrainOPDNode": ("output_path",),
    "ModelTrainSFTNode": ("output_path",),
    "PythonFileProcessorNode": ("output_file_path",),
}


def validate_persistent_output_paths(nodes: object) -> None:
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        node_type = data.get("nodeType")
        if not isinstance(node_type, str):
            continue
        fields = _PERSISTENT_OUTPUT_PATH_FIELDS.get(node_type)
        if fields is None:
            continue
        config = data.get("config")
        if not isinstance(config, dict):
            continue
        for field in fields:
            output_path = config.get(field)
            if not isinstance(output_path, str) or not output_path.strip():
                continue
            normalized = PurePosixPath(output_path.strip())
            if (
                not normalized.is_absolute()
                or normalized.parts[:2] != ("/", "workspace")
                or len(normalized.parts) < 3
                or ".." in normalized.parts
            ):
                node_id = node.get("id", "unknown")
                raise ValueError(
                    f"Node {node_id} {field} must be under /workspace/ so its "
                    "output persists in user Storage."
                )
