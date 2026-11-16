import pytest

from openhands.tools.workflow.storage_paths import validate_persistent_output_paths


@pytest.mark.parametrize(
    ("node_type", "field"),
    [
        ("ModelTrainSFTNode", "output_path"),
        ("CloneAndCacheModel", "target_path"),
        ("DownloadAndCacheModel", "cache_dir"),
    ],
)
@pytest.mark.parametrize(
    "output_path",
    ["/models/run-1", "models/run-1", "/workspace", "/workspace/../models/run-1"],
)
def test_output_must_be_inside_the_persistent_workspace(node_type, field, output_path):
    with pytest.raises(ValueError, match="must be under /workspace"):
        validate_persistent_output_paths(
            [
                {
                    "id": "node",
                    "data": {
                        "nodeType": node_type,
                        "config": {field: output_path},
                    },
                }
            ]
        )


def test_training_output_accepts_a_persistent_workspace_directory():
    validate_persistent_output_paths(
        [
            {
                "id": "train",
                "data": {
                    "nodeType": "ModelTrainSFTNode",
                    "config": {"output_path": "/workspace/models/run-1"},
                },
            }
        ]
    )
