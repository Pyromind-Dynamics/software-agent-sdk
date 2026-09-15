"""Managed review entry: pipeline.py review_input.jsonl review_output.jsonl."""

from image_utils import ImagePipelineConfig, run_image_pipeline_from_cli


CONFIG = ImagePipelineConfig(
    labeling_system_prompt=(
        "Inspect images using their stated roles and category definitions. "
        "For a template audit, check the extracted mask against the annotated source "
        "and reference: it must isolate the anomaly without copying normal structures. "
        "For a synthesis audit, independently identify the anomaly in image 1; image 2 "
        "is the normal reference; image 3 shows grayscale Gaussian high-pass response. "
        "High-pass edges are not automatically defects. Report category (unknown if "
        "unclear), pixel xyxy bbox with exclusive right/bottom edges in image 1, "
        "realism, extra anomalies and uncertainty. Assess local texture, material "
        "seams and structural compatibility, not just whether pixels changed. "
        "When source context is provided, compare candidate material support and "
        "boundary relationships against that source. Do not infer material identity "
        "from light/dark colors or assume any placement is valid. "
        "Set context_consistent=false if that relationship changes without an "
        "evidence-backed mapping in the strategy; if evidence is absent, also set "
        "uncertain=true. Use the stated pixel coordinates without guessing transforms. "
        "For no identifiable anomaly use bbox [0,0,0,0] and uncertain=true. "
        "Provide concrete visual reasons; never assume a candidate is valid."
    ),
    training_system_prompt="Inspect image anomalies using the supplied definitions.",
    response_json_schema={
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "answer": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "bbox_xyxy": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "realistic": {"type": "boolean"},
                    "extra_anomalies": {"type": "boolean"},
                    "uncertain": {"type": "boolean"},
                    "context_consistent": {"type": "boolean"},
                },
                "required": [
                    "category",
                    "bbox_xyxy",
                    "realistic",
                    "extra_anomalies",
                    "uncertain",
                    "context_consistent",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["reasoning", "answer"],
        "additionalProperties": False,
    },
    answer_is_json=True,
    batch_size=1,
    max_workers=1,
)


if __name__ == "__main__":
    run_image_pipeline_from_cli(CONFIG)
