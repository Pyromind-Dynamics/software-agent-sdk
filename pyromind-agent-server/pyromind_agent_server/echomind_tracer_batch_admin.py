from collections.abc import Mapping


def is_tracer_batch_admin_response(payload: object) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("ok") is True
        and payload.get("batch_mode") is True
        and isinstance(payload.get("batches"), list)
    )
