from collections.abc import Callable

from pydantic import ValidationError

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.graph_repair import GraphRepairRequest
from pyromind_runtime.domain.snapshot import ExternalTaskState
from pyromind_runtime.ports.graph_repair import GraphRepairSubmitter


def handle_graph_repair(
    service: GraphRepairSubmitter,
    conversation_id: str,
    tasks: tuple[ExternalTaskState, ...],
    arguments: JsonObject,
    context: RequestContext,
    read_asset: Callable[[str], bytes] | None,
) -> JsonObject:
    try:
        request = GraphRepairRequest.model_validate(arguments)
    except ValidationError as exc:
        return {
            "status": "invalid",
            "errors": [
                f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                for item in exc.errors(include_input=False)[:8]
            ],
        }
    evidence: dict[str, JsonObject] = {}
    for task in tasks:
        if task.kind != "echomind_debug_repair":
            continue
        result = task.completion_result
        key = result.get("handoff_id")
        if isinstance(key, str):
            # Revision notices share the handoff and may omit the original log.
            evidence.setdefault(key, {}).update(result)
    handoff_id = request.handoff_id
    if request.action == "submit" and (not handoff_id or request.proposal is None):
        return {"status": "invalid", "error": "submit requires handoff_id and proposal"}
    if request.action == "inspect" and (
        request.proposal is not None or request.recipe_asset_replacement is not None
    ):
        return {"status": "invalid", "error": "Only submit accepts a proposal"}
    if handoff_id is None:
        if len(evidence) != 1:
            return {
                "status": "select_handoff" if evidence else "no_handoff",
                "handoffs": [
                    {
                        "handoff_id": key,
                        "run_id": item.get("run_id"),
                        "failed_node_id": item.get("failed_node_id"),
                        "provider_task_id": item.get("provider_task_id"),
                    }
                    for key, item in evidence.items()
                ],
            }
        handoff_id = next(iter(evidence))
    if handoff_id not in evidence:
        return {"status": "invalid", "error": "Unknown handoff in this conversation"}
    record = evidence[handoff_id]
    project_id, run_id = record.get("project_id"), record.get("run_id")
    if not isinstance(project_id, str) or not isinstance(run_id, str):
        return {"status": "invalid", "error": "Incomplete repair handoff"}
    try:
        if request.proposal is None:
            snapshot = service.inspect(
                conversation_id=conversation_id,
                project_id=project_id,
                run_id=run_id,
                handoff_id=handoff_id,
                context=context,
            )
            return {"status": "inspected", "evidence": {**record, **snapshot}}
        proposal = request.proposal.model_dump(mode="json", exclude_unset=True)
        if request.recipe_asset_replacement is not None:
            proposal["recipe_asset_replacement"] = (
                request.recipe_asset_replacement.model_dump(
                    mode="json", exclude_none=True
                )
            )
        return service.submit(
            conversation_id=conversation_id,
            project_id=project_id,
            run_id=run_id,
            handoff_id=handoff_id,
            context=context,
            proposal=proposal,
            read_asset=read_asset,
        )
    except ValueError as exc:
        return {"status": "invalid", "error": str(exc)}
