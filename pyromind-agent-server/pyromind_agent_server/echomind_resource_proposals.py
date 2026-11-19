from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pyromind_agent_server.echomind_projects import ProjectInfo
from pyromind_agent_server.echomind_publication import PublishedGraph
from pyromind_agent_server.echomind_runtime_slots import (
    RuntimeSlotSetupRequest,
    ServingInferenceCreateConfig,
    TracerCreateConfig,
)
from pyromind_agent_server.echomind_storage import (
    canonical_sha256,
    canonical_uuid,
    project_directory,
    read_model,
    write_model,
)
from pyromind_runtime.domain.echomind_resources import RuntimeResourceProposalArguments


class RuntimeResourceProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    release_id: str
    setup: RuntimeSlotSetupRequest
    digest: str


def prepare_runtime_resources(
    root: Path,
    project: ProjectInfo,
    release: PublishedGraph,
    arguments: RuntimeResourceProposalArguments,
) -> RuntimeResourceProposal:
    if release.publication_mode != "production":
        raise ValueError(
            "Review the execution plan before configuring runtime resources."
        )
    if arguments.inference_id is None:
        if not arguments.model_path or not arguments.gpu_card:
            raise ValueError(
                "Provide the Storage model path and GPU type for Inference."
            )
        if arguments.tracer_id:
            raise ValueError(
                "An existing Tracer requires its current Inference job ID."
            )
        inference = ServingInferenceCreateConfig(
            model_path=arguments.model_path,
            model_name=arguments.model_name,
            gpu_card=arguments.gpu_card,
        )
    else:
        if arguments.model_path or arguments.gpu_card:
            raise ValueError(
                "Choose either an existing Inference or a model to deploy."
            )
        inference = None
    project_id = canonical_uuid(project.project.project_id, kind="project")
    setup = RuntimeSlotSetupRequest(
        serving_inference_id=arguments.inference_id,
        serving_inference_create=inference,
        attach_existing_tracer_id=arguments.tracer_id,
        tracer=TracerCreateConfig(
            origin_model=arguments.model_name,
            training_model=arguments.model_name,
            training_batch_size=arguments.batch_size,
            trajectory_buffer_size=arguments.batch_size * 3,
            time_per_round_seconds=60,
            training_rounds=1,
            training_save_path=f"/workspace/.pyromind-agent/runtime/{project_id}/models",
        ),
    )
    data = {
        "project_id": project_id,
        "release_id": release.release_id,
        "setup": setup.model_dump(mode="json"),
    }
    proposal = RuntimeResourceProposal.model_validate(
        {**data, "digest": canonical_sha256(data)}
    )
    write_model(_proposal_path(root, project_id), proposal)
    return proposal


def read_runtime_resource_proposal(
    root: Path, project_id: str, release_id: str
) -> RuntimeResourceProposal | None:
    try:
        proposal = read_model(_proposal_path(root, project_id), RuntimeResourceProposal)
    except FileNotFoundError:
        return None
    if (
        proposal.project_id != canonical_uuid(project_id, kind="project")
        or proposal.release_id != release_id
        or proposal.digest
        != canonical_sha256(proposal.model_dump(mode="json", exclude={"digest"}))
    ):
        return None
    return proposal


def _proposal_path(root: Path, project_id: str) -> Path:
    return project_directory(root, project_id) / "runtime_resource_proposal.json"
