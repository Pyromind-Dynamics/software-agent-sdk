from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pyromind_runtime.domain.base import ContractModel


GRAPH_REPAIR_PROPOSAL = "graph_repair_proposal"


class RuntimeGraphRepairChanges(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    command_template: str | None = Field(default=None, min_length=1, max_length=20000)
    workflow_bindings: dict[str, str] | None = None
    workflow_templates: dict[str, str] | None = None
    workflow_parameter_updates: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict
    )


class GraphRepairDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    release_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    node_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2000)
    runtime_config: RuntimeGraphRepairChanges


class RecipeAssetReplacement(ContractModel):
    old_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    new_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    new_asset_path: str | None = Field(
        default=None,
        max_length=4096,
        description=(
            "Absolute /workspace path to a frozen script in this conversation. "
            "Required when a new replay produced the asset in a different directory. "
            "End the path with recipe-assets/<new_sha256>/<original filename>."
        ),
    )


class GraphRepairRequest(ContractModel):
    """Inspect a Graph Debug failure or submit a candidate for workbench review."""

    action: Literal["inspect", "submit"]
    handoff_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Required for submit. Omit on inspect to discover the handoff.",
    )
    proposal: GraphRepairDraft | None = Field(
        default=None, description="Required on submit; use the frozen release digest."
    )
    recipe_asset_replacement: RecipeAssetReplacement | None = Field(
        default=None,
        description=(
            "For submit only. Replace one frozen recipe script from inspect using "
            "its current SHA-256 and the verified replacement SHA-256. Leave "
            "proposal.runtime_config empty for an asset-only change."
        ),
    )
