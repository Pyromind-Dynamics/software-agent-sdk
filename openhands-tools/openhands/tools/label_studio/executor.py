"""Executor for Label Studio project operations."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from openhands.sdk.tool import ToolExecutor
from openhands.tools.label_studio.api_client import (
    LabelStudioAPIClient,
    LabelStudioAPIError,
)
from openhands.tools.label_studio.converter import (
    DATASET_FILE_MAX_BYTES,
    FILE_ADAPTERS,
    AVITrainToLabelStudioConverter,
    ConversionError,
    ConvertedManifest,
    LabelStudioToAVITrainConverter,
)
from openhands.tools.label_studio.definition import (
    LabelStudioProjectAction,
    LabelStudioProjectObservation,
)
from openhands.tools.label_studio.export_ticket import PortalExportTicketProvider
from openhands.tools.label_studio.field_map import (
    FieldMap,
    default_field_map,
    field_map_digest,
    parse_field_map,
)
from openhands.tools.label_studio.media_signer import PortalMediaSigner
from openhands.tools.label_studio.models import ProjectState
from openhands.tools.label_studio.skill_helpers import (
    extract_control_names,
    validate_label_config_xml,
)
from openhands.tools.label_studio.token_provider import PortalTokenProvider
from openhands.tools.pyromind_dataset.definition import (
    PYROMIND_AGENT_STORAGE_ROOT,
    StorageFileNotFoundError,
    _strip_workspace_prefix,
    download_file_from_pyromind,
    upload_local_file_to_pyromind,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


logger = logging.getLogger(__name__)


# Batch files are capped at 10 MB by the converter.
_MANIFEST_BATCH_MAX_BYTES = 16 * 1024 * 1024
_PROJECT_STATE_MAX_BYTES = 64 * 1024

_MEDIA_PATH_SUFFIX = "_path"
# Label Studio has no bulk task update, so refreshing re-signs one media URL per
# task data field and writes each task back on its own request.
_MEDIA_REFRESH_WORKERS = 8

# The Label Studio deployment's export button pushes the project export to this
# object, and the portal only accepts a write to that exact key. All three sides
# spell the name out on their own, so changing it is a three-repository change.
EXPORT_OBJECT_NAME = "label_studio_export.json"
_PROJECT_DESCRIPTION_HEADER = "PyroMind 导出位置 / PyroMind export target:"


def _project_ref(
    ls_base_url: str,
    cluster: str,
    dataset_path: str,
    adapter: str,
    config_hash: str,
    idempotency_key: str | None,
    field_map_hash: str = "",
    dataset_digest: str = "",
) -> str:
    """Derive a stable project ref so a retried create resumes, not duplicates.

    Every input that changes the imported task set is part of the seed, so the
    same dataset, adapter, and label config always resolve to the same project.
    ``cluster`` is part of the seed because Storage is replicated per cluster: the
    same path names different objects in us-west-1 and us-west-2, and one shared
    Label Studio would otherwise adopt the other cluster's project by title.
    ``field_map_hash`` is part of it because bindings decide where each value
    lands: reusing a project across two different maps would keep serving the
    first one, since the reuse check only looks at status.
    ``dataset_digest`` is part of it for a dataset that lives in one file: such a
    file is rewritten in place, so the path alone cannot tell a re-export from
    the revision already imported. A directory dataset gets a new path per
    export, which is why its content is not read here.
    ``idempotency_key`` lets a caller ask for a distinct project over otherwise
    identical inputs.
    """
    seed = "|".join(
        (
            ls_base_url,
            cluster,
            dataset_path,
            adapter,
            config_hash,
            field_map_hash,
            dataset_digest,
            idempotency_key or "",
        )
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pyromind-label-studio:{seed}"))


class LabelStudioProjectExecutor(
    ToolExecutor[
        LabelStudioProjectAction,
        LabelStudioProjectObservation,
    ]
):
    """Executes Label Studio project operations server-side."""

    def __init__(
        self,
        *,
        ls_base_url: str,
        ls_token_secret: str,
        storage_base_url: str,
        headers: dict[str, str] | None = None,
        secret_headers: dict[str, str] | None = None,
        portal_base_url: str = "",
        cluster: str = "",
        batch_size: int = 500,
        timeout: int = 60,
    ) -> None:
        self._ls_base_url = ls_base_url.rstrip("/")
        self._ls_token_secret = ls_token_secret
        self._storage_base_url = storage_base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._secret_headers = dict(secret_headers or {})
        self._portal_base_url = portal_base_url.rstrip("/")
        self._cluster = cluster
        self._batch_size = batch_size
        self._timeout = timeout

    def __call__(
        self,
        action: LabelStudioProjectAction,
        conversation: BaseConversation | None = None,
    ) -> LabelStudioProjectObservation:
        handlers = {
            "create": self._handle_create,
            "get": self._handle_get,
            "update_config": self._handle_update_config,
            "status": self._handle_status,
            "export": self._handle_export,
            "refresh_media": self._handle_refresh_media,
        }
        handler = handlers.get(action.operation)
        if handler is None:
            return self._error(
                action.operation, f"Unknown operation: {action.operation}"
            )
        try:
            return handler(action, conversation)
        except (ValueError, ConversionError, LabelStudioAPIError) as exc:
            return self._error(action.operation, str(exc))

    def _handle_create(
        self,
        action: LabelStudioProjectAction,
        conversation: BaseConversation | None,
    ) -> LabelStudioProjectObservation:
        if not action.dataset_path:
            raise ValueError("dataset_path is required for operation='create'.")
        if not action.label_config_path:
            raise ValueError("label_config_path is required for operation='create'.")

        dataset_path = _normalize_storage_path(action.dataset_path, "dataset_path")
        xml_content = self._read_workspace_file(action.label_config_path, conversation)
        if xml_content is None:
            raise ValueError(
                f"label_config.xml not found in workspace: {action.label_config_path}"
            )

        # Both halves of the contract are read before anything is uploaded: the
        # XML says which controls exist, the field map says which values land on
        # them, and a mismatch between the two is cheapest to catch right here.
        field_map, field_map_hash = self._resolve_field_map(action, conversation)
        xml_text = xml_content.decode("utf-8")
        # Local checks catch a malformed document; the binding contract catches a
        # well-formed one the converter could not pre-annotate against.
        validate_label_config_xml(xml_text, adapter=action.adapter, field_map=field_map)
        config_hash = hashlib.sha256(xml_content).hexdigest()
        dataset_content = self._read_dataset_file(
            action.adapter, dataset_path, conversation
        )
        project_ref = _project_ref(
            self._ls_base_url,
            self._cluster,
            dataset_path,
            action.adapter,
            config_hash,
            action.idempotency_key,
            field_map_hash,
            hashlib.sha256(dataset_content).hexdigest() if dataset_content else "",
        )
        artifact_dir = f"{PYROMIND_AGENT_STORAGE_ROOT}/label-studio/{project_ref}"

        state = self._load_state(project_ref, conversation)
        if state is not None and state.status == "READY":
            summary = (
                f"Label Studio project already exists for this dataset and label "
                f"config: project_ref={state.project_ref} "
                f"project_id={state.project_id} status=READY "
                f"tasks={state.imported_count}/{state.total_tasks} "
                f"open_url={self._open_url(state.project_id)}"
            )
            return self._state_to_observation("create", state, summary=summary)

        manifest: ConvertedManifest | None = None
        if state is None:
            # Label Studio's own validator runs before any artifact is uploaded, so
            # one bad config fails in a single request instead of after the dataset
            # has been converted, its media signed, and its manifest uploaded.
            self._validate_with_label_studio(xml_text, conversation)
            manifest, state = self._create_project(
                action=action,
                dataset_path=dataset_path,
                config_hash=config_hash,
                project_ref=project_ref,
                artifact_dir=artifact_dir,
                xml_content=xml_content,
                media_signer=self._build_media_signer(conversation),
                conversation=conversation,
                field_map=field_map,
                field_map_hash=field_map_hash,
                dataset_content=dataset_content,
            )

        ls_api = self._build_ls_api(conversation)
        try:
            for index, tasks in self._iter_pending_batches(
                artifact_dir, state, manifest, conversation
            ):
                ls_api.import_tasks(project_id=state.project_id, tasks=tasks)
                state.imported_count += len(tasks)
                state.next_batch = index + 1
                self._save_state(artifact_dir, state, conversation)
        except (LabelStudioAPIError, ConversionError, ValueError) as exc:
            state.status = "ERROR"
            state.last_error = str(exc)
            self._save_state(artifact_dir, state, conversation)
            return self._error(
                "create",
                f"Import failed after {state.imported_count}/{state.total_tasks} "
                f"tasks: {exc}",
                state,
            )

        state.status = "READY"
        state.last_error = None
        self._save_state(artifact_dir, state, conversation)

        summary = (
            f"Label Studio project created: project_ref={state.project_ref} "
            f"project_id={state.project_id} status=READY "
            f"tasks={state.imported_count}/{state.total_tasks} "
            f"manifest_path={artifact_dir}/manifests/manifest.json "
            f"open_url={self._open_url(state.project_id)}"
        )
        if manifest is not None and manifest.unmapped_quality:
            summary += (
                " warning=unmapped_quality:"
                + ",".join(manifest.unmapped_quality)
                + " (written to predictions verbatim, so they may not render -- "
                "make sure the config's <Choice> values cover them)"
            )
        if manifest is not None and manifest.unmatched_regions:
            summary += (
                " warning=unmatched_regions:"
                + ",".join(manifest.unmatched_regions)
                + " (no sample carries these region fields, so no rectangles were "
                "built -- fix the binding's source or the upstream row contract)"
            )
        return self._state_to_observation("create", state, summary=summary)

    def _validate_with_label_studio(
        self,
        xml_text: str,
        conversation: BaseConversation | None,
    ) -> None:
        """Ask Label Studio to validate the config before anything is uploaded.

        ``POST /api/projects/validate/`` is the project-less validator, so this
        costs one request and creates no state. Without it an invalid config only
        fails at ``create_project`` -- after the whole dataset has been converted
        and every media URL signed.
        """
        ls_api = self._build_ls_api(conversation)
        try:
            ls_api.validate_config_standalone(label_config=xml_text)
        except LabelStudioAPIError as exc:
            raise LabelStudioAPIError(
                f"Label Studio rejected label_config.xml: {exc}"
            ) from exc

    def _resolve_field_map(
        self,
        action: LabelStudioProjectAction,
        conversation: BaseConversation | None,
    ) -> tuple[FieldMap, str]:
        """Load the caller's bindings, or the adapter's built-in ones.

        A declared map is read and validated before anything is converted, so an
        unknown key or a mistyped control fails in one request rather than after
        the dataset has been converted and every media URL signed. The digest is
        empty for built-in bindings, which keeps an existing project resumable
        across a deploy that did not change them.
        """
        if not action.field_map_path:
            return default_field_map(action.adapter), ""

        raw = self._read_workspace_file(action.field_map_path, conversation)
        if raw is None:
            raise ValueError(
                f"field map not found in workspace: {action.field_map_path}"
            )
        try:
            declared = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"field map is not readable JSON: {action.field_map_path}: {exc}"
            ) from exc
        try:
            field_map = parse_field_map(declared, adapter=action.adapter)
        except ValueError as exc:
            raise ValueError(
                f"invalid field map {action.field_map_path}: {exc}"
            ) from exc
        return field_map, field_map_digest(field_map)

    def _read_dataset_file(
        self,
        adapter: str,
        dataset_path: str,
        conversation: BaseConversation | None,
    ) -> bytes | None:
        """Read a file dataset, so its content can seed the project ref.

        The file is read here rather than by the converter alone because the ref
        has to be known before any artifact is written, and a file dataset is
        edited in place: hashing what it holds is the only way a re-export lands
        on a new project instead of resuming the one built from the old rows.
        """
        if adapter not in FILE_ADAPTERS:
            return None
        return self._download_from_storage(
            dataset_path, conversation, max_bytes=DATASET_FILE_MAX_BYTES
        )

    def _create_project(
        self,
        *,
        action: LabelStudioProjectAction,
        dataset_path: str,
        config_hash: str,
        project_ref: str,
        artifact_dir: str,
        xml_content: bytes,
        media_signer: PortalMediaSigner | None,
        conversation: BaseConversation | None,
        field_map: FieldMap,
        field_map_hash: str,
        dataset_content: bytes | None,
    ) -> tuple[ConvertedManifest, ProjectState]:
        """Convert the dataset, persist its artifacts, and create the LS project."""
        self._upload_to_storage(
            f"{artifact_dir}/label_config.xml",
            xml_content,
            conversation,
        )

        converter = AVITrainToLabelStudioConverter(
            dataset_path=dataset_path,
            storage_base_url=self._storage_base_url,
            storage_headers=self._resolved_storage_headers(conversation),
            batch_task_limit=self._batch_size,
            config_hash=config_hash,
            timeout=float(self._timeout),
            adapter=action.adapter,
            media_signer=media_signer,
            field_map=field_map,
            field_map_hash=field_map_hash,
            field_map_path=action.field_map_path,
            dataset_content=dataset_content,
        )
        manifest = converter.convert()
        self._upload_manifest(artifact_dir, manifest, conversation)

        ls_api = self._build_ls_api(conversation)
        title = f"pyromind_{project_ref[:8]}"
        # The title is derived from the deterministic ref, so an attempt that died
        # after Label Studio stored the project can adopt it instead of orphaning it.
        ls_project = ls_api.find_project_by_title(title)
        if ls_project is None:
            ls_project = ls_api.create_project(
                title=title,
                label_config=xml_content.decode("utf-8"),
            )
        raw_id = ls_project.get("id")
        if not isinstance(raw_id, int):
            raise LabelStudioAPIError(
                f"Label Studio create_project response missing integer id: {ls_project}"
            )
        # Annotators see the export target in the project's settings, and the
        # deployment's export button reads it back to decide where to push.
        ls_api.update_project_description(
            project_id=raw_id,
            description=_project_description(
                project_ref,
                ticket=self._export_ticket(project_ref, conversation),
            ),
        )

        state = ProjectState(
            project_ref=project_ref,
            project_id=raw_id,
            dataset_path=dataset_path,
            adapter=action.adapter,
            status="IMPORTING",
            total_tasks=manifest.total_tasks,
            total_batches=len(manifest.batch_payloads),
            config_hash=config_hash,
            field_map_hash=field_map_hash,
            # Kept verbatim rather than re-read from field_map_path: export has
            # to read controls back through the names that produced the tasks,
            # and the file may have been edited or deleted since.
            field_map=field_map.model_dump() if field_map_hash else None,
            idempotency_key=action.idempotency_key,
            created_at=datetime.now(UTC).isoformat(),
            media_expires_at=_media_expiry_iso(
                media_signer.expires_in if media_signer else None
            ),
        )
        self._save_state(artifact_dir, state, conversation)
        return manifest, state

    def _handle_get(
        self,
        action: LabelStudioProjectAction,
        conversation: BaseConversation | None,
    ) -> LabelStudioProjectObservation:
        if not action.project_ref:
            raise ValueError("project_ref is required for operation='get'.")
        state = self._load_state(action.project_ref, conversation)
        if state is None:
            raise ValueError(f"Project not found: {action.project_ref}")
        return self._state_to_observation("get", state)

    def _handle_status(
        self,
        action: LabelStudioProjectAction,
        conversation: BaseConversation | None,
    ) -> LabelStudioProjectObservation:
        if not action.project_ref:
            raise ValueError("project_ref is required for operation='status'.")
        state = self._load_state(action.project_ref, conversation)
        if state is None:
            raise ValueError(f"Project not found: {action.project_ref}")

        ls_api = self._build_ls_api(conversation)
        try:
            ls_project = ls_api.get_project(state.project_id)
            task_count = int(ls_project.get("task_number", 0))
        except LabelStudioAPIError:
            task_count = None

        obs = self._state_to_observation("status", state)
        # Observations are frozen models, so the live count replaces the persisted one
        # on a copy rather than by assignment.
        return obs.model_copy(update={"task_count": task_count})

    def _handle_update_config(
        self,
        action: LabelStudioProjectAction,
        conversation: BaseConversation | None,
    ) -> LabelStudioProjectObservation:
        if not action.project_ref:
            raise ValueError("project_ref is required for operation='update_config'.")
        if not action.label_config_path:
            raise ValueError(
                "label_config_path is required for operation='update_config'."
            )

        state = self._load_state(action.project_ref, conversation)
        if state is None:
            raise ValueError(f"Project not found: {action.project_ref}")

        if (
            action.expected_config_version is not None
            and action.expected_config_version != state.config_version
        ):
            raise ValueError(
                f"Config version mismatch: expected {state.config_version}, "
                f"got {action.expected_config_version}. Call get first."
            )

        xml_content = self._read_workspace_file(action.label_config_path, conversation)
        if xml_content is None:
            raise ValueError(
                f"label_config.xml not found in workspace: {action.label_config_path}"
            )
        new_xml = xml_content.decode("utf-8")

        ls_api = self._build_ls_api(conversation)
        # A project keeps the bindings it was converted through, so a new config
        # still has to satisfy them: renaming or dropping a control here would
        # silently stop the stored pre-annotations from landing anywhere.
        validate_label_config_xml(new_xml, field_map=self._bindings_for(state))
        ls_api.validate_config(project_id=state.project_id, label_config=new_xml)

        existing_from_names = ls_api.get_annotation_from_names(state.project_id)
        new_from_names = extract_control_names(new_xml)
        removed = existing_from_names - new_from_names
        if removed:
            raise ValueError(
                f"Cannot remove controls used by existing annotations: "
                f"{sorted(removed)}. Create a new project instead."
            )

        ls_api.update_project_config(project_id=state.project_id, label_config=new_xml)
        state.config_version += 1
        artifact_dir = f"{PYROMIND_AGENT_STORAGE_ROOT}/label-studio/{state.project_ref}"
        self._save_state(artifact_dir, state, conversation)
        self._upload_to_storage(
            f"{artifact_dir}/label_config.xml", xml_content, conversation
        )

        return self._state_to_observation("update_config", state)

    def _handle_export(
        self,
        action: LabelStudioProjectAction,
        conversation: BaseConversation | None,
    ) -> LabelStudioProjectObservation:
        if not action.project_ref:
            raise ValueError("project_ref is required for operation='export'.")
        state = self._load_state(action.project_ref, conversation)
        if state is None:
            raise ValueError(f"Project not found: {action.project_ref}")

        ls_api = self._build_ls_api(conversation)
        export_data = ls_api.export_annotations(state.project_id)
        converter = LabelStudioToAVITrainConverter(self._export_field_map(state))
        samples = converter.convert(export_data)

        output_path = (
            _normalize_storage_path(action.output_path, "output_path")
            if action.output_path
            else (
                f"{PYROMIND_AGENT_STORAGE_ROOT}/label-studio/{state.project_ref}/export"
            )
        )
        payload = json.dumps(samples, ensure_ascii=False).encode("utf-8")
        self._upload_to_storage(
            f"{output_path}/annotations.json", payload, conversation
        )
        # Best effort: the export is already stored, so a stale "last export" note in
        # the project description is not worth failing an otherwise good export.
        try:
            ls_api.update_project_description(
                project_id=state.project_id,
                description=_project_description(
                    state.project_ref,
                    ticket=self._export_ticket(state.project_ref, conversation),
                    exported=(len(samples), state.total_tasks),
                ),
            )
        except LabelStudioAPIError:
            pass

        return LabelStudioProjectObservation.from_text(
            text=(
                f"Label Studio export finished: project_ref={state.project_ref} "
                f"project_id={state.project_id} "
                f"samples={len(samples)}/{len(export_data)} "
                f"export_path={output_path}/annotations.json"
            ),
            operation="export",
            project_ref=state.project_ref,
            project_id=state.project_id,
            status="EXPORTED",
            task_count=len(export_data),
            annotation_count=len(samples),
            export_path=f"{output_path}/annotations.json",
        )

    def _export_field_map(self, state: ProjectState) -> FieldMap | None:
        """Bindings to read a project's annotations back through.

        A declared map is rebuilt from the state that created the tasks, so
        export looks controls up under exactly the names the import wrote them
        with. Built-in bindings return None, which lets the converter fall back
        to the default reverse index that spans every adapter.
        """
        if not state.field_map:
            return None
        return FieldMap.model_validate(state.field_map)

    def _bindings_for(self, state: ProjectState) -> FieldMap:
        """The bindings a project's tasks were converted through.

        Unlike export's reverse index, this always names a concrete map, so a
        caller checking a config against a project does not have to treat the
        built-in bindings as a separate case.
        """
        return self._export_field_map(state) or default_field_map(state.adapter)

    def _handle_refresh_media(
        self,
        action: LabelStudioProjectAction,
        conversation: BaseConversation | None,
    ) -> LabelStudioProjectObservation:
        """Re-sign the media URLs baked into task data before they stop working.

        Label Studio keeps whatever URL the import produced and never asks for a
        new one, so a project that outlives the portal's media URL lifetime shows
        broken images until its tasks are written back with fresh URLs.
        """
        if not action.project_ref:
            raise ValueError("project_ref is required for operation='refresh_media'.")
        state = self._load_state(action.project_ref, conversation)
        if state is None:
            raise ValueError(f"Project not found: {action.project_ref}")

        signer = self._build_media_signer(conversation)
        if signer is None:
            raise ValueError(
                "refresh_media requires a portal base URL: the media route that "
                "signs these URLs lives there."
            )

        ls_api = self._build_ls_api(conversation)
        tasks = ls_api.list_project_tasks(state.project_id)
        paths = sorted(
            {path for task in tasks for _, path in _media_fields(_task_data(task))}
        )
        if not paths:
            raise ValueError(
                "No media paths found in the task data of this project, so its "
                "URLs cannot be re-signed. Create the project again."
            )

        urls = signer.sign_many(paths)
        refreshed, failed, first_error = self._refresh_task_media(tasks, urls, ls_api)
        if refreshed == 0:
            return self._error(
                "refresh_media",
                f"Re-signing media URLs failed for all {failed} tasks: {first_error}",
                state,
            )

        artifact_dir = f"{PYROMIND_AGENT_STORAGE_ROOT}/label-studio/{state.project_ref}"
        state.media_expires_at = _media_expiry_iso(signer.expires_in)
        self._save_state(artifact_dir, state, conversation)

        summary = (
            f"Label Studio media URLs re-signed: project_ref={state.project_ref} "
            f"project_id={state.project_id} refreshed={refreshed}/{len(tasks)}"
        )
        if failed:
            summary += f" failed={failed} first_error={first_error}"
        return self._state_to_observation("refresh_media", state, summary=summary)

    def _refresh_task_media(
        self,
        tasks: list[dict[str, Any]],
        urls: dict[str, str],
        ls_api: LabelStudioAPIClient,
    ) -> tuple[int, int, str]:
        """Write fresh media URLs back into each task's data, concurrently.

        Label Studio updates one task per request, so a large project takes as
        many requests as it has tasks. Failures stay per task: the tasks that
        were written keep working, and running the operation again retries the
        rest, because signing a URL twice is harmless.
        """
        pending: list[tuple[int, dict[str, Any]]] = []
        for task in tasks:
            task_id = task.get("id")
            data, changed = _resigned_data(_task_data(task), urls)
            if isinstance(task_id, int) and changed:
                pending.append((task_id, data))

        refreshed = 0
        first_error = ""
        with ThreadPoolExecutor(max_workers=_MEDIA_REFRESH_WORKERS) as pool:
            futures = {
                pool.submit(
                    ls_api.update_task_data, task_id=task_id, data=data
                ): task_id
                for task_id, data in pending
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except LabelStudioAPIError as exc:
                    if not first_error:
                        first_error = f"task {futures[future]}: {exc}"
                    continue
                refreshed += 1
        return refreshed, len(pending) - refreshed, first_error

    def _iter_pending_batches(
        self,
        artifact_dir: str,
        state: ProjectState,
        manifest: ConvertedManifest | None,
        conversation: BaseConversation | None,
    ) -> Iterator[tuple[int, list[dict[str, Any]]]]:
        """Yield the batches a resumed import still has to send, in order.

        A fresh create already holds the payloads in memory; a resumed one reads
        them back from the manifest so retrying never re-converts the dataset.
        """
        if manifest is not None:
            for index, (_, payload, _) in enumerate(manifest.batch_payloads):
                if index >= state.next_batch:
                    yield index, json.loads(payload.decode("utf-8"))
            return
        if state.total_batches <= 0:
            raise ValueError(
                f"Persisted state for {state.project_ref} records no batches; "
                "cannot resume the import."
            )
        for index in range(state.next_batch, state.total_batches):
            payload = self._download_from_storage(
                f"{artifact_dir}/manifests/tasks-{index + 1:05d}.json", conversation
            )
            yield index, json.loads(payload.decode("utf-8"))

    def _open_url(self, project_id: int) -> str:
        """Portal SSO entry point, so the browser lands on a project it can open.

        The portal signs the browser in to Label Studio on the way through, which
        the bare project link cannot do on its own.
        """
        if self._portal_base_url:
            return (
                f"{self._portal_base_url}/label_studio/sso"
                f"?target={self._project_path(project_id)}"
            )
        return self._project_url(project_id)

    def _project_path(self, project_id: int) -> str:
        return f"/projects/{project_id}/data"

    def _project_url(self, project_id: int) -> str:
        return f"{self._ls_base_url}{self._project_path(project_id)}"

    def _build_media_signer(
        self, conversation: BaseConversation | None
    ) -> PortalMediaSigner | None:
        if not self._portal_base_url:
            return None
        return PortalMediaSigner(
            portal_base_url=self._portal_base_url,
            headers=self._resolved_storage_headers(conversation),
            cluster=self._cluster,
            timeout=float(self._timeout),
        )

    def _build_export_ticket_provider(
        self, conversation: BaseConversation | None
    ) -> PortalExportTicketProvider | None:
        if not self._portal_base_url:
            return None
        return PortalExportTicketProvider(
            portal_base_url=self._portal_base_url,
            headers=self._resolved_storage_headers(conversation),
            cluster=self._cluster,
            timeout=float(self._timeout),
        )

    def _export_ticket(
        self, project_ref: str, conversation: BaseConversation | None
    ) -> str:
        """Mint a fresh export ticket, or nothing at all if the portal cannot.

        Every description write asks for a new ticket because the old one expires;
        a project whose ticket has lapsed simply stops pushing its export. The
        ticket is a capability, so it travels to the caller as an opaque string and
        is never logged.
        """
        provider = self._build_export_ticket_provider(conversation)
        if provider is None:
            return ""
        try:
            return provider.fetch(project_ref)
        except ConversionError as exc:
            logger.warning(
                "Label Studio export ticket unavailable for %s: %s", project_ref, exc
            )
            return ""

    def _download_from_storage(
        self,
        storage_path: str,
        conversation: BaseConversation | None,
        *,
        max_bytes: int = _MANIFEST_BATCH_MAX_BYTES,
    ) -> bytes:
        return download_file_from_pyromind(
            storage_path=storage_path,
            storage_base_url=self._storage_base_url,
            headers=self._resolved_storage_headers(conversation),
            timeout=float(self._timeout),
            max_bytes=max_bytes,
        )

    def _build_ls_api(
        self, conversation: BaseConversation | None
    ) -> LabelStudioAPIClient:
        if conversation is None:
            raise ValueError("label_studio_project requires an active conversation.")
        state = cast("ConversationState", conversation.state)
        return LabelStudioAPIClient(
            base_url=self._ls_base_url,
            token=self._ls_token(conversation, state),
            timeout=float(self._timeout),
        )

    def _ls_token(
        self, conversation: BaseConversation, state: ConversationState
    ) -> str:
        """The caller's own token, read from the portal when one is configured.

        Label Studio shows an account only its own organization's projects, so a
        token shared by every conversation files each user's projects under one
        account, where the user who asked for them cannot see them. The shared
        secret stays as the fallback for deployments that hold a single token.
        """
        if self._portal_base_url:
            return PortalTokenProvider(
                portal_base_url=self._portal_base_url,
                headers=self._resolved_storage_headers(conversation),
                timeout=float(self._timeout),
            ).fetch()
        token = state.secret_registry.get_secret_value(self._ls_token_secret)
        if not token:
            raise ValueError(
                f"Label Studio token secret '{self._ls_token_secret}' is not "
                "configured. Check server-side tool wiring."
            )
        return token

    def _resolved_storage_headers(
        self, conversation: BaseConversation | None
    ) -> dict[str, str]:
        from openhands.tools.pyromind_dataset.definition import (
            _resolve_conversation_headers,
            _resolve_secret_headers,
        )

        headers = {"accept": "*/*", **self._headers}
        if conversation is not None:
            headers.update(_resolve_conversation_headers(conversation))
            headers.update(_resolve_secret_headers(conversation, self._secret_headers))
        return headers

    def _read_workspace_file(
        self, relative_path: str, conversation: BaseConversation | None
    ) -> bytes | None:
        if conversation is None:
            return None
        workspace = cast(Any, conversation).workspace
        file_path = Path(workspace.working_dir) / relative_path
        if not file_path.is_file():
            return None
        return file_path.read_bytes()

    def _upload_to_storage(
        self,
        storage_path: str,
        content: bytes,
        conversation: BaseConversation | None,
    ) -> None:
        if conversation is None:
            return
        target_dir = str(PurePosixPath(storage_path).parent)
        filename = PurePosixPath(storage_path).name
        # Storage names the object after the local file, so the temp file has to
        # carry the real name inside a temp directory. A suffixed temp name
        # uploads a file nothing can ever read back.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / filename
            tmp_path.write_bytes(content)
            upload_local_file_to_pyromind(
                local_path=tmp_path,
                target_dir=target_dir,
                storage_base_url=self._storage_base_url,
                headers=self._resolved_storage_headers(conversation),
                timeout=float(self._timeout),
            )

    def _upload_manifest(
        self,
        artifact_dir: str,
        manifest: ConvertedManifest,
        conversation: BaseConversation | None,
    ) -> None:
        manifests_dir = f"{artifact_dir}/manifests"
        for batch_path, payload, _ in manifest.batch_payloads:
            self._upload_to_storage(
                f"{manifests_dir}/{batch_path}", payload, conversation
            )
        manifest.project_ref = self._extract_ref(artifact_dir)
        manifest_json = manifest.to_manifest_data().model_dump_json(
            exclude_none=True, indent=2
        )
        self._upload_to_storage(
            f"{manifests_dir}/manifest.json",
            manifest_json.encode("utf-8"),
            conversation,
        )

    def _extract_ref(self, artifact_dir: str) -> str:
        return PurePosixPath(artifact_dir).name

    def _save_state(
        self,
        artifact_dir: str,
        state: ProjectState,
        conversation: BaseConversation | None,
    ) -> None:
        self._upload_to_storage(
            f"{artifact_dir}/project_state.json",
            state.model_dump_json(exclude_none=True, indent=2).encode("utf-8"),
            conversation,
        )

    def _load_state(
        self, project_ref: str, conversation: BaseConversation | None
    ) -> ProjectState | None:
        """Read persisted state, or None only when it genuinely does not exist.

        A storage outage must not read as "project not found": that makes a valid
        project reference look invalid and invites a duplicate create.
        """
        if conversation is None:
            return None
        storage_path = (
            f"{PYROMIND_AGENT_STORAGE_ROOT}/label-studio/{project_ref}/"
            "project_state.json"
        )
        try:
            content = download_file_from_pyromind(
                storage_path=storage_path,
                storage_base_url=self._storage_base_url,
                headers=self._resolved_storage_headers(conversation),
                timeout=float(self._timeout),
                max_bytes=_PROJECT_STATE_MAX_BYTES,
            )
        except StorageFileNotFoundError:
            return None
        try:
            return ProjectState.model_validate_json(content.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Persisted state for {project_ref} is unreadable: {exc}"
            ) from exc

    def _state_to_observation(
        self,
        operation: str,
        state: ProjectState,
        *,
        summary: str | None = None,
    ) -> LabelStudioProjectObservation:
        if summary is None:
            summary = (
                f"Label Studio project {state.project_ref} "
                f"(project_id={state.project_id}) operation={operation} "
                f"status={state.status} config_version={state.config_version} "
                f"tasks={state.imported_count}/{state.total_tasks}"
            )
            if state.last_error:
                summary += f" last_error={state.last_error}"
        return LabelStudioProjectObservation.from_text(
            text=summary,
            operation=operation,
            project_ref=state.project_ref,
            project_id=state.project_id,
            status=state.status,
            config_version=state.config_version,
            task_count=state.total_tasks,
            imported_count=state.imported_count,
            next_batch=state.next_batch,
            last_error=state.last_error,
            manifest_path=(
                f"{PYROMIND_AGENT_STORAGE_ROOT}/label-studio/{state.project_ref}"
                "/manifests/manifest.json"
            ),
            open_url=self._open_url(state.project_id),
            project_url=self._project_url(state.project_id),
        )

    def _error(
        self,
        operation: str,
        message: str,
        state: ProjectState | None = None,
    ) -> LabelStudioProjectObservation:
        """Report a failure, keeping any project identity the caller needs.

        Dropping the reference on failure strands a project that already exists
        in Label Studio, so the agent re-creates it instead of resuming.
        """
        return LabelStudioProjectObservation.from_text(
            text=message,
            operation=operation,
            is_error=True,
            project_ref=state.project_ref if state else "",
            project_id=state.project_id if state else None,
            status=state.status if state else "",
            task_count=state.total_tasks if state else None,
            imported_count=state.imported_count if state else None,
            next_batch=state.next_batch if state else None,
            last_error=state.last_error if state else None,
            open_url=self._open_url(state.project_id) if state else None,
            project_url=self._project_url(state.project_id) if state else None,
        )


def _task_data(task: dict[str, Any]) -> dict[str, Any]:
    data = task.get("data")
    return data if isinstance(data, dict) else {}


def _media_fields(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Pair each rendered media URL field with the storage path behind it.

    The converter writes both halves of a sample's images, so a project that
    outlives its signed URLs can be repaired from the paths alone, without
    reading any object back out of storage.
    """
    pairs: list[tuple[str, str]] = []
    for key, value in data.items():
        if not key.endswith(_MEDIA_PATH_SUFFIX) or not isinstance(value, str):
            continue
        field = key[: -len(_MEDIA_PATH_SUFFIX)]
        if field and value:
            pairs.append((field, value))
    return pairs


def _resigned_data(
    data: dict[str, Any], urls: dict[str, str]
) -> tuple[dict[str, Any], bool]:
    """Copy task data, replacing every baked media URL with a fresh one.

    Label Studio replaces the whole data object on update, so the caller sends
    the copy in full and keeps the fields it did not touch.
    """
    refreshed = dict(data)
    changed = False
    for field, path in _media_fields(data):
        url = urls.get(path)
        if url is None:
            continue
        refreshed[field] = url
        changed = True
    return refreshed, changed


def _media_expiry_iso(expires_in: int | None) -> str | None:
    """Record the window the portal reported, for diagnostics only.

    The portal keeps serving a media token past its own `exp` claim -- Label
    Studio cannot re-mint the URL it stored, so the token is bound to one user
    and object instead of a deadline. Nothing warns on this window, and
    refresh_media remains available for callers that want a fresh signature.
    """
    if expires_in is None:
        return None
    return (datetime.now(UTC) + timedelta(seconds=expires_in)).isoformat()


def _project_description(
    project_ref: str,
    *,
    ticket: str,
    exported: tuple[int, int] | None = None,
) -> str:
    """Write down where this project's export lands, for the annotator to see.

    The Label Studio deployment reads the path line back out of the description to
    decide where the project's export button pushes the file, and the ticket line is
    the capability that authorises the push: it is limited to writing that one
    object key for this project's owner, so the deployment needs no storage
    credential of its own. Both lines stay machine-readable; see app-deploy.yaml's
    plugin, which parses the ticket by prefix and therefore gets it bare.
    """
    lines = [
        _PROJECT_DESCRIPTION_HEADER,
        f"{PYROMIND_AGENT_STORAGE_ROOT}/label-studio/{project_ref}/export/"
        f"{EXPORT_OBJECT_NAME}",
    ]
    if ticket:
        lines.append(f"export-token: {ticket}")
    if exported is not None:
        annotated, total = exported
        lines.append(
            f"最后导出 / last export: "
            f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · "
            f"samples={annotated}/{total}"
        )
    return "\n".join(lines)


def _normalize_storage_path(value: str, field_name: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError(f"{field_name} must be a non-empty storage path.")
    if any(ord(character) < 32 for character in raw):
        raise ValueError(f"{field_name} contains control characters.")
    # Callers quote platform workspace paths (`/workspace/exports/...`) as readily
    # as storage paths, and both name the same objects.
    parts = [
        part
        for part in _strip_workspace_prefix(raw).split("/")
        if part not in {"", "."}
    ]
    if not parts or ".." in parts:
        raise ValueError(f"{field_name} must not be the root or contain '..'.")
    return "/" + "/".join(parts)
