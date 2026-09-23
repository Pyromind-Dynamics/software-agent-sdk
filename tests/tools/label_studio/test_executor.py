"""Tests for LabelStudioProjectExecutor."""

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openhands.tools.label_studio.api_client import LabelStudioAPIError
from openhands.tools.label_studio.converter import (
    DATASET_FILE_MAX_BYTES,
    ConversionError,
)
from openhands.tools.label_studio.definition import (
    LabelStudioProjectAction,
)
from openhands.tools.label_studio.executor import (
    LabelStudioProjectExecutor,
    _normalize_storage_path,
)
from openhands.tools.label_studio.export_ticket import PortalExportTicketProvider
from openhands.tools.label_studio.models import ProjectState
from openhands.tools.label_studio.skill_helpers import extract_control_values
from openhands.tools.label_studio.token_provider import PortalTokenProvider
from openhands.tools.pyromind_dataset.definition import StorageFileNotFoundError


VALID_XML = (
    '<View><Image name="defect_image" value="$defect_image"/>'
    '<Choices name="quality_label" toName="defect_image">'
    '<Choice value="ok"/><Choice value="defect"/>'
    "</Choices>"
    '<RectangleLabels name="finding_category" toName="defect_image">'
    '<Label value="开路"/></RectangleLabels>'
    '<TextArea name="finding_observation" toName="defect_image"/>'
    "</View>"
)


@pytest.fixture()
def workspace_xml(tmp_path: Path) -> Path:
    xml_path = tmp_path / "label_config.xml"
    xml_path.write_text(VALID_XML, encoding="utf-8")
    return xml_path


@pytest.fixture()
def conversation(tmp_path: Path, workspace_xml: Path) -> MagicMock:
    conv = MagicMock()
    conv.id = "test-conversation-id"
    conv.workspace.working_dir = str(tmp_path)
    conv.state.secret_registry.get_secret_value.return_value = "test-token"
    return conv


@pytest.fixture()
def executor() -> LabelStudioProjectExecutor:
    return LabelStudioProjectExecutor(
        ls_base_url="http://ls.example.com",
        ls_token_secret="LABEL_STUDIO_TOKEN",
        storage_base_url="http://storage.example.com",
    )


def _mock_ls_api(
    *, project_id: int = 42, export_data: list[dict] | None = None
) -> MagicMock:
    api = MagicMock()
    api.find_project_by_title.return_value = None
    api.create_project.return_value = {"id": project_id}
    api.get_project.return_value = {"id": project_id, "task_number": 5}
    api.export_annotations.return_value = export_data or []
    api.get_annotation_from_names.return_value = {"quality_label"}
    return api


def _action(**kwargs: Any) -> LabelStudioProjectAction:
    defaults: dict[str, Any] = {"operation": "create"}
    defaults.update(kwargs)
    return LabelStudioProjectAction(**defaults)


class TestCreate:
    def test_missing_dataset_path(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        obs = executor(
            _action(operation="create", label_config_path="label_config.xml"),
            conversation,
        )
        assert obs.is_error
        assert "dataset_path is required" in obs.text

    def test_missing_label_config_path(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        obs = executor(
            _action(operation="create", dataset_path="/datasets/pcb"),
            conversation,
        )
        assert obs.is_error
        assert "label_config_path is required" in obs.text

    def test_missing_xml_file(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        obs = executor(
            _action(
                operation="create",
                dataset_path="/datasets/pcb",
                label_config_path="nonexistent.xml",
            ),
            conversation,
        )
        assert obs.is_error
        assert "not found" in obs.text

    def test_create_success(
        self,
        conversation: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        executor = _create_executor(portal_base_url=PORTAL)
        monkeypatch.setattr(
            PortalExportTicketProvider, "fetch", lambda self, project_ref: TICKET
        )
        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(
                executor, "_upload_to_storage", return_value=None
            ) as mock_upload,
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
        ):
            mock_manifest = MockConverter.return_value.convert.return_value
            mock_manifest.total_tasks = 3
            mock_manifest.batch_payloads = [("tasks-00001.json", b"[{},{},{}]", 3)]
            mock_manifest.unmapped_quality = ()
            mock_manifest.unmatched_regions = ()
            mock_manifest.unlisted_values = {}
            mock_manifest.to_manifest_data.return_value.model_dump_json.return_value = (
                "{}"
            )

            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert not obs.is_error
        assert obs.project_id == 42
        assert obs.status == "READY"
        assert obs.imported_count == 3
        assert mock_ls.create_project.called
        assert mock_ls.import_tasks.called
        assert mock_upload.called
        assert obs.project_ref is not None
        # The LLM-facing content must summarize the result; an empty content
        # list makes the tool return a blank output to the agent.
        assert "project_ref=" in obs.text
        assert "tasks=3/3" in obs.text

        # The project advertises where its export button will push the file, and
        # the portal accepts a write to exactly that key and nothing else.
        assert mock_ls.update_project_description.call_args.kwargs["project_id"] == 42
        description = mock_ls.update_project_description.call_args.kwargs["description"]
        path_line = next(
            line for line in description.splitlines() if line.startswith("/")
        )
        assert re.fullmatch(
            r"/\.pyromind-agent/label-studio/[A-Za-z0-9._-]+/export/"
            r"label_studio_export\.json",
            path_line,
        )
        # The ticket line follows the path line so the deployment's plugin can
        # read the capability that authorises the push without holding one itself.
        lines = description.splitlines()
        assert lines[lines.index(path_line) + 1] == f"export-token: {TICKET}"

    def test_create_survives_a_portal_that_cannot_mint_a_ticket(
        self,
        conversation: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Annotators must never be blocked by a portal outage."""
        executor = _create_executor(portal_base_url=PORTAL)

        def _unreachable(self, project_ref: str) -> str:
            raise ConversionError(f"Portal export-token API {PORTAL} unreachable")

        monkeypatch.setattr(PortalExportTicketProvider, "fetch", _unreachable)
        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch.object(executor, "_save_state", return_value=None),
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
            caplog.at_level(
                logging.WARNING, logger="openhands.tools.label_studio.executor"
            ),
        ):
            _mock_manifest(MockConverter, 1, b"[{}]")
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert not obs.is_error
        assert obs.status == "READY"
        description = mock_ls.update_project_description.call_args.kwargs["description"]
        assert "export-token:" not in description
        assert any(
            line.startswith("/.pyromind-agent/label-studio/")
            for line in description.splitlines()
        )
        assert "export ticket unavailable" in caplog.text

    def test_create_survives_a_portal_url_that_is_malformed(
        self,
        conversation: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A mistyped portal_base_url must degrade, not abandon the project.

        httpx answers a malformed base URL with InvalidURL, which is not a
        RequestError. Unconverted it would sail past the executor's except
        clause, error the tool out, and leave a created project unreachable
        behind a ProjectState that was never saved.
        """
        executor = _create_executor(portal_base_url="http://portal:notaport")

        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch.object(executor, "_save_state", return_value=None),
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
            caplog.at_level(
                logging.WARNING, logger="openhands.tools.label_studio.executor"
            ),
        ):
            _mock_manifest(MockConverter, 1, b"[{}]")
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert not obs.is_error
        assert obs.status == "READY"
        description = mock_ls.update_project_description.call_args.kwargs["description"]
        assert "export-token:" not in description
        assert "export ticket unavailable" in caplog.text

    def test_invalid_xml_rejected(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
        tmp_path: Path,
    ) -> None:
        bad_xml_path = tmp_path / "bad.xml"
        bad_xml_path.write_text("<View><broken>", encoding="utf-8")
        obs = executor(
            _action(
                operation="create",
                dataset_path="/datasets/pcb",
                label_config_path="bad.xml",
            ),
            conversation,
        )
        assert obs.is_error
        assert "Malformed XML" in obs.text

    def test_config_outside_the_adapter_contract_is_rejected(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A config missing the converter's controls fails instead of importing blind.

        Predictions are written against fixed control names, so a config that
        renames them produces a project with no pre-annotations and no error at
        all -- the failure only shows up as an empty review screen.
        """
        contract_xml = tmp_path / "renamed.xml"
        contract_xml.write_text(
            '<View><Image name="defect_image" value="$defect_image"/>'
            '<Choices name="verdict" toName="defect_image">'
            '<Choice value="ok"/><Choice value="defect"/></Choices></View>',
            encoding="utf-8",
        )
        obs = executor(
            _action(
                operation="create",
                dataset_path="/datasets/pcb",
                label_config_path="renamed.xml",
            ),
            conversation,
        )
        assert obs.is_error
        assert "prediction contract" in obs.text
        assert "quality_label" in obs.text

    def test_official_validation_runs_before_the_dataset_is_converted(
        self,
        conversation: MagicMock,
    ) -> None:
        """Label Studio's validator is asked before the expensive conversion.

        Converting downloads every sample's meta file and signs every media URL,
        so a config Label Studio rejects has to fail before any of that starts
        instead of surfacing at create_project.
        """
        executor = _create_executor(portal_base_url=PORTAL)
        mock_ls = _mock_ls_api()
        mock_ls.validate_config_standalone.side_effect = LabelStudioAPIError(
            "HTTP 400: invalid label config"
        )
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
        ):
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert obs.is_error
        assert "rejected" in obs.text
        assert mock_ls.validate_config_standalone.called
        assert mock_ls.validate_config_standalone.call_args.kwargs == {
            "label_config": VALID_XML
        }
        # Nothing downstream may have started: no conversion, no project.
        assert MockConverter.call_count == 0
        assert mock_ls.create_project.called is False


class TestGet:
    def test_missing_project_ref(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        obs = executor(_action(operation="get"), conversation)
        assert obs.is_error
        assert "project_ref is required" in obs.text

    def test_project_not_found(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        with patch.object(executor, "_load_state", return_value=None):
            obs = executor(_action(operation="get", project_ref="ghost"), conversation)
        assert obs.is_error
        assert "not found" in obs.text

    def test_get_success(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
    ) -> None:
        from openhands.tools.label_studio.models import ProjectState

        state = ProjectState(
            project_ref="abc",
            project_id=42,
            dataset_path="/datasets/pcb",
            adapter="avi_train",
            status="READY",
            total_tasks=10,
            imported_count=10,
        )
        with patch.object(executor, "_load_state", return_value=state):
            obs = executor(_action(operation="get", project_ref="abc"), conversation)
        assert not obs.is_error
        assert obs.project_id == 42
        assert obs.status == "READY"


class TestStatus:
    """operation='status' reports the task count Label Studio has right now.

    That count comes from Label Studio rather than from the persisted state, so it
    has to be written into an observation the shared helper has already built.
    """

    @staticmethod
    def _state() -> ProjectState:
        return ProjectState(
            project_ref="abc",
            project_id=42,
            dataset_path="/datasets/pcb",
            adapter="avi_train",
            status="READY",
            total_tasks=10,
            imported_count=10,
        )

    def test_missing_project_ref(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        obs = executor(_action(operation="status"), conversation)
        assert obs.is_error
        assert "project_ref is required" in obs.text

    def test_project_not_found(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        with patch.object(executor, "_load_state", return_value=None):
            obs = executor(
                _action(operation="status", project_ref="ghost"), conversation
            )
        assert obs.is_error
        assert "not found" in obs.text

    def test_reports_the_count_label_studio_has_now(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
    ) -> None:
        mock_ls = _mock_ls_api()
        mock_ls.get_project.return_value = {"id": 42, "task_number": 684}
        with (
            patch.object(executor, "_load_state", return_value=self._state()),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
        ):
            obs = executor(_action(operation="status", project_ref="abc"), conversation)
        assert not obs.is_error
        assert obs.operation == "status"
        # 684 is Label Studio's count, not the 10 in the persisted state.
        assert obs.task_count == 684

    def test_falls_back_when_label_studio_does_not_answer(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
    ) -> None:
        mock_ls = _mock_ls_api()
        mock_ls.get_project.side_effect = LabelStudioAPIError("unreachable")
        with (
            patch.object(executor, "_load_state", return_value=self._state()),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
        ):
            obs = executor(_action(operation="status", project_ref="abc"), conversation)
        assert not obs.is_error
        assert obs.task_count is None


class TestUpdateConfig:
    def test_version_mismatch(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
        workspace_xml: Path,
    ) -> None:
        from openhands.tools.label_studio.models import ProjectState

        state = ProjectState(
            project_ref="abc", project_id=42, dataset_path="/d", adapter="avi_train"
        )
        with patch.object(executor, "_load_state", return_value=state):
            obs = executor(
                _action(
                    operation="update_config",
                    project_ref="abc",
                    label_config_path="label_config.xml",
                    expected_config_version=5,
                ),
                conversation,
            )
        assert obs.is_error
        assert "version mismatch" in obs.text.lower()

    def test_removing_annotated_control_rejected(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
    ) -> None:
        from openhands.tools.label_studio.models import ProjectState

        state = ProjectState(
            project_ref="abc", project_id=42, dataset_path="/d", adapter="avi_train"
        )
        mock_ls = _mock_ls_api()
        mock_ls.get_annotation_from_names.return_value = {"removed_control"}
        with (
            patch.object(executor, "_load_state", return_value=state),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
        ):
            obs = executor(
                _action(
                    operation="update_config",
                    project_ref="abc",
                    label_config_path="label_config.xml",
                    expected_config_version=1,
                ),
                conversation,
            )
        assert obs.is_error
        assert "removed_control" in obs.text
        assert not mock_ls.update_project_config.called

    def test_config_dropping_a_bound_control_rejected(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
        tmp_path: Path,
    ) -> None:
        # pre-annotations are written through quality_label, so a config that
        # leaves it out would import with that field silently empty.
        stripped = tmp_path / "stripped.xml"
        stripped.write_text(
            '<View><Image name="defect_image" value="$defect_image"/>'
            '<TextArea name="finding_observation" toName="defect_image"/>'
            "</View>",
            encoding="utf-8",
        )
        state = ProjectState(
            project_ref="abc", project_id=42, dataset_path="/d", adapter="avi_train"
        )
        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=state),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
        ):
            obs = executor(
                _action(
                    operation="update_config",
                    project_ref="abc",
                    label_config_path="stripped.xml",
                    expected_config_version=1,
                ),
                conversation,
            )
        assert obs.is_error
        assert "quality_label" in obs.text
        assert not mock_ls.update_project_config.called

    def test_declared_bindings_override_the_defaults(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
    ) -> None:
        from openhands.tools.label_studio.field_map import parse_field_map

        # The project was converted through a map that renamed the quality
        # control, so the default name no longer satisfies its own config.
        declared = parse_field_map(
            {
                "samples": [
                    {"field": "quality", "control": "my_verdict", "type": "choices"}
                ]
            },
            adapter="avi_train",
        )
        state = ProjectState(
            project_ref="abc",
            project_id=42,
            dataset_path="/d",
            adapter="avi_train",
            field_map_hash="deadbeef",
            field_map=declared.model_dump(),
        )
        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=state),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
        ):
            obs = executor(
                _action(
                    operation="update_config",
                    project_ref="abc",
                    label_config_path="label_config.xml",
                    expected_config_version=1,
                ),
                conversation,
            )
        assert obs.is_error
        assert "my_verdict" in obs.text
        assert not mock_ls.update_project_config.called


class TestExport:
    def test_export_success(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
    ) -> None:
        from openhands.tools.label_studio.models import ProjectState

        state = ProjectState(
            project_ref="abc",
            project_id=42,
            dataset_path="/d",
            adapter="avi_train",
            total_tasks=3,
        )
        export_data = [
            {
                "data": {"sample_id": "s1"},
                "annotations": [
                    {
                        "result": [
                            {"from_name": "quality_label", "value": {"choices": ["ok"]}}
                        ]
                    }
                ],
            }
        ]
        mock_ls = _mock_ls_api(export_data=export_data)
        with (
            patch.object(executor, "_load_state", return_value=state),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None) as mock_up,
        ):
            obs = executor(_action(operation="export", project_ref="abc"), conversation)
        assert not obs.is_error
        assert obs.status == "EXPORTED"
        assert obs.annotation_count == 1
        assert mock_up.called
        upload_path = mock_up.call_args[0][0]
        assert upload_path.endswith("/annotations.json")
        uploaded = json.loads(mock_up.call_args[0][1].decode("utf-8"))
        assert uploaded[0]["quality"] == "ok"
        assert mock_ls.update_project_description.call_args.kwargs["project_id"] == 42
        assert (
            "samples=1/3"
            in (mock_ls.update_project_description.call_args.kwargs["description"])
        )

    def test_export_renews_the_export_ticket(
        self,
        conversation: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ticket lapses, so every description write mints a fresh one."""
        state = ProjectState(
            project_ref="abc",
            project_id=42,
            dataset_path="/d",
            adapter="avi_train",
            total_tasks=3,
        )
        mock_ls = _mock_ls_api(export_data=[{"data": {}, "annotations": []}])
        minted: list[str] = []

        def _fetch(self, project_ref: str) -> str:
            minted.append(project_ref)
            return "fresh-ticket"

        monkeypatch.setattr(PortalExportTicketProvider, "fetch", _fetch)
        executor = _create_executor(portal_base_url=PORTAL)
        with (
            patch.object(executor, "_load_state", return_value=state),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None),
        ):
            obs = executor(_action(operation="export", project_ref="abc"), conversation)

        assert not obs.is_error
        description = mock_ls.update_project_description.call_args.kwargs["description"]
        assert "export-token: fresh-ticket" in description.splitlines()
        assert minted == ["abc"]

    def test_export_missing_project(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        with patch.object(executor, "_load_state", return_value=None):
            obs = executor(
                _action(operation="export", project_ref="ghost"), conversation
            )
        assert obs.is_error


class TestErrorHandling:
    def test_ls_api_error_captured(
        self,
        executor: LabelStudioProjectExecutor,
        conversation: MagicMock,
    ) -> None:
        from openhands.tools.label_studio.models import ProjectState

        state = ProjectState(
            project_ref="abc", project_id=42, dataset_path="/d", adapter="avi_train"
        )
        mock_ls = MagicMock()
        mock_ls.export_annotations.side_effect = LabelStudioAPIError("boom")
        with (
            patch.object(executor, "_load_state", return_value=state),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
        ):
            obs = executor(_action(operation="export", project_ref="abc"), conversation)
        assert obs.is_error
        assert "boom" in obs.text


def _create_executor(**overrides: Any) -> LabelStudioProjectExecutor:
    params: dict[str, Any] = {
        "ls_base_url": "http://ls.example.com",
        "ls_token_secret": "LABEL_STUDIO_TOKEN",
        "storage_base_url": "http://storage.example.com",
    }
    params.update(overrides)
    return LabelStudioProjectExecutor(**params)


def _mock_manifest(
    MockConverter: MagicMock,
    tasks: int,
    payload: bytes,
    unmapped_quality: tuple[str, ...] = (),
    unmatched_regions: tuple[str, ...] = (),
    unlisted_values: dict[str, tuple[str, ...]] | None = None,
) -> MagicMock:
    manifest = MockConverter.return_value.convert.return_value
    manifest.total_tasks = tasks
    manifest.batch_payloads = [("tasks-00001.json", payload, tasks)]
    # Set explicitly. An unset attribute reads back as an auto-created MagicMock,
    # which is both truthy and iterable-as-empty -- so the executor's warning
    # branch would be taken and satisfied without a single test intending it.
    manifest.unmapped_quality = unmapped_quality
    manifest.unmatched_regions = unmatched_regions
    manifest.unlisted_values = unlisted_values or {}
    return manifest


def test_create_widens_the_config_with_the_values_the_data_uses(
    conversation: MagicMock,
) -> None:
    """The config is a template: a value it does not list is added, not dropped.

    Label Studio silently ignores a prediction whose value is absent from the
    control's own list, so the value the data carries has to reach the project's
    config. The caller's file is left as written.
    """
    executor = _create_executor()
    mock_ls = _mock_ls_api()
    with (
        patch.object(executor, "_load_state", return_value=None),
        patch.object(executor, "_build_ls_api", return_value=mock_ls),
        patch.object(executor, "_upload_to_storage", return_value=None),
        patch.object(executor, "_save_state", return_value=None),
        patch(
            "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
        ) as MockConverter,
    ):
        _mock_manifest(
            MockConverter,
            1,
            b"[{}]",
            unlisted_values={"quality_label": ("NG+",)},
        )
        obs = executor(
            _action(
                operation="create",
                dataset_path="/datasets/pcb",
                label_config_path="label_config.xml",
            ),
            conversation,
        )

    assert not obs.is_error
    assert "widened=quality_label:NG+" in obs.text
    created = mock_ls.create_project.call_args.kwargs["label_config"]
    # The values the template already declared are kept alongside the new one.
    assert extract_control_values(created)["quality_label"] == (
        "ok",
        "defect",
        "NG+",
    )


def test_create_reports_bindings_that_name_an_undeclared_control(
    conversation: MagicMock,
) -> None:
    """Widening cannot place a value whose control the config does not have."""
    executor = _create_executor()
    mock_ls = _mock_ls_api()
    with (
        patch.object(executor, "_load_state", return_value=None),
        patch.object(executor, "_build_ls_api", return_value=mock_ls),
        patch.object(executor, "_upload_to_storage", return_value=None),
        patch.object(executor, "_save_state", return_value=None),
        patch(
            "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
        ) as MockConverter,
    ):
        _mock_manifest(
            MockConverter,
            1,
            b"[{}]",
            unlisted_values={"missing_control": ("x",)},
        )
        obs = executor(
            _action(
                operation="create",
                dataset_path="/datasets/pcb",
                label_config_path="label_config.xml",
            ),
            conversation,
        )

    assert not obs.is_error
    assert "warning=unmapped_controls:missing_control" in obs.text


def test_create_reports_unmatched_regions_to_the_agent(
    conversation: MagicMock,
) -> None:
    """A binding that names a field no sample carries has to reach the agent.

    The editor renders it as a project with no rectangles, which is
    indistinguishable from a region the annotator has to draw by hand, so the
    only place this declaration/data mismatch can surface is here.
    """
    executor = _create_executor()
    mock_ls = _mock_ls_api()
    with (
        patch.object(executor, "_load_state", return_value=None),
        patch.object(executor, "_build_ls_api", return_value=mock_ls),
        patch.object(executor, "_upload_to_storage", return_value=None),
        patch.object(executor, "_save_state", return_value=None),
        patch(
            "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
        ) as MockConverter,
    ):
        _mock_manifest(MockConverter, 1, b"[{}]", unmatched_regions=("boxes",))
        obs = executor(
            _action(
                operation="create",
                dataset_path="/datasets/pcb",
                label_config_path="label_config.xml",
            ),
            conversation,
        )

    assert not obs.is_error
    assert "unmatched_regions:boxes" in obs.text


def test_create_stays_quiet_when_every_verdict_mapped(
    conversation: MagicMock,
) -> None:
    """The warning is the signal, so it must be absent when nothing is wrong."""
    executor = _create_executor()
    mock_ls = _mock_ls_api()
    with (
        patch.object(executor, "_load_state", return_value=None),
        patch.object(executor, "_build_ls_api", return_value=mock_ls),
        patch.object(executor, "_upload_to_storage", return_value=None),
        patch.object(executor, "_save_state", return_value=None),
        patch(
            "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
        ) as MockConverter,
    ):
        _mock_manifest(MockConverter, 1, b"[{}]")
        obs = executor(
            _action(
                operation="create",
                dataset_path="/datasets/pcb",
                label_config_path="label_config.xml",
            ),
            conversation,
        )

    assert not obs.is_error
    assert "unmapped_quality" not in obs.text


def _create_patches(executor: LabelStudioProjectExecutor, mock_ls: MagicMock):
    return (
        patch.object(executor, "_load_state", return_value=None),
        patch.object(executor, "_build_ls_api", return_value=mock_ls),
        patch.object(executor, "_upload_to_storage", return_value=None),
        patch.object(executor, "_save_state", return_value=None),
        patch("openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"),
    )


class TestCreateIdempotency:
    """A retried create must resume the same project, never duplicate it."""

    def test_existing_ready_project_is_reused(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        from openhands.tools.label_studio.models import ProjectState

        state = ProjectState(
            project_ref="abc",
            project_id=42,
            dataset_path="/datasets/pcb",
            adapter="avi_train",
            status="READY",
            imported_count=10,
            total_tasks=10,
        )
        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=state),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
        ):
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert not obs.is_error
        assert obs.project_id == 42
        assert "already exists" in obs.text
        assert not mock_ls.create_project.called
        assert not mock_ls.import_tasks.called

    def test_retry_resumes_from_the_recorded_batch(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        from openhands.tools.label_studio.models import ProjectState

        state = ProjectState(
            project_ref="abc",
            project_id=42,
            dataset_path="/datasets/pcb",
            adapter="avi_train",
            status="IMPORTING",
            imported_count=1,
            total_tasks=2,
            total_batches=2,
            next_batch=1,
        )
        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=state),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_save_state", return_value=None),
            patch.object(
                executor, "_download_from_storage", return_value=b'[{"data": {}}]'
            ) as mock_download,
        ):
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert not obs.is_error
        assert obs.status == "READY"
        assert obs.imported_count == 2
        assert not mock_ls.create_project.called
        assert mock_ls.import_tasks.call_count == 1
        assert mock_download.call_args[0][0].endswith("tasks-00002.json")

    def test_project_ref_is_stable_across_attempts(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        refs: list[str] = []
        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch.object(executor, "_save_state", return_value=None),
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
        ):
            _mock_manifest(MockConverter, 1, b"[{}]")
            for _ in range(3):
                refs.append(
                    executor(
                        _action(
                            operation="create",
                            dataset_path="/datasets/pcb",
                            label_config_path="label_config.xml",
                        ),
                        conversation,
                    ).project_ref
                )

        assert len(set(refs)) == 1

    def test_a_file_dataset_is_read_here_and_seeds_the_project_ref(
        self, conversation: MagicMock
    ) -> None:
        """A JSONL export is rewritten in place, so its path is not a revision.

        The converter cannot seed the ref itself -- the ref is decided before any
        artifact exists -- so the file is read once here, hashed into the ref,
        and handed over rather than fetched a second time.
        """
        payloads = [
            b'{"id": "s1", "defect_image": "/a.jpg"}\n',
            b'{"id": "s2"}\n',
        ]
        refs: list[str] = []
        for payload in payloads:
            executor = _create_executor()
            with (
                patch.object(executor, "_load_state", return_value=None),
                patch.object(executor, "_build_ls_api", return_value=_mock_ls_api()),
                patch.object(executor, "_upload_to_storage", return_value=None),
                patch.object(executor, "_save_state", return_value=None),
                patch.object(
                    executor, "_download_from_storage", return_value=payload
                ) as mock_download,
                patch(
                    "openhands.tools.label_studio.executor."
                    "AVITrainToLabelStudioConverter"
                ) as MockConverter,
            ):
                _mock_manifest(MockConverter, 1, b"[{}]")
                obs = executor(
                    _action(
                        operation="create",
                        dataset_path="/datasets/pcb-001/processed.jsonl",
                        label_config_path="label_config.xml",
                        adapter="jsonl",
                    ),
                    conversation,
                )
                refs.append(obs.project_ref)
                assert not obs.is_error
                assert MockConverter.call_args.kwargs["dataset_content"] == payload
            assert mock_download.call_args[0][0] == (
                "/datasets/pcb-001/processed.jsonl"
            )
            assert mock_download.call_args.kwargs["max_bytes"] == DATASET_FILE_MAX_BYTES

        assert refs[0] != refs[1]

    def test_a_directory_dataset_is_not_downloaded_to_seed_the_ref(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=_mock_ls_api()),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch.object(executor, "_save_state", return_value=None),
            patch.object(executor, "_download_from_storage") as mock_download,
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
        ):
            _mock_manifest(MockConverter, 1, b"[{}]")
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert not obs.is_error
        assert not mock_download.called
        assert MockConverter.call_args.kwargs["dataset_content"] is None

    def test_idempotency_key_separates_projects(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        refs: list[str] = []
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=_mock_ls_api()),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch.object(executor, "_save_state", return_value=None),
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
        ):
            _mock_manifest(MockConverter, 1, b"[{}]")
            for key in ("run-a", "run-b"):
                refs.append(
                    executor(
                        _action(
                            operation="create",
                            dataset_path="/datasets/pcb",
                            label_config_path="label_config.xml",
                            idempotency_key=key,
                        ),
                        conversation,
                    ).project_ref
                )

        assert refs[0] != refs[1]

    def test_cluster_separates_projects_for_the_same_dataset_path(
        self, conversation: MagicMock
    ) -> None:
        """Storage is replicated per cluster, so one path names two object sets."""
        refs: list[str] = []
        for cluster in ("us-west-1#pre", "us-west-2#pre"):
            executor = _create_executor(cluster=cluster)
            with (
                patch.object(executor, "_load_state", return_value=None),
                patch.object(executor, "_build_ls_api", return_value=_mock_ls_api()),
                patch.object(executor, "_upload_to_storage", return_value=None),
                patch.object(executor, "_save_state", return_value=None),
                patch(
                    "openhands.tools.label_studio.executor."
                    "AVITrainToLabelStudioConverter"
                ) as MockConverter,
            ):
                _mock_manifest(MockConverter, 1, b"[{}]")
                refs.append(
                    executor(
                        _action(
                            operation="create",
                            dataset_path="/datasets/pcb",
                            label_config_path="label_config.xml",
                        ),
                        conversation,
                    ).project_ref
                )

        assert refs[0] != refs[1]

    def test_import_failure_keeps_project_identity(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        mock_ls = _mock_ls_api()
        mock_ls.import_tasks.side_effect = LabelStudioAPIError("import exploded")
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch.object(executor, "_save_state", return_value=None),
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
        ):
            _mock_manifest(MockConverter, 3, b"[{},{},{}]")
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert obs.is_error
        assert obs.project_id == 42
        assert obs.project_ref
        assert "import exploded" in obs.text


class TestPortalWiring:
    def test_open_url_targets_the_portal_sso_endpoint(
        self, conversation: MagicMock
    ) -> None:
        executor = _create_executor(portal_base_url="http://localhost:8000")
        mock_ls = _mock_ls_api()
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch.object(executor, "_save_state", return_value=None),
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
        ):
            _mock_manifest(MockConverter, 1, b"[{}]")
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert obs.open_url == (
            "http://localhost:8000/label_studio/sso?target=/projects/42/data"
        )
        # The same project, named directly: only the portal entry establishes the
        # browser session that makes this one render.
        assert obs.project_url == "http://ls.example.com/projects/42/data"

    def test_without_a_portal_open_url_is_already_the_label_studio_address(
        self, executor: LabelStudioProjectExecutor
    ) -> None:
        assert executor._open_url(42) == "http://ls.example.com/projects/42/data"
        assert executor._open_url(42) == executor._project_url(42)

    def test_media_signer_is_built_only_with_a_portal(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        assert executor._build_media_signer(conversation) is None
        portal_executor = _create_executor(portal_base_url="http://localhost:8000")
        assert portal_executor._build_media_signer(conversation) is not None

    def test_media_signer_targets_the_conversation_cluster(
        self, conversation: MagicMock
    ) -> None:
        executor = _create_executor(
            portal_base_url="http://localhost:8000", cluster="us-west-1#pre"
        )

        signer = executor._build_media_signer(conversation)

        assert signer is not None
        assert signer._cluster == "us-west-1#pre"

    def test_export_ticket_provider_is_built_only_with_a_portal(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        assert executor._build_export_ticket_provider(conversation) is None
        portal_executor = _create_executor(portal_base_url="http://localhost:8000")
        assert portal_executor._build_export_ticket_provider(conversation) is not None

    def test_export_ticket_provider_targets_the_conversation_cluster(
        self, conversation: MagicMock
    ) -> None:
        executor = _create_executor(
            portal_base_url="http://localhost:8000", cluster="us-west-1#pre"
        )

        provider = executor._build_export_ticket_provider(conversation)

        assert provider is not None
        assert provider._cluster == "us-west-1#pre"


class TestLoadState:
    def test_missing_state_reads_as_absent(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        with patch(
            "openhands.tools.label_studio.executor.download_file_from_pyromind",
            side_effect=StorageFileNotFoundError("gone"),
        ):
            assert executor._load_state("ref", conversation) is None

    def test_storage_failure_is_not_reported_as_absent(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        with patch(
            "openhands.tools.label_studio.executor.download_file_from_pyromind",
            side_effect=ValueError("storage down"),
        ):
            with pytest.raises(ValueError, match="storage down"):
                executor._load_state("ref", conversation)


class TestOrphanRecovery:
    def test_existing_label_studio_project_is_adopted(
        self, executor: LabelStudioProjectExecutor, conversation: MagicMock
    ) -> None:
        """A project left behind by a failed attempt must not be duplicated."""
        mock_ls = _mock_ls_api()
        mock_ls.find_project_by_title.return_value = {
            "id": 77,
            "title": "pyromind_deadbeef",
        }
        with (
            patch.object(executor, "_load_state", return_value=None),
            patch.object(executor, "_build_ls_api", return_value=mock_ls),
            patch.object(executor, "_upload_to_storage", return_value=None),
            patch.object(executor, "_save_state", return_value=None),
            patch(
                "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
            ) as MockConverter,
        ):
            _mock_manifest(MockConverter, 1, b"[{}]")
            obs = executor(
                _action(
                    operation="create",
                    dataset_path="/datasets/pcb",
                    label_config_path="label_config.xml",
                ),
                conversation,
            )

        assert not obs.is_error
        assert obs.project_id == 77
        assert not mock_ls.create_project.called
        mock_ls.find_project_by_title.assert_called_once()


def test_uploads_keep_the_object_name_storage_looks_for(conversation):
    """Storage names an object after the local file, so a suffixed temp name loses."""
    executor = LabelStudioProjectExecutor(
        ls_base_url="http://ls.example.com",
        ls_token_secret="LABEL_STUDIO_TOKEN",
        storage_base_url="http://storage.example.com",
    )
    uploaded: list[dict[str, str]] = []

    def fake_upload(*, local_path: Path, target_dir: str, **kwargs: Any) -> None:
        uploaded.append({"name": local_path.name, "dir": target_dir})

    with patch(
        "openhands.tools.label_studio.executor.upload_local_file_to_pyromind",
        fake_upload,
    ):
        executor._upload_to_storage(
            "/.pyromind-agent/label-studio/ref/project_state.json",
            b"{}",
            conversation,
        )

    assert uploaded == [
        {
            "name": "project_state.json",
            "dir": "/.pyromind-agent/label-studio/ref",
        }
    ]


@pytest.mark.parametrize(
    "value",
    ["/workspace/exports/asset", "workspace/exports/asset", "exports/asset"],
)
def test_workspace_prefixed_paths_name_the_same_storage_object(value):
    """Callers quote platform workspace paths as readily as storage paths."""
    assert _normalize_storage_path(value, "dataset_path") == "/exports/asset"


def test_ls_api_uses_the_token_the_portal_issues_for_this_caller(
    monkeypatch, conversation
):
    """A token shared by every conversation hides each user's own projects."""
    executor = LabelStudioProjectExecutor(
        ls_base_url="http://ls.example.com",
        ls_token_secret="LABEL_STUDIO_TOKEN",
        storage_base_url="http://storage.example.com",
        portal_base_url="https://pre-api-portal.pyromind.ai",
    )
    monkeypatch.setattr(PortalTokenProvider, "fetch", lambda self: "per-user-token")

    api = executor._build_ls_api(conversation)

    assert api._headers()["Authorization"] == "Token per-user-token"
    # The shared token is never consulted when the portal can issue one.
    read_secrets = [
        call.args[0]
        for call in conversation.state.secret_registry.get_secret_value.call_args_list
    ]
    assert "LABEL_STUDIO_TOKEN" not in read_secrets


def test_ls_api_falls_back_to_the_shared_token(conversation):
    """Deployments that hold a single Label Studio token keep working."""
    executor = LabelStudioProjectExecutor(
        ls_base_url="http://ls.example.com",
        ls_token_secret="LABEL_STUDIO_TOKEN",
        storage_base_url="http://storage.example.com",
    )

    api = executor._build_ls_api(conversation)

    assert api._headers()["Authorization"] == "Token test-token"


PORTAL = "https://portal.example.com"
TICKET = "opaque-export-ticket"


def _ready_state(**overrides: Any) -> ProjectState:
    params: dict[str, Any] = {
        "project_ref": "abc",
        "project_id": 42,
        "dataset_path": "/datasets/pcb",
        "adapter": "aoi_export",
        "status": "READY",
        "imported_count": 1,
        "total_tasks": 1,
    }
    params.update(overrides)
    return ProjectState(**params)


def _mock_media_signer(expires_in: int = 604800) -> MagicMock:
    signer = MagicMock()
    signer.expires_in = expires_in
    signer.sign_many.side_effect = lambda paths: {
        path: f"{PORTAL}/label_studio/media?path={path}&media_token=fresh"
        for path in paths
    }
    return signer


def _task(task_id: int, **data: Any) -> dict[str, Any]:
    return {"id": task_id, "data": data}


def _refresh(
    executor: LabelStudioProjectExecutor,
    conversation: MagicMock,
    state: ProjectState,
    ls_api: MagicMock,
    signer: MagicMock,
):
    with (
        patch.object(executor, "_load_state", return_value=state),
        patch.object(executor, "_build_ls_api", return_value=ls_api),
        patch.object(executor, "_build_media_signer", return_value=signer),
        patch.object(executor, "_save_state"),
    ):
        return executor(
            _action(operation="refresh_media", project_ref="abc"), conversation
        )


def test_refresh_media_resigns_the_urls_and_keeps_the_rest_of_the_data(conversation):
    """Only the rendered URL changes; the path and the labels stay untouched."""
    executor = _create_executor(portal_base_url=PORTAL)
    signer = _mock_media_signer()
    ls_api = _mock_ls_api()
    ls_api.list_project_tasks.return_value = [
        _task(
            1,
            sample_id="10_B1",
            defect_image="https://stale/defect.jpg",
            defect_image_path="exports/10_B1/defect.jpg",
        )
    ]

    obs = _refresh(executor, conversation, _ready_state(), ls_api, signer)

    assert not obs.is_error
    assert signer.sign_many.call_args.args[0] == ["exports/10_B1/defect.jpg"]
    written = ls_api.update_task_data.call_args.kwargs
    assert written["task_id"] == 1
    assert written["data"]["defect_image"].endswith("media_token=fresh")
    # Label Studio replaces the whole data object, so everything else has to
    # travel with the new URL.
    assert written["data"]["defect_image_path"] == "exports/10_B1/defect.jpg"
    assert written["data"]["sample_id"] == "10_B1"
    assert "refreshed=1/1" in obs.text


def test_refresh_media_records_the_window_without_reporting_a_deadline(conversation):
    """The signature window is bookkeeping; the portal does not enforce it."""
    executor = _create_executor(portal_base_url=PORTAL)
    signer = _mock_media_signer(expires_in=604800)
    ls_api = _mock_ls_api()
    ls_api.list_project_tasks.return_value = [
        _task(1, defect_image="x", defect_image_path="exports/a.jpg")
    ]
    state = _ready_state()

    obs = _refresh(executor, conversation, state, ls_api, signer)

    assert state.media_expires_at is not None
    assert "media_urls_expire" not in obs.text


def test_refresh_media_signs_each_path_once(conversation):
    """A path shared by several tasks costs one signature, not one per task."""
    executor = _create_executor(portal_base_url=PORTAL)
    signer = _mock_media_signer()
    ls_api = _mock_ls_api()
    ls_api.list_project_tasks.return_value = [
        _task(1, defect_image="x", defect_image_path="exports/a.jpg"),
        _task(2, defect_image="x", defect_image_path="exports/a.jpg"),
    ]

    _refresh(executor, conversation, _ready_state(), ls_api, signer)

    assert signer.sign_many.call_args.args[0] == ["exports/a.jpg"]
    assert ls_api.update_task_data.call_count == 2


def test_refresh_media_skips_tasks_without_media_paths(conversation):
    executor = _create_executor(portal_base_url=PORTAL)
    signer = _mock_media_signer()
    ls_api = _mock_ls_api()
    ls_api.list_project_tasks.return_value = [
        _task(1, defect_image="x", defect_image_path="exports/a.jpg"),
        _task(2, sample_id="meta-only"),
    ]

    obs = _refresh(executor, conversation, _ready_state(), ls_api, signer)

    assert not obs.is_error
    assert ls_api.update_task_data.call_count == 1
    assert "refreshed=1/2" in obs.text


def test_refresh_media_keeps_the_tasks_that_were_written(conversation):
    """A task that fails stays for the next run; the rest keep rendering."""
    executor = _create_executor(portal_base_url=PORTAL)
    signer = _mock_media_signer()
    ls_api = _mock_ls_api()
    ls_api.list_project_tasks.return_value = [
        _task(1, defect_image="x", defect_image_path="exports/a.jpg"),
        _task(2, defect_image="x", defect_image_path="exports/b.jpg"),
    ]

    def _update(*, task_id: int, data: dict[str, Any]) -> dict[str, Any]:
        if task_id == 1:
            raise LabelStudioAPIError("HTTP 500: task is locked")
        return {"id": task_id}

    ls_api.update_task_data.side_effect = _update

    obs = _refresh(executor, conversation, _ready_state(), ls_api, signer)

    assert not obs.is_error
    assert "refreshed=1/2" in obs.text
    assert "failed=1" in obs.text
    assert "task 1" in obs.text


def test_refresh_media_reports_an_error_when_no_task_could_be_written(conversation):
    executor = _create_executor(portal_base_url=PORTAL)
    signer = _mock_media_signer()
    ls_api = _mock_ls_api()
    ls_api.list_project_tasks.return_value = [
        _task(1, defect_image="x", defect_image_path="exports/a.jpg")
    ]
    ls_api.update_task_data.side_effect = LabelStudioAPIError("HTTP 500: offline")

    obs = _refresh(executor, conversation, _ready_state(), ls_api, signer)

    assert obs.is_error
    assert "failed for all 1 tasks" in obs.text
    # The project keeps its identity so the agent retries instead of recreating.
    assert obs.project_ref == "abc"


def test_refresh_media_requires_a_portal(conversation):
    executor = _create_executor()
    with patch.object(executor, "_load_state", return_value=_ready_state()):
        obs = executor(
            _action(operation="refresh_media", project_ref="abc"), conversation
        )

    assert obs.is_error
    assert "portal" in obs.text


def test_refresh_media_rejects_a_project_without_media_paths(conversation):
    """Projects imported before media paths were stored cannot be re-signed."""
    executor = _create_executor(portal_base_url=PORTAL)
    ls_api = _mock_ls_api()
    ls_api.list_project_tasks.return_value = [_task(1, sample_id="meta-only")]

    obs = _refresh(executor, conversation, _ready_state(), ls_api, _mock_media_signer())

    assert obs.is_error
    assert "cannot be re-signed" in obs.text


@pytest.mark.parametrize(
    "remaining",
    [timedelta(days=6), timedelta(hours=2), timedelta(hours=-1)],
)
def test_get_never_reports_a_media_deadline(conversation, remaining):
    """However old the recorded window looks, there is no deadline to warn about."""
    executor = _create_executor()
    state = _ready_state(media_expires_at=(datetime.now(UTC) + remaining).isoformat())

    with patch.object(executor, "_load_state", return_value=state):
        obs = executor(_action(operation="get", project_ref="abc"), conversation)

    assert "media_urls_expire" not in obs.text


def test_create_records_the_window_the_portal_reported(conversation):
    executor = _create_executor(portal_base_url=PORTAL)
    signer = _mock_media_signer(expires_in=604800)
    mock_ls = _mock_ls_api()
    with (
        patch.object(executor, "_load_state", return_value=None),
        patch.object(executor, "_build_ls_api", return_value=mock_ls),
        patch.object(executor, "_build_media_signer", return_value=signer),
        patch.object(executor, "_upload_to_storage", return_value=None),
        patch.object(executor, "_save_state") as save_state,
        patch(
            "openhands.tools.label_studio.executor.AVITrainToLabelStudioConverter"
        ) as MockConverter,
    ):
        manifest = MockConverter.return_value.convert.return_value
        manifest.total_tasks = 1
        manifest.batch_payloads = [("tasks-00001.json", b"[{}]", 1)]
        manifest.unmapped_quality = ()
        manifest.unmatched_regions = ()
        manifest.unlisted_values = {}
        manifest.to_manifest_data.return_value.model_dump_json.return_value = "{}"

        obs = executor(
            _action(
                operation="create",
                dataset_path="/datasets/pcb",
                label_config_path="label_config.xml",
            ),
            conversation,
        )

    assert not obs.is_error
    saved = [call.args[1] for call in save_state.call_args_list]
    assert saved[-1].media_expires_at is not None
    assert "media_urls_expire" not in obs.text


def test_export_keeps_working_when_the_description_write_fails(conversation):
    """The stored file is the deliverable; a stale note in the project is not."""
    executor = _create_executor()
    state = ProjectState(
        project_ref="abc", project_id=42, dataset_path="/d", adapter="avi_train"
    )
    ls_api = _mock_ls_api(export_data=[{"data": {}, "annotations": []}])
    ls_api.update_project_description.side_effect = LabelStudioAPIError("nope")

    with (
        patch.object(executor, "_load_state", return_value=state),
        patch.object(executor, "_build_ls_api", return_value=ls_api),
        patch.object(executor, "_upload_to_storage", return_value=None) as upload,
    ):
        obs = executor(_action(operation="export", project_ref="abc"), conversation)

    assert not obs.is_error
    assert obs.status == "EXPORTED"
    assert upload.called
