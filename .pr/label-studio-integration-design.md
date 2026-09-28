# Label Studio 集成实现方案

## 1. 总体架构

四层职责分离，复用现有 PyroMind Agent，不新增 Agent 类型：

| 层 | 组件 | 职责 |
|----|------|------|
| 数据理解 | `preview_dataset`（已有 Tool） | 帮 Agent 理解数据结构 |
| 配置生成 | `.agents/skills/label-studio/` Skill | Agent 根据数据结构生成 XML |
| 格式转换 | `AVITrainToLabelStudioConverter` | 确定性转换完整数据集 |
| 平台操作 | `label_studio_project` Tool | 调 Label Studio REST API |

```
用户提供 dataset_path
  → Agent 激活 label-studio Skill
  → preview_dataset(path)                    [已有 Tool, 不改]
  → Agent 生成 label_config.xml + label_studio_data_schema.json
  → Agent 校验 XML（Skill 内置 validate_label_config.py）
  → Agent 调用 label_studio_project(operation="create", ...)
      Tool 内部:
        1. 校验 dataset_path
        2. 生成 project_ref (UUID)
        3. 调用 AVITrainToLabelStudioConverter → Task Manifest
        4. 保存 XML + Manifest 到 /.pyromind-agent/label-studio/<ref>/
        5. 校验 Manifest 与 XML 匹配
        6. POST /api/projects (创建 LS Project)
        7. POST /api/projects/{id}/import (分批)
        8. 更新 project_state.json
        9. 返回 project_id, open_url, manifest_path
  → 用户标注完成
  → Agent 调用 label_studio_project(operation="export", ...)
      Tool 内部:
        1. GET /api/projects/{id}/export?export_type=JSON
        2. 调用 LabelStudioToAVITrainConverter → PyroMind 格式
        3. 保存到用户 Storage
        4. 返回 output_path
```

## 2. Skill 定义

### 2.1 目录结构

```
.agents/skills/label-studio/
├── SKILL.md
├── scripts/
│   └── validate_label_config.py
└── references/
    ├── pcb-avi-review.xml            # PCB AVI 数据的完整 XML 模板
    ├── label-studio-schema.md        # Label Studio XML 语法要点
    └── task-manifest-format.md       # Task Manifest JSON 格式规范
```

### 2.2 SKILL.md 内容

```markdown
---
name: label-studio
description: >-
  使用 label_studio_project 工具将用户数据集创建为 Label Studio
  标注项目。用户要求创建标注项目、标注数据、打标、review 图片质检
  结果时使用。需要先 preview_dataset 理解数据再生成 XML。
---

# Label Studio 标注集成

将用户 Storage 中的数据集导入 Label Studio 进行人工标注。
认证由服务端管理，不需要用户提供 Token 或 UID。

## 固定流程

1. **理解数据**：调用 preview_dataset(path=...)，分析目录结构、
   图片文件、meta 字段。
2. **生成配置**：根据样本结构生成 label_config.xml。
   参考 references/pcb-avi-review.xml（PCB AVI 数据）或
   references/label-studio-schema.md 生成自定义 XML。
3. **校验 XML**：执行 scripts/validate_label_config.py。
4. **创建项目**：调用 label_studio_project(operation="create", ...)。

## 禁止

- 未 preview_dataset 就生成 XML
- 在 operation 参数中传递 UID、Token 或认证信息
- 用 terminal/curl 直接调用 Label Studio API
- 由 Agent 遍历完整数据集或生成 Manifest

## 修改已有项目

调用 label_studio_project(operation="get", project_ref=...) 获取当前
配置 → 在 workspace 中修改 XML → 校验 → 调用
label_studio_project(operation="update_config", ...)。
如果新 XML 删除了已有标注使用的控件名，Tool 会拒绝更新。

## 导出标注结果

调用 label_studio_project(operation="export", project_ref=...)。
Tool 自动将标注结果转回 PyroMind 格式并保存到用户 Storage。
```

### 2.3 validate_label_config.py

本地校验脚本，不调用 Label Studio API。检查：

```python
"""Validate a Label Studio label_config.xml for structural correctness.

Checks:
1. XML is well-formed
2. All <Image name=...> objects have unique names
3. All <Choices name=...> / <RectangleLabels name=...> / <TextArea name=...>
   controls have unique names
4. Every control references an existing object via toName=
5. from_name/to_name/type triplets can be extracted for Manifest validation
6. Output: JSON list of {"from_name", "to_name", "type"} triplets to stdout
"""
import sys
import xml.etree.ElementTree as ET

def validate(xml_path: str) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = {}     # name -> tag type (Image, Text, etc.)
    controls = []    # list of (from_name, to_name, type)
    seen_control_names = set()

    for elem in root.iter():
        tag = elem.tag
        name = elem.get("name", "")
        if not name:
            continue
        if tag in ("Image", "Text", "Audio", "Video", "HyperText"):
            if name in objects:
                raise ValueError(f"Duplicate object name: {name}")
            objects[name] = tag
        elif tag in ("Choices", "RectangleLabels", "Labels", "TextArea",
                     "BrushLabels", "PolygonLabels", "KeyPointLabels"):
            if name in seen_control_names:
                raise ValueError(f"Duplicate control name: {name}")
            seen_control_names.add(name)
            to_name = elem.get("toName", "")
            if to_name not in objects:
                raise ValueError(
                    f"Control '{name}' references unknown object '{to_name}'"
                )
            controls.append({"from_name": name, "to_name": to_name, "type": _ls_type(tag)})

    return controls

def _ls_type(tag: str) -> str:
    return {
        "Choices": "choices",
        "RectangleLabels": "rectanglelabels",
        "Labels": "labels",
        "TextArea": "textarea",
        "BrushLabels": "brushlabels",
        "PolygonLabels": "polygonlabels",
        "KeyPointLabels": "keypointlabels",
    }.get(tag, tag.lower())

if __name__ == "__main__":
    controls = validate(sys.argv[1])
    print(json.dumps({"valid": True, "controls": controls}, ensure_ascii=False))
```

### 2.4 references/pcb-avi-review.xml（PCB AVI 完整模板）

```xml
<View>
  <Header value="PCB AVI Review"/>
  <View style="display:flex; gap:8px;">
    <View style="flex:1;">
      <Header value="缺陷图"/>
      <Image name="defect_image" value="$defect_image" maxWidth="600"/>
    </View>
    <View style="flex:1;">
      <Header value="差分图"/>
      <Image name="diff_image" value="$diff_image" maxWidth="600"/>
    </View>
    <View style="flex:1;">
      <Header value="金手指/GT"/>
      <Image name="gt_image" value="$gt_image" maxWidth="600"/>
    </View>
  </View>
  <View style="margin-top:16px;">
    <Header value="整体判定"/>
    <Choices name="quality_label" toName="defect_image" choice="single">
      <Choice value="ok"/>
      <Choice value="defect"/>
    </Choices>
  </View>
  <View style="margin-top:16px;">
    <Header value="缺陷区域标注"/>
    <RectangleLabels name="finding_category" toName="defect_image">
      <Label value="开路" background="#FF6B6B"/>
      <Label value="短路" background="#4ECDC4"/>
      <Label value="缺口" background="#45B7D1"/>
      <Label value="毛刺" background="#FFA07A"/>
      <Label value="余铜" background="#98D8C8"/>
      <Label value="针孔" background="#F7DC6F"/>
    </RectangleLabels>
    <TextArea name="finding_observation" toName="defect_image"
              placeholder="描述缺陷现象" maxRows="3" editable="true"
              perRegion="true"/>
  </View>
</View>
```

## 3. Tool 定义

### 3.1 目录结构

```
openhands-tools/openhands/tools/label_studio/
├── __init__.py
├── definition.py          # Action, Observation, ToolDefinition
├── converter.py           # AVITrainToLabelStudioConverter + LabelStudioToAVITrainConverter
├── executor.py            # LabelStudioProjectExecutor
├── models.py              # ProjectState, ManifestBatch, ManifestData
└── api_client.py          # Label Studio REST API 封装
```

### 3.2 Action 定义

```python
class LabelStudioProjectAction(Action):
    operation: Literal["create", "get", "update_config", "status", "export"] = Field(
        description="The operation to perform."
    )
    dataset_path: str | None = Field(
        default=None,
        description="User storage relative path. Required for operation='create'.",
    )
    label_config_path: str | None = Field(
        default=None,
        description=(
            "Workspace-relative path to the label_config.xml file. "
            "Required for operation='create' and 'update_config'."
        ),
    )
    adapter: str = Field(
        default="avi_train",
        description="Data adapter that determines the conversion strategy.",
    )
    project_ref: str | None = Field(
        default=None,
        description="UUID of an existing project. Required for get/update_config/status/export.",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Client-generated key for safe retries of operation='create'.",
    )
    expected_config_version: int | None = Field(
        default=None,
        description=(
            "For operation='update_config': the config_version from the last "
            "get operation. Prevents concurrent modification."
        ),
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "For operation='export': storage-relative output directory. "
            "Defaults to /export under the project's artifact dir."
        ),
    )
```

### 3.3 Observation 定义

```python
class LabelStudioProjectObservation(Observation):
    operation: str
    project_ref: str | None = None
    project_id: int | None = None
    status: str | None = None            # READY / IMPORTING / ERROR / EXPORTED
    config_version: int | None = None
    task_count: int | None = None
    imported_count: int | None = None
    annotation_count: int | None = None
    manifest_path: str | None = None
    open_url: str | None = None
    export_path: str | None = None
    next_batch: int | None = None
    last_error: str | None = None
```

### 3.4 Tool Description

```python
TOOL_DESCRIPTION = """Create, inspect, update, or export Label Studio annotation projects.

operation='create': Import a user-storage dataset into Label Studio.
Requires dataset_path and label_config_path. The tool reads the XML,
converts the full dataset via the configured adapter, and imports tasks
in batches. Returns project_id and open_url.

operation='get': Fetch current project state and config.
operation='update_config': Update the label config XML. Rejects if
  existing annotations reference controls removed in the new XML.
operation='status': Check import/annotation progress.
operation='export': Download annotations and convert back to PyroMind format.

Do not pass UID, Token, or auth credentials. The tool resolves auth
server-side. Always call preview_dataset before generating XML for a
new dataset. Do not use terminal or curl to call Label Studio directly.
"""
```

### 3.5 Executor

```python
class LabelStudioProjectExecutor(
    ToolExecutor[LabelStudioProjectAction, LabelStudioProjectObservation]
):
    def __init__(
        self,
        *,
        ls_base_url: str,
        ls_token_secret: str,          # SecretRegistry key name
        storage_base_url: str,
        storage_headers: dict[str, str],
        storage_secret_headers: dict[str, str],
        artifact_dir: str | None = None,  # default: PYROMIND_AGENT_STORAGE_ROOT / label-studio
        adapter_dir: str | None = None,   # runtime_dir for converter scripts
        sso_base_url: str | None = None,  # PyroMind SSO endpoint for open_url generation
        batch_size: int = 500,
        timeout: int = 60,
    ):
        ...

    def __call__(self, action, conversation=None):
        dispatch = {
            "create": self._handle_create,
            "get": self._handle_get,
            "update_config": self._handle_update_config,
            "status": self._handle_status,
            "export": self._handle_export,
        }
        handler = dispatch.get(action.operation)
        if handler is None:
            return error(f"Unknown operation: {action.operation}")
        return handler(action, conversation)

    def _handle_create(self, action, conversation):
        # 1. Validate dataset_path
        dataset_path = _normalize_storage_path(action.dataset_path)

        # 2. Read label_config.xml from workspace
        xml_path = Path(workspace.working_dir) / action.label_config_path
        xml_content = xml_path.read_text()

        # 3. Validate XML locally
        controls = validate_label_config(xml_path)

        # 4. Generate project_ref
        project_ref = str(uuid.uuid4())

        # 5. Create artifact directory in user Storage
        artifact_dir = f"{PYROMIND_AGENT_STORAGE_ROOT}/label-studio/{project_ref}"

        # 6. Upload XML to Storage
        upload_to_storage(f"{artifact_dir}/label_config.xml", xml_content.encode())

        # 7. Run Converter
        converter = AVITrainToLabelStudioConverter(
            dataset_path=dataset_path,
            storage_base_url=self._storage_base_url,
            storage_headers=self._resolved_storage_headers(conversation),
            media_gateway_url=self._media_gateway_url,
        )
        manifest = converter.convert(controls=controls)

        # 8. Validate Manifest against XML controls
        _validate_manifest_against_config(manifest, controls)

        # 9. Upload Manifest batches to Storage
        for batch in manifest.batches:
            upload_to_storage(f"{artifact_dir}/manifests/{batch.path}", batch.data)
        upload_to_storage(f"{artifact_dir}/manifests/manifest.json", manifest.index_json())

        # 10. Create LS Project
        ls_project = self._ls_api.create_project(
            title=f"pyromind_{project_ref[:8]}",
            label_config=xml_content,
        )

        # 11. Import batches
        imported = 0
        for batch in manifest.batches:
            self._ls_api.import_tasks(ls_project.id, batch.tasks)
            imported += batch.task_count
            self._save_state(artifact_dir, ProjectState(
                project_id=ls_project.id,
                status="IMPORTING",
                next_batch=batch.index + 1,
                imported_count=imported,
            ))

        # 12. Generate open_url
        open_url = f"{self._sso_base_url}/sso/label-studio?project_id={ls_project.id}"

        # 13. Final state
        self._save_state(artifact_dir, ProjectState(
            project_id=ls_project.id,
            status="READY",
            imported_count=imported,
        ))

        return LabelStudioProjectObservation(...)

    def _handle_update_config(self, action, conversation):
        # 1. Load project state
        state = self._load_state(action.project_ref)

        # 2. Check config_version matches expected
        if action.expected_config_version != state.config_version:
            return error("Config version mismatch; call get first.")

        # 3. Read new XML
        new_xml = read_file(action.label_config_path)

        # 4. Validate new XML
        new_controls = validate_label_config(new_xml)

        # 5. Check annotation compatibility
        existing_from_names = self._ls_api.get_annotation_from_names(state.project_id)
        new_from_names = {c["from_name"] for c in new_controls}
        removed = existing_from_names - new_from_names
        if removed:
            return error(
                f"Cannot remove controls used by existing annotations: {removed}. "
                "Create a new project instead."
            )

        # 6. PATCH project
        self._ls_api.update_project_config(state.project_id, new_xml)

        # 7. Update state
        state.config_version += 1
        self._save_state(...)

        return LabelStudioProjectObservation(config_version=state.config_version, ...)
```

### 3.6 API Client

```python
class LabelStudioAPIClient:
    """Thin wrapper over Label Studio REST API."""

    def __init__(self, base_url: str, token: str, timeout: int = 60):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self._token}",
            "Content-Type": "application/json",
        }

    def create_project(self, title: str, label_config: str) -> dict:
        resp = httpx.post(
            f"{self._base_url}/api/projects",
            headers=self._headers(),
            json={"title": title, "label_config": label_config},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def import_tasks(self, project_id: int, tasks: list[dict]) -> dict:
        resp = httpx.post(
            f"{self._base_url}/api/projects/{project_id}/import",
            headers=self._headers(),
            json={"tasks": tasks},
            timeout=max(self._timeout, 120),
        )
        resp.raise_for_status()
        return resp.json()

    def update_project_config(self, project_id: int, label_config: str) -> dict:
        # Validate first
        validate_resp = httpx.post(
            f"{self._base_url}/api/projects/{project_id}/validate-config",
            headers=self._headers(),
            json={"label_config": label_config},
            timeout=self._timeout,
        )
        validate_resp.raise_for_status()

        resp = httpx.patch(
            f"{self._base_url}/api/projects/{project_id}",
            headers=self._headers(),
            json={"label_config": label_config},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def export_annotations(self, project_id: int) -> list[dict]:
        resp = httpx.get(
            f"{self._base_url}/api/projects/{project_id}/export",
            params={"exportType": "JSON"},
            headers=self._headers(),
            timeout=max(self._timeout, 120),
        )
        resp.raise_for_status()
        return resp.json()

    def get_project(self, project_id: int) -> dict:
        resp = httpx.get(
            f"{self._base_url}/api/projects/{project_id}",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_annotation_from_names(self, project_id: int) -> set[str]:
        """Extract all from_name values used in existing annotations."""
        tasks = self.export_annotations(project_id)
        names = set()
        for task in tasks:
            for ann in task.get("annotations", []):
                for result in ann.get("result", []):
                    names.add(result.get("from_name", ""))
            for pred in task.get("predictions", []):
                for result in pred.get("result", []):
                    names.add(result.get("from_name", ""))
        names.discard("")
        return names
```

## 4. Converter

### 4.1 AVITrainToLabelStudioConverter

确定性转换器，不调用 Label Studio API，只读 Storage、生成 Manifest。

```python
class AVITrainToLabelStudioConverter:
    """Convert AVI Train dataset (sample dirs with 3 images + meta_vlm.json)
    into Label Studio Task Manifest JSON batches."""

    BATCH_TASK_LIMIT = 500
    BATCH_SIZE_LIMIT = 10 * 1024 * 1024  # 10MB per batch

    def __init__(
        self,
        dataset_path: str,
        storage_base_url: str,
        storage_headers: dict[str, str],
        media_gateway_url: str,
        timeout: int = 30,
    ):
        self._dataset_path = dataset_path
        self._storage = storage_base_url
        self._headers = storage_headers
        self._media_gateway_url = media_gateway_url
        self._timeout = timeout

    def convert(self, controls: list[dict]) -> ConvertedManifest:
        # 1. List sample directories
        samples = self._list_samples()

        # 2. For each sample, build a Label Studio Task
        tasks = []
        for sample_dir in samples:
            task = self._build_task(sample_dir, controls)
            if task is not None:
                tasks.append(task)

        # 3. Split into batches
        batches = self._split_batches(tasks)

        return ConvertedManifest(
            total_tasks=len(tasks),
            batches=batches,
        )

    def _build_task(self, sample_dir: str, controls: list[dict]) -> dict | None:
        # Read meta_vlm.json
        meta = self._read_json(f"{sample_dir}/meta_vlm.json")
        if meta is None:
            return None

        # Build media URLs via Media Gateway
        defect_url = self._media_url(f"{sample_dir}/defect.jpg")
        diff_url = self._media_url(f"{sample_dir}/diff.jpg")
        gt_url = self._media_url(f"{sample_dir}/gt.jpg")

        # Task data
        data = {
            "defect_image": defect_url,
            "diff_image": diff_url,
            "gt_image": gt_url,
            "sample_id": meta.get("sample_id", ""),
        }

        # Build predictions from meta fields
        predictions = self._build_predictions(meta, controls)

        return {"data": data, "predictions": [predictions] if predictions else []}

    def _media_url(self, storage_path: str) -> str:
        """Generate a signed Media Gateway URL for the image."""
        signed_ref = self._sign_media_ref(storage_path)
        return f"{self._media_gateway_url}/{signed_ref}"

    def _build_predictions(self, meta: dict, controls: list[dict]) -> dict | None:
        """Convert meta_vlm fields into Label Studio predictions."""
        results = []

        # quality_label from meta
        if "quality" in meta:
            results.append({
                "from_name": "quality_label",
                "to_name": "defect_image",
                "type": "choices",
                "value": {"choices": [meta["quality"]]},
            })

        # findings → rectanglelabels + textarea
        for i, finding in enumerate(meta.get("findings", [])):
            region_id = f"finding_{i + 1}"
            bbox = finding.get("bbox", {})
            # norm1000 → percentage
            x = bbox.get("x_min_norm", 0) / 10.0
            y = bbox.get("y_min_norm", 0) / 10.0
            w = (bbox.get("x_max_norm", 0) - bbox.get("x_min_norm", 0)) / 10.0
            h = (bbox.get("y_max_norm", 0) - bbox.get("y_min_norm", 0)) / 10.0
            results.append({
                "id": region_id,
                "from_name": "finding_category",
                "to_name": "defect_image",
                "type": "rectanglelabels",
                "value": {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(w, 2),
                    "height": round(h, 2),
                    "rectanglelabels": [finding.get("category", "")],
                },
            })
            obs = finding.get("observation", "")
            if obs:
                results.append({
                    "id": region_id,
                    "from_name": "finding_observation",
                    "to_name": "defect_image",
                    "type": "textarea",
                    "value": {"text": [obs]},
                })

        if not results:
            return None
        return {"model_version": "vlm-v1", "score": meta.get("score", 0.92), "result": results}
```

### 4.2 LabelStudioToAVITrainConverter

```python
class LabelStudioToAVITrainConverter:
    """Convert Label Studio export JSON back to PyroMind AVI Train format."""

    def convert(self, export_data: list[dict]) -> list[dict]:
        """Each element in export_data is a Label Studio Task with annotations."""
        results = []
        for task in export_data:
            sample = self._task_to_sample(task)
            if sample is not None:
                results.append(sample)
        return results

    def _task_to_sample(self, task: dict) -> dict | None:
        data = task.get("data", {})
        annotations = task.get("annotations", [])
        if not annotations:
            return None

        # Use the latest annotation
        ann = annotations[-1]
        sample = {
            "sample_id": data.get("sample_id", ""),
            "defect_image_path": data.get("defect_image", ""),
            "diff_image_path": data.get("diff_image", ""),
            "gt_image_path": data.get("gt_image", ""),
            "quality": None,
            "findings": [],
        }

        # Build region_id → finding map
        findings_by_region = defaultdict(dict)
        for result in ann.get("result", []):
            from_name = result.get("from_name", "")
            region_id = result.get("id", "")
            value = result.get("value", {})

            if from_name == "quality_label":
                choices = value.get("choices", [])
                if choices:
                    sample["quality"] = choices[0]
            elif from_name == "finding_category":
                labels = value.get("rectanglelabels", [])
                findings_by_region[region_id]["category"] = labels[0] if labels else ""
                findings_by_region[region_id]["bbox"] = {
                    "x_min_norm": round(value.get("x", 0) * 10, 1),
                    "y_min_norm": round(value.get("y", 0) * 10, 1),
                    "x_max_norm": round((value.get("x", 0) + value.get("width", 0)) * 10, 1),
                    "y_max_norm": round((value.get("y", 0) + value.get("height", 0)) * 10, 1),
                }
            elif from_name == "finding_observation":
                texts = value.get("text", [])
                findings_by_region[region_id]["observation"] = texts[0] if texts else ""

        sample["findings"] = list(findings_by_region.values())
        return sample
```

## 5. Project State 模型

```python
class ProjectState(BaseModel):
    """Persisted project state for resume and status tracking."""

    project_ref: str
    project_id: int
    dataset_path: str
    adapter: str
    config_version: int = 1
    status: Literal["CREATED", "IMPORTING", "READY", "ERROR"]
    next_batch: int = 0
    imported_count: int = 0
    total_tasks: int = 0
    last_error: str | None = None
    idempotency_key: str | None = None
    created_at: str


class ManifestBatch(BaseModel):
    path: str            # tasks-00001.json
    task_count: int
    sha256: str
    index: int


class ManifestData(BaseModel):
    project_ref: str
    dataset_path: str
    converter: str       # "avi_train"
    converter_version: int = 1
    config_hash: str     # sha256 of label_config.xml
    total_tasks: int
    batches: list[ManifestBatch]
```

### 5.1 project_state.json 存储路径

```
/.pyromind-agent/label-studio/<project_ref>/
├── label_config.xml
├── project_state.json
└── manifests/
    ├── manifest.json
    └── tasks-00001.json ...
```

### 5.2 Idempotent Retry

```python
def _handle_create(self, action, conversation):
    # Check idempotency
    if action.idempotency_key:
        existing = self._find_by_idempotency_key(action.idempotency_key)
        if existing is not None:
            state = self._load_state(existing.project_ref)
            if state.status == "IMPORTING" and state.next_batch < state.total_batches:
                # Resume from next_batch
                return self._resume_import(existing.project_ref, state)
            # Already ready
            return self._state_to_observation(state)

    # ... normal create flow
```

## 6. Router 集成（pyromind_router.py）

### 6.1 新增常量

```python
LABEL_STUDIO_BASE_URL_SECRET = "LABEL_STUDIO_BASE_URL"
LABEL_STUDIO_TOKEN_SECRET = "LABEL_STUDIO_TOKEN"
LABEL_STUDIO_MEDIA_GATEWAY_SECRET = "LABEL_STUDIO_MEDIA_GATEWAY"
```

### 6.2 Tool 构建函数

```python
def _build_label_studio_tool(
    http_request: Request,
    extra: dict[str, Any],
) -> tuple[Tool, dict[str, SecretSource]]:
    """Build label_studio_project with server-side auth wiring."""

    # Label Studio auth token comes from extra config (server-side injection)
    ls_token = extra.get("label_studio_token")
    ls_base_url = extra.get("label_studio_base_url", "")
    media_gateway = extra.get("label_studio_media_gateway", "")

    params: dict[str, Any] = {}
    secrets: dict[str, SecretSource] = {}

    if isinstance(ls_base_url, str) and ls_base_url:
        params["ls_base_url"] = ls_base_url
    if isinstance(media_gateway, str) and media_gateway:
        params["media_gateway_url"] = media_gateway

    if isinstance(ls_token, str) and ls_token:
        params["ls_token_secret"] = LABEL_STUDIO_TOKEN_SECRET
        secrets[LABEL_STUDIO_TOKEN_SECRET] = StaticSecret(
            value=SecretStr(ls_token)
        )

    return Tool(name=LabelStudioProjectTool.name, params=params), secrets
```

### 6.3 注册到 Agent

在 `_build_agent()` 的 `extra_tools` 列表中加入：

```python
label_studio_tool, label_studio_secrets = _build_label_studio_tool(
    http_request, request.extra
)
# ...
extra_tools=[
    # ... existing tools
    label_studio_tool,
]
# ...
secrets={
    # ... existing secrets
    **label_studio_secrets,
}
```

### 6.4 Skill Allow-list

```python
_PYROMIND_SKILL_NAMES = [
    # ... existing
    "label-studio",
]
```

## 7. SSO 集成（PyroMind 侧，方案 B）

不改 Label Studio 源码。在 PyroMind 平台（或 agent-server）新增一个 SSO 跳转 endpoint。

### 7.1 流程

```
用户点击 open_url
  → GET /sso/label-studio?project_id=123
  → PyroMind SSO endpoint（与 LS 同域）:
      1. 验证 PyroMind session cookie
      2. 查/创建 Label Studio 用户（通过 LS Admin API 或直接 Django ORM）
      3. 用确定性密码调 LS login API
      4. 拿 sessionid cookie
      5. redirect 到 LS 页面，附带 sessionid cookie
  → 浏览器 302 到 /label-studio/projects/123/data，已登录
```

### 7.2 实现位置

SSO endpoint 不放在 SDK/Tool 层，放在 PyroMind 平台的 Web 层（如 agent-server 的 FastAPI router 或者独立服务）。Tool 只负责生成 `open_url` 字符串。

```python
# agent-server 层（非 SDK），示例
@router.get("/sso/label-studio")
async def sso_label_studio(request: Request, project_id: int):
    uid = get_current_user_id(request)

    # Ensure Label Studio user exists
    ls_email = f"pyromind_{uid}@pyromind.internal"
    ls_password = hashlib.sha256(
        f"{uid}:{LS_SSO_SERVER_SECRET}".encode()
    ).hexdigest()
    await ensure_label_studio_user(ls_email, ls_password, uid)

    # Login and get session cookie
    session = httpx.AsyncClient()
    resp = await session.post(
        f"{LABEL_STUDIO_URL}/user/login",
        data={"email": ls_email, "password": ls_password},
    )

    # Forward session cookie to browser
    response = RedirectResponse(
        url=f"{LABEL_STUDIO_URL}/projects/{project_id}/data",
        status_code=302,
    )
    for name, value in session.cookies.items():
        if name == "sessionid":
            response.set_cookie(name, value, path="/", httponly=True)
    return response
```

### 7.3 Media Gateway

Media Gateway 是一个独立服务（或反向代理的 location），负责：

1. 验证当前用户 UID（从 PyroMind session cookie）
2. 验证媒体引用签名（HMAC，防篡改）
3. 从 S3/MinIO 流式读取原始图片
4. 支持 Range、ETag

```
URL 格式: /label-studio/pyromind-media/<hmac-signature>/<encoded-storage-path>
```

签名生成：

```python
def sign_media_ref(storage_path: str, secret: str) -> str:
    sig = hmac.new(secret.encode(), storage_path.encode(), hashlib.sha256).hexdigest()
    encoded_path = base64.urlsafe_b64encode(storage_path.encode()).decode()
    return f"{sig[:16]}/{encoded_path}"
```

Media Gateway 不在 SDK 范围内，是平台基础设施。SDK 侧只需要知道 base URL 用于生成签名 URL。

## 8. 数据流安全约束

| 约束 | 实现方式 |
|------|----------|
| Tool 不接收 UID | Tool 通过 SecretRegistry 获取 Token，UID 不出现在 Action 参数中 |
| Storage 认证由服务端注入 | Router 层通过 StaticSecret 注入 cookie/token |
| Agent 不遍历完整数据集 | Converter 在 Tool 内运行，Agent 只看到 preview_dataset 的结果 |
| Media URL 签名防篡改 | HMAC + 短有效期（如需要） |
| 不复制原始图片 | Manifest 只保存 URL，Media Gateway 流式读取 |

## 9. 测试策略

### 9.1 单元测试

```
tests/tools/label_studio/
├── test_definition.py       # Action/Observation 模型
├── test_converter.py        # AVITrainToLabelStudioConverter / LabelStudioToAVITrainConverter
├── test_api_client.py       # LabelStudioAPIClient (mock HTTP)
├── test_executor.py         # LabelStudioProjectExecutor (mock LS API + Storage)
└── test_validate_config.py  # validate_label_config.py
```

### 9.2 集成测试（可选）

需要一个真实 Label Studio 实例。可用 Docker 启动：

```bash
docker run -d -p 8080:8080 heartexlabs/label-studio:latest
```

标记 `@pytest.mark.integration`，CI 中按需触发。

### 9.3 Skill 测试

Skill 本身是 Markdown + Python 脚本，通过 `tests/test_skills.py` 验证 SKILL.md 格式和 validate_label_config.py 可执行。

## 10. 实施阶段

### Phase 1: 核心 Tool + Converter（当前 PR）

- [ ] `openhands-tools/openhands/tools/label_studio/` 目录及核心模块
- [ ] `AVITrainToLabelStudioConverter` + `LabelStudioToAVITrainConverter`
- [ ] `label_studio_project` Tool 注册
- [ ] `pyromind_router.py` 集成
- [ ] `.agents/skills/label-studio/SKILL.md`
- [ ] `validate_label_config.py`
- [ ] `references/pcb-avi-review.xml`
- [ ] 单元测试

### Phase 2: SSO + Media Gateway（平台侧）

- [ ] PyroMind 平台 SSO endpoint
- [ ] Media Gateway 服务
- [ ] open_url 生成逻辑
- [ ] Label Studio 用户自动创建

### Phase 3: 失败恢复 + 导出（后续 PR）

- [ ] project_state.json 断点续导
- [ ] LabelStudioToAVITrainConverter 完善
- [ ] export 操作
- [ ] update_config 操作

## 11. 兼容性

- 不修改现有 Tool / Skill / Agent 行为
- 新增 Tool 仅在 `extra` 配置中提供 `label_studio_token` 时激活（feature flag 模式）
- 无 Label Studio 配置时，Tool 不注入 Agent，零影响
- Converter 和 Tool 独立于 Label Studio 版本（只依赖 REST API）

## 12. 与现有代码的模式对齐

| 模式 | 现有参考 | Label Studio 对应 |
|------|----------|-------------------|
| operation-based Tool | `TrainingAnalysisAction.operation` | `LabelStudioProjectAction.operation` |
| Secret 注入 | `StaticSecret` + `SecretRegistry` | `LABEL_STUDIO_TOKEN_SECRET` |
| Storage 上下文 | `_build_pyromind_storage_tools()` 的 headers/secret_headers 模式 | 同样复用 |
| Tool 注册 | `register_tool()` 模块级 | `register_tool(LabelStudioProjectTool.name, ...)` |
| Router 集成 | `_build_workflow_run_tool()` | `_build_label_studio_tool()` |
| Skill 加载 | `_PYROMIND_SKILL_NAMES` allow-list | 加入 `"label-studio"` |

