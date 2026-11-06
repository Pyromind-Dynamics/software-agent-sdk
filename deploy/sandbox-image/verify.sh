#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-pyromind-sandbox-all-in-one:local}"
PLATFORM="${PLATFORM:-linux/amd64}"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || die "docker is not installed"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "image is not loaded: ${IMAGE}"

IMAGE_PLATFORM="$(
    docker image inspect "${IMAGE}" \
        --format '{{.Os}}/{{.Architecture}}'
)"
[ "${IMAGE_PLATFORM}" = "${PLATFORM}" ] \
    || die "expected ${PLATFORM}, image is ${IMAGE_PLATFORM}"

docker run --rm --platform "${PLATFORM}" --entrypoint /bin/bash "${IMAGE}" -lc '
    set -euo pipefail

    python=/opt/pyromind/dataflow-venv/bin/python
    resources=/opt/pyromind/agent-resources

    test -x "${python}"
    "${python}" -c \
        "import importlib.metadata as m; assert m.version(\"open-dataflow\") == \"1.0.13\""
    "${python}" -m pip check
    "${python}" -c \
        "import cv2, matplotlib, numpy, PIL, torch, torchaudio, torchvision; assert torch.version.cuda is None"

    test -f "${resources}/manifest.json"
    test -f "${resources}/skills/data-processing/SKILL.md"
    test -f "${resources}/knowledge/basic/sandbox.mdx"
    jq -e "
        .schema_version == 1 and
        .skills_path == \"/opt/pyromind/agent-resources/skills\" and
        .knowledge_path == \"/opt/pyromind/agent-resources/knowledge\" and
        .dataflow_python == \"/opt/pyromind/dataflow-venv/bin/python\" and
        .dataflow_version == \"1.0.13\" and
        .torch_version == \"2.6.0+cpu\"
    " "${resources}/manifest.json" >/dev/null

    check="$("${python}" "${resources}/skills/data-processing/scripts/preparation/sandbox_sample.py" --check)"
    printf "%s\n" "${check}" | jq -e "
        .ready == true and
        .version == \"1.0.13\" and
        .interpreter == \"/opt/pyromind/dataflow-venv/bin/python\" and
        (.source == \"image\" or .source == \"configured\")
    " >/dev/null
'

printf 'Verified %s (%s)\n' "${IMAGE}" "${PLATFORM}"
