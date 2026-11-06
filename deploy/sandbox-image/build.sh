#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_IMAGE="${BASE_IMAGE:-quay.io/jupyter/base-notebook:python-3.12}"
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-pyromind-sandbox-all-in-one}"
PLATFORM="${PLATFORM:-linux/amd64}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/dist}"
DATAFLOW_VERSION="${DATAFLOW_VERSION:-1.0.13}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
KEEP_BUILD_CONTEXT="${KEEP_BUILD_CONTEXT:-0}"

CONTEXT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pyromind-sandbox-image.XXXXXX")"

cleanup() {
    if [ "${KEEP_BUILD_CONTEXT}" = "1" ]; then
        printf 'Build context kept at %s\n' "${CONTEXT_DIR}"
    else
        rm -rf "${CONTEXT_DIR}"
    fi
}
trap cleanup EXIT

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

copy_resource() {
    local source_dir="$1"
    local source_parent
    local source_name

    source_parent="$(dirname "${source_dir}")"
    source_name="$(basename "${source_dir}")"

    (
        cd "${ROOT_DIR}/${source_parent}"
        COPYFILE_DISABLE=1 tar \
            --exclude='.DS_Store' \
            --exclude='.git' \
            --exclude='.ssh' \
            --exclude='.aws' \
            --exclude='.netrc' \
            --exclude='.env' \
            --exclude='.env.*' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='*.pyo' \
            --exclude='*.pem' \
            --exclude='*.key' \
            --exclude='id_rsa*' \
            -cf - "${source_name}"
    ) | tar -C "${CONTEXT_DIR}/agent-resources" -xf -
}

command -v docker >/dev/null 2>&1 || die "docker is not installed"
docker buildx version >/dev/null 2>&1 || die "docker buildx is not installed"

for resource_dir in .agents/skills knowledge; do
    [ -d "${ROOT_DIR}/${resource_dir}" ] || die "missing resource directory: ${resource_dir}"
    if find "${ROOT_DIR}/${resource_dir}" -type l -print -quit | grep -q .; then
        die "resource directory contains a symlink: ${resource_dir}"
    fi
done

mkdir -p "${CONTEXT_DIR}/agent-resources"
copy_resource ".agents/skills"
copy_resource "knowledge"

if find "${CONTEXT_DIR}/agent-resources" -type f \
    \( -name '.env' -o -name '.env.*' -o -name '*.pem' -o -name '*.key' \
       -o -name 'id_rsa*' -o -name '.netrc' \) \
    -print -quit | grep -q .; then
    die "refusing to package a resource that looks like a secret"
fi

if command -v sha256sum >/dev/null 2>&1; then
    HASH_COMMAND=(sha256sum)
else
    HASH_COMMAND=(shasum -a 256)
fi

RESOURCE_DIGEST="$(
    cd "${CONTEXT_DIR}"
    find agent-resources/skills agent-resources/knowledge -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 "${HASH_COMMAND[@]}" \
        | "${HASH_COMMAND[@]}" \
        | awk '{print $1}'
)"
RESOURCE_FILE_COUNT="$(
    find "${CONTEXT_DIR}/agent-resources/skills" \
         "${CONTEXT_DIR}/agent-resources/knowledge" \
         -type f | wc -l | tr -d ' '
)"

RESOURCE_REVISION="${RESOURCE_REVISION:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"
if ! git -C "${ROOT_DIR}" diff --quiet HEAD -- .agents/skills knowledge \
    || [ -n "$(git -C "${ROOT_DIR}" ls-files --others --exclude-standard -- .agents/skills knowledge)" ]; then
    RESOURCE_REVISION="${RESOURCE_REVISION}-dirty"
fi

printf '%s\n' \
    '{' \
    '  "schema_version": 1,' \
    "  \"resources_revision\": \"${RESOURCE_REVISION}\"," \
    "  \"resources_digest\": \"${RESOURCE_DIGEST}\"," \
    "  \"resource_file_count\": ${RESOURCE_FILE_COUNT}," \
    '  "skills_path": "/opt/pyromind/agent-resources/skills",' \
    '  "knowledge_path": "/opt/pyromind/agent-resources/knowledge",' \
    '  "dataflow_python": "/opt/pyromind/dataflow-venv/bin/python",' \
    "  \"dataflow_version\": \"${DATAFLOW_VERSION}\"," \
    "  \"torch_version\": \"${TORCH_VERSION}+cpu\"," \
    "  \"base_image\": \"${BASE_IMAGE}\"" \
    '}' \
    > "${CONTEXT_DIR}/agent-resources/manifest.json"

cp "${SCRIPT_DIR}/Dockerfile" "${CONTEXT_DIR}/Dockerfile"

printf 'Pulling base image %s for %s\n' "${BASE_IMAGE}" "${PLATFORM}"
docker pull --platform "${PLATFORM}" "${BASE_IMAGE}"

BASE_USER="$(docker image inspect --format '{{.Config.User}}' "${BASE_IMAGE}")"
RUNTIME_USER="${RUNTIME_USER:-${BASE_USER:-root}}"

REVISION_TAG="${IMAGE_REPOSITORY}:${RESOURCE_REVISION}"
LOCAL_TAG="${IMAGE_REPOSITORY}:local"

printf 'Building %s\n' "${REVISION_TAG}"
docker buildx build \
    --platform "${PLATFORM}" \
    --file "${CONTEXT_DIR}/Dockerfile" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "RUNTIME_USER=${RUNTIME_USER}" \
    --build-arg "RESOURCE_REVISION=${RESOURCE_REVISION}" \
    --build-arg "RESOURCE_DIGEST=${RESOURCE_DIGEST}" \
    --build-arg "RESOURCE_FILE_COUNT=${RESOURCE_FILE_COUNT}" \
    --build-arg "DATAFLOW_VERSION=${DATAFLOW_VERSION}" \
    --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}" \
    --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
    --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}" \
    --build-arg "TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}" \
    --build-arg "TORCH_CPU_INDEX_URL=${TORCH_CPU_INDEX_URL}" \
    --label "org.opencontainers.image.revision=${RESOURCE_REVISION}" \
    --label "ai.pyromind.resources.digest=${RESOURCE_DIGEST}" \
    --provenance=false \
    --tag "${REVISION_TAG}" \
    --tag "${LOCAL_TAG}" \
    --load \
    "${CONTEXT_DIR}"

if [ "${SKIP_VERIFY}" != "1" ]; then
    PLATFORM="${PLATFORM}" "${SCRIPT_DIR}/verify.sh" "${REVISION_TAG}"
fi

mkdir -p "${OUTPUT_DIR}"
PLATFORM_SLUG="${PLATFORM//\//-}"
ARCHIVE_PATH="${OUTPUT_DIR}/pyromind-sandbox-all-in-one-${RESOURCE_REVISION}-${PLATFORM_SLUG}.tar.gz"
TEMP_ARCHIVE="${ARCHIVE_PATH}.tmp"

printf 'Writing %s\n' "${ARCHIVE_PATH}"
docker save "${REVISION_TAG}" | gzip -1 > "${TEMP_ARCHIVE}"
mv "${TEMP_ARCHIVE}" "${ARCHIVE_PATH}"

printf 'Image: %s\n' "${REVISION_TAG}"
printf 'Alias: %s\n' "${LOCAL_TAG}"
printf 'Archive: %s\n' "${ARCHIVE_PATH}"
printf 'Resources: %s files, sha256 %s\n' "${RESOURCE_FILE_COUNT}" "${RESOURCE_DIGEST}"
