#!/bin/bash
set -e

# 根据 .gitmodules 更新 git submodule
# 用法: bash scripts/update_proto.sh [版本号/分支/commit] [submodule路径]
# 示例:
#   bash scripts/update_proto.sh
#   bash scripts/update_proto.sh v1.0.0
#   bash scripts/update_proto.sh main docs-mintlify

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

GITMODULES=".gitmodules"

if [ ! -f "$GITMODULES" ]; then
    echo "❌ .gitmodules not found"
    exit 1
fi

list_submodule_paths() {
    git config --file "$GITMODULES" --get-regexp '^submodule\..*\.path$' \
        | awk '{print $2}'
}

resolve_latest_ref() {
    local latest_tag
    latest_tag="$(git tag -l 'v*' | sort -V | tail -1)"
    if [ -n "$latest_tag" ]; then
        echo "$latest_tag"
        return
    fi

    local default_branch
    default_branch="$(git symbolic-ref -q --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||')"
    echo "${default_branch:-main}"
}

update_one() {
    local path="$1"
    local version="$2"

    # 检查 submodule 是否存在 (submodule 的 .git 是文件,不是目录)
    if [ ! -e "$path/.git" ]; then
        echo "📥 Initializing submodule $path..."
        git submodule sync -- "$path"
        git submodule update --init --recursive -- "$path"
    fi

    if [ ! -e "$path/.git" ]; then
        echo "❌ submodule not found: $path"
        echo "Run: git submodule add <repo-url> $path"
        exit 1
    fi

    echo "🔄 Updating $path to ${version:-latest}..."

    # 进入 submodule
    cd "$path"

    # 获取最新标签和远程分支
    echo "📥 Fetching latest refs..."
    git fetch origin --tags --prune

    if [ -z "$version" ]; then
        version="$(resolve_latest_ref)"
        echo "✅ Latest version: $version"
    fi

    # 切换到指定版本
    echo "🔀 Switching to $version..."
    if git show-ref --verify --quiet "refs/remotes/origin/$version"; then
        git checkout -B "$version" "origin/$version"
    else
        git checkout "$version"
    fi

    # 确认版本
    local actual_version
    actual_version="$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)"
    echo "✅ $path now at: $actual_version"

    # 返回上级目录
    cd "$PROJECT_ROOT"

    git add "$path"
}

VERSION="${1:-}"
TARGET_PATH="${2:-}"

if [ -n "$TARGET_PATH" ]; then
    PATHS="$TARGET_PATH"
else
    PATHS="$(list_submodule_paths)"
fi

if [ -z "$PATHS" ]; then
    echo "❌ No submodules found in .gitmodules"
    exit 1
fi

if [ -z "$VERSION" ]; then
    echo "🔄 No version specified, updating to latest..."
fi

UPDATED=""
while IFS= read -r path; do
    [ -n "$path" ] || continue
    update_one "$path" "$VERSION"
    UPDATED="${UPDATED} ${path}"
done <<EOF
$PATHS
EOF

# 提交变更
echo ""
echo "📝 Committing submodule update..."
if [ -n "$VERSION" ]; then
    git commit -m "chore: update submodule${UPDATED} to $VERSION" || echo "No changes to commit"
else
    git commit -m "chore: update submodule${UPDATED} to latest" || echo "No changes to commit"
fi

echo ""
echo "✅ submodule updated${UPDATED}!"
echo ""
echo "Next steps:"
echo "  1. Review the submodule pointer: git submodule status"
echo "  2. Push when ready: git push"
