#!/bin/bash
set -e

# k8s_middleware Proto 版本更新脚本
# 用法: bash scripts/update_proto.sh [版本号]
# 示例:
#   bash scripts/update_proto.sh v1.0.0
#   bash scripts/update_proto.sh v1.1.0

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PROTO_DEPS_DIR="proto_deps"

if [ $# -eq 0 ]; then
    echo "🔄 No version specified, updating to latest..."
    if [ -d "$PROTO_DEPS_DIR" ]; then
        cd $PROTO_DEPS_DIR
        git fetch origin --tags 2>/dev/null || true
        LATEST_VERSION=$(git tag -l 'v*' | sort -V | tail -1)
        cd ..
        
        if [ -z "$LATEST_VERSION" ]; then
            echo "❌ No versions found"
            exit 1
        fi
        
        VERSION=$LATEST_VERSION
        echo "✅ Latest version: $VERSION"
    else
        echo "❌ proto_deps submodule not found"
        echo "Run: git submodule add <repo-url> proto_deps"
        exit 1
    fi
else
    VERSION=$1
fi

echo "🔄 Updating proto_deps to $VERSION..."

# 检查 submodule 是否存在 (submodule 的 .git 是文件,不是目录)
if [ ! -e "$PROTO_DEPS_DIR/.git" ]; then
    echo "❌ proto_deps submodule not found"
    echo "Run: git submodule add <repo-url> proto_deps"
    exit 1
fi

# 进入 submodule
cd $PROTO_DEPS_DIR

# 获取最新标签
echo "📥 Fetching latest tags..."
git fetch origin --tags

# 切换到指定版本
echo "🔀 Switching to $VERSION..."
git checkout $VERSION

# 确认版本
ACTUAL_VERSION=$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)
echo "✅ Now at: $ACTUAL_VERSION"

# 返回上级目录
cd ..

# 提交变更
echo ""
echo " Committing submodule update..."
git add $PROTO_DEPS_DIR
git commit -m "chore: update proto_deps to $VERSION" || echo "No changes to commit"

echo ""
echo "✅ proto_deps updated to $VERSION!"
echo ""
echo "Next steps:"
echo "  1. Restart the application to apply changes"
echo "  2. Test the integration: python -c 'from pyromind_proto.proto import kv_pb2; print(\"OK\")'"
