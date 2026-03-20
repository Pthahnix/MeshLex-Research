#!/bin/bash
# Clone repo and set up workspace
# Usage: bash deploy-context.sh '<REPO_URL>' '<WORKSPACE_DIR>'
set -e

REPO_URL="$1"
WORKSPACE_DIR="$2"

if [ -z "$REPO_URL" ] || [ -z "$WORKSPACE_DIR" ]; then
    echo "Usage: deploy-context.sh <REPO_URL> <WORKSPACE_DIR>"
    exit 1
fi

REPO_NAME=$(basename "$REPO_URL" .git)
TARGET="$WORKSPACE_DIR/$REPO_NAME"

if [ -d "$TARGET/.git" ]; then
    echo "Repo already cloned at $TARGET, pulling latest..."
    su - cc -c "cd $TARGET && git pull"
else
    echo "Cloning $REPO_URL to $TARGET..."
    su - cc -c "git clone $REPO_URL $TARGET"
fi

chown -R cc:cc "$TARGET"
echo "Repo ready at $TARGET"
