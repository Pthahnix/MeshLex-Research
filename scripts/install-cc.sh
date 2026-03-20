#!/bin/bash
# Install Claude Code CLI globally
set -e

if command -v claude &>/dev/null; then
    echo "Claude Code already installed: $(claude --version 2>/dev/null || echo 'unknown version')"
    exit 0
fi

echo "Installing Claude Code..."
npm install -g @anthropic-ai/claude-code
echo "Claude Code installed: $(claude --version 2>/dev/null || echo 'done')"
