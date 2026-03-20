#!/bin/bash
# Configure API credentials for cc user
# Usage: bash setup-env.sh '<BASE_URL>' '<AUTH_TOKEN>' '<MODEL>'
set -e

BASE_URL="$1"
AUTH_TOKEN="$2"
MODEL="$3"

if [ -z "$BASE_URL" ] || [ -z "$AUTH_TOKEN" ] || [ -z "$MODEL" ]; then
    echo "Usage: setup-env.sh <BASE_URL> <AUTH_TOKEN> <MODEL>"
    exit 1
fi

# Write to cc user's .bashrc
cat >> /home/cc/.bashrc << ENVEOF

# Claude Code API credentials
export ANTHROPIC_BASE_URL="$BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$AUTH_TOKEN"
export ANTHROPIC_MODEL="$MODEL"
ENVEOF

chown cc:cc /home/cc/.bashrc
echo "API credentials configured in /home/cc/.bashrc"
