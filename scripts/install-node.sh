#!/bin/bash
# Install Node.js v22 LTS via NodeSource
set -e

if command -v node &>/dev/null; then
    NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VER" -ge 22 ]; then
        echo "Node.js $(node --version) already installed, skipping."
        exit 0
    fi
fi

echo "Installing Node.js v22..."
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
echo "Node.js $(node --version) installed."
