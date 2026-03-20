#!/bin/bash
# Create 'cc' user with sudo, preserving GPU access
set -e

if id cc &>/dev/null; then
    echo "User 'cc' already exists, skipping."
    exit 0
fi

echo "Creating user 'cc'..."
useradd -m -s /bin/bash -G sudo cc
echo "cc ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/cc
chmod 0440 /etc/sudoers.d/cc

# Copy SSH keys from root
mkdir -p /home/cc/.ssh
cp /root/.ssh/authorized_keys /home/cc/.ssh/ 2>/dev/null || true
chown -R cc:cc /home/cc/.ssh
chmod 700 /home/cc/.ssh
chmod 600 /home/cc/.ssh/authorized_keys 2>/dev/null || true

echo "User 'cc' created."
