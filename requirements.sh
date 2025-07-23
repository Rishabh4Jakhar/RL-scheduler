#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if pip is installed
if ! command -v pip &>/dev/null && ! command -v pip3 &>/dev/null; then
    echo "pip is not installed. Please install pip first."
    exit 1
fi

# Use pip3 or fallback to pip
PIP_CMD=$(command -v pip3 || command -v pip)

echo "Using pip command: $PIP_CMD"

# Install required packages
$PIP_CMD install posix_ipc psutil scipy pyDOE gym stable-baselines3
