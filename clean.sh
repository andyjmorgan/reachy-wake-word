#!/bin/bash
# Clean virtual environment for Reachy Wake Word Tester

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -d "venv" ]; then
    echo "Removing virtual environment..."
    rm -rf venv
    echo "✓ Cleaned!"
    echo ""
    echo "Run ./setup.sh to reinstall"
else
    echo "Nothing to clean - venv doesn't exist"
fi
