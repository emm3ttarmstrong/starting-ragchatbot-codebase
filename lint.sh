#!/bin/bash

# Code quality check script
# Usage: ./lint.sh [check|fix]

set -e

MODE=${1:-check}

echo "Running code quality checks..."
echo "================================"

if [ "$MODE" = "fix" ]; then
    echo "Mode: FIX (applying formatting changes)"
    echo ""

    echo "Running isort..."
    uv run isort backend/ main.py

    echo "Running black..."
    uv run black backend/ main.py

    echo ""
    echo "Formatting complete!"
else
    echo "Mode: CHECK (no changes will be made)"
    echo ""

    echo "Checking import sorting with isort..."
    uv run isort --check-only --diff backend/ main.py

    echo "Checking formatting with black..."
    uv run black --check --diff backend/ main.py

    echo "Running flake8 linter..."
    uv run flake8 backend/ main.py

    echo ""
    echo "All checks passed!"
fi
