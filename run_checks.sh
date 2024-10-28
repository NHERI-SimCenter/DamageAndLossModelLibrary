#!/bin/bash

# Spell-check
echo "Spell-checking."
echo
codespell .
if [ $? -ne 0 ]; then
    echo "Spell-checking failed."
    exit 1
fi

# Check formatting with ruff
echo "Checking formatting with 'ruff format --diff'."
echo
ruff format --diff
if [ $? -ne 0 ]; then
    echo "ruff format failed."
    exit 1
fi

# Run ruff for linting
echo "Linting with 'ruff check --fix'."
echo
ruff check --fix --output-format concise
if [ $? -ne 0 ]; then
    echo "ruff check failed."
    exit 1
fi

echo "All checks passed successfully."
echo
