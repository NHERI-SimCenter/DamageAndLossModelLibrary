#!/bin/bash
#
# Local mirror of the CI checks. Run from the repository root:
#     ./scripts/check.sh
# (also wired up as a git pre-commit hook).
#
# Scope note: checks are currently scoped to `src/dlml` and `tests`, the
# code authored for the v3 package. The legacy Streamlit app still lives at
# the repository root and migrates into `src/dlml/web` in a later step; once
# it does, it falls under these same checks automatically.

set -e

echo '=== Stage 1: Lint & Format ==='

echo 'Running Ruff Linter...'
uv run ruff check src/dlml tests

echo 'Running Ruff Formatter Check...'
uv run ruff format --check --diff src/dlml tests

echo 'Running Codespell...'
uv run codespell src/dlml tests README.md

echo '=== Stage 2: Type Check ==='

echo 'Running Mypy...'
uv run mypy

echo '=== Stage 3: Tests ==='

echo 'Running Pytest with Coverage...'
uv run pytest \
    --cov=dlml \
    --cov-branch \
    --cov-report=term-missing \
    --cov-fail-under=90 \
    -v

echo 'All checks passed!'
