#!/bin/bash
#
# Local mirror of the CI checks. Run from the repository root:
#     ./scripts/check.sh
# (also wired up as a git pre-commit hook).
#
# Scope note: checks cover `src/dlml` and `tests`. The Streamlit web app under
# `src/dlml/web` is intentionally excluded from every gate for now (see the
# exclusions in pyproject.toml); conforming it to the package's standards is a
# tracked follow-up.

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
