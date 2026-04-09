#!/bin/bash

echo "Running Claude Code session-start hook..."

# 1. Annihilate the Anthropic defaults and set your core Git identity
git config --global user.name "michua3cm"
git config --global user.email "202406466+michua3cm@users.noreply.github.com"

# 2. Disable Anthropic's forced fake cryptographic signing
git config --global commit.gpgsign false
git config --global --unset user.signingkey

echo "Git identity successfully updated and fake signing disabled."

set -euo pipefail

# Only run in remote (Claude Code on the web) environments.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
  fi

  cd "$CLAUDE_PROJECT_DIR"

  # Install all dependencies including dev group (pytest, ruff).
  # uv sync is idempotent and uses the locked uv.lock for reproducibility.
  uv sync --all-groups

  # Ensure the project root is on PYTHONPATH so `import vision`, etc. resolve.
  echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR\"" >> "$CLAUDE_ENV_FILE"