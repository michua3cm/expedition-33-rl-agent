#!/bin/bash
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
