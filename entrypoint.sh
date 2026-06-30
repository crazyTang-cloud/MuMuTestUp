#!/usr/bin/env bash
set -euo pipefail

if [ "${SKIP_SAFE_DIRECTORY:-0}" != "1" ]; then
  git config --system --add safe.directory /data/david/project/mumutestup/repos || true
  git config --system --add safe.directory '*' || true
fi

exec "$@"
