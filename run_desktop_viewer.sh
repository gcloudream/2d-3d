#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  uv venv .venv
fi

uv pip install --python .venv/bin/python -r desktop_viewer/requirements.txt
.venv/bin/python desktop_viewer/app.py
