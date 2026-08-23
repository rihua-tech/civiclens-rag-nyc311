#!/bin/sh
set -eu

python -m scripts.bootstrap
exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
