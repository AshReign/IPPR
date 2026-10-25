#!/usr/bin/env bash
set -euo pipefail
P2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$P2_ROOT/run_python.sh" -m pytest tests "$@"
