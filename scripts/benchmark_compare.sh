#!/usr/bin/env zsh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/benchmark_compare.sh <before.txt> <after.txt>

Compares two InferGo benchmark snapshot files using benchstat.

If benchstat is not installed locally, this script falls back to:
  go run golang.org/x/perf/cmd/benchstat@latest

Examples:
  ./scripts/benchmark_compare.sh /tmp/infergo-before.txt /tmp/infergo-after.txt
  benchstat /tmp/infergo-before.txt /tmp/infergo-after.txt
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

BEFORE_PATH="$1"
AFTER_PATH="$2"

if [[ ! -f "${BEFORE_PATH}" ]]; then
  echo "missing benchmark snapshot: ${BEFORE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${AFTER_PATH}" ]]; then
  echo "missing benchmark snapshot: ${AFTER_PATH}" >&2
  exit 1
fi

if command -v benchstat >/dev/null 2>&1; then
  exec benchstat "${BEFORE_PATH}" "${AFTER_PATH}"
fi

echo "benchstat not found in PATH; falling back to go run golang.org/x/perf/cmd/benchstat@latest" >&2
exec go run golang.org/x/perf/cmd/benchstat@latest "${BEFORE_PATH}" "${AFTER_PATH}"
