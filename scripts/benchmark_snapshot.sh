#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${(%):-%N}")/.." && pwd)"
DEFAULT_DIR="${ROOT_DIR}/benchmarks/local"
COUNT="5"
BENCH_REGEX="."
OUTPUT_PATH=""

usage() {
  cat <<'EOF'
Usage:
  ./scripts/benchmark_snapshot.sh [-out <path>] [-count <n>] [-bench <regex>]

Captures the current InferGo benchmark suite into a local raw-output file that
can be compared later with benchstat.

Defaults:
  - output path: ./benchmarks/local/bench-<timestamp>.txt
  - count: 5
  - bench regex: .

Examples:
  ./scripts/benchmark_snapshot.sh
  ./scripts/benchmark_snapshot.sh -count 3 -out /tmp/infergo-before.txt
  ./scripts/benchmark_snapshot.sh -bench '^BenchmarkLoad' -out /tmp/load-only.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -out)
      [[ $# -ge 2 ]] || { echo "missing value for -out" >&2; usage >&2; exit 2; }
      OUTPUT_PATH="$2"
      shift 2
      ;;
    -count)
      [[ $# -ge 2 ]] || { echo "missing value for -count" >&2; usage >&2; exit 2; }
      COUNT="$2"
      shift 2
      ;;
    -bench)
      [[ $# -ge 2 ]] || { echo "missing value for -bench" >&2; usage >&2; exit 2; }
      BENCH_REGEX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
TIMESTAMP_LOCAL="$(date '+%Y-%m-%d %H:%M:%S %Z')"

if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${DEFAULT_DIR}/bench-${TIMESTAMP_UTC}.txt"
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

if [[ "${OUTPUT_PATH}" == *.txt ]]; then
  META_PATH="${OUTPUT_PATH%.txt}.meta"
else
  META_PATH="${OUTPUT_PATH}.meta"
fi

cat > "${META_PATH}" <<EOF
timestamp_utc=${TIMESTAMP_UTC}
timestamp_local=${TIMESTAMP_LOCAL}
repo_root=${ROOT_DIR}
go_version=$(go version)
goos=$(go env GOOS)
goarch=$(go env GOARCH)
count=${COUNT}
bench_regex=${BENCH_REGEX}
EOF

echo "Running InferGo benchmark suite..."
(
  cd "${ROOT_DIR}"
  go test ./infer/packs ./infer/httpserver -run '^$' -bench "${BENCH_REGEX}" -benchmem -count "${COUNT}"
) | tee "${OUTPUT_PATH}"

echo
echo "Saved raw benchmark output to ${OUTPUT_PATH}"
echo "Saved benchmark metadata to ${META_PATH}"
