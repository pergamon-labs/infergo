#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${(%):-%N}")/.." && pwd)"
CACHE_DIR="${ROOT_DIR}/.cache/libtorch"
ARCHIVE_PATH="${ROOT_DIR}/.cache/libtorch-macos-arm64-2.10.0.zip"
DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.10.0.zip"

mkdir -p "${ROOT_DIR}/.cache"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Downloading libtorch to ${ARCHIVE_PATH}..."
  curl -L "${DOWNLOAD_URL}" -o "${ARCHIVE_PATH}"
fi

if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "Extracting libtorch into ${CACHE_DIR}..."
  tmp_dir="${ROOT_DIR}/.cache/libtorch-extract.$$"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"
  unzip -q "${ARCHIVE_PATH}" -d "${tmp_dir}"
  rm -rf "${CACHE_DIR}"
  mv "${tmp_dir}/libtorch" "${CACHE_DIR}"
  rm -rf "${tmp_dir}"
fi

export CGO_ENABLED=1
export CGO_CXXFLAGS="-I${CACHE_DIR}/include -I${CACHE_DIR}/include/torch/csrc/api/include"
export CGO_LDFLAGS="-L${CACHE_DIR}/lib -Wl,-rpath,${CACHE_DIR}/lib"

echo "Configured libtorch from ${CACHE_DIR}"
echo "CGO_ENABLED=${CGO_ENABLED}"
echo "CGO_CXXFLAGS=${CGO_CXXFLAGS}"
echo "CGO_LDFLAGS=${CGO_LDFLAGS}"
