#!/usr/bin/env bash
# Download a pinned mediamtx release binary into third_party/mediamtx.
# No root required; runs on macOS (arm64/amd64) and Linux (amd64/arm64).
set -euo pipefail

VERSION="v1.9.3"   # pinned for reproducibility
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_DIR}/third_party"
DEST="${DEST_DIR}/mediamtx"

if [[ -x "${DEST}" ]]; then
    echo "mediamtx already present at ${DEST}"
    exit 0
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64)  ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    arm64)   ARCH="arm64" ;;
    *) echo "unsupported arch: ${ARCH}" >&2; exit 1 ;;
esac

TARBALL="mediamtx_${VERSION}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/bluenviron/mediamtx/releases/download/${VERSION}/${TARBALL}"

mkdir -p "${DEST_DIR}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "downloading ${URL}"
curl -fsSL -o "${TMP}/${TARBALL}" "${URL}"
tar -xzf "${TMP}/${TARBALL}" -C "${TMP}" mediamtx
mv "${TMP}/mediamtx" "${DEST}"
chmod +x "${DEST}"
echo "installed ${DEST}"
"${DEST}" --version || true
