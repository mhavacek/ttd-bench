#!/usr/bin/env bash
# Download a static ffmpeg binary (with libx264) into third_party/ffmpeg.
#
# Why: some HPC ffmpeg modules (e.g. MetaCentrum's spack build) are compiled
# WITHOUT libx264 for licensing reasons, so the replay's H.264 encode fails
# ("Unknown encoder 'libx264'"). John Van Sickle's static builds are GPL,
# self-contained (no system deps), and include libx264. On macOS this is a
# no-op — use Homebrew's ffmpeg (brew install ffmpeg), which has libx264.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_DIR}/third_party"
DEST="${DEST_DIR}/ffmpeg"

if [[ -x "${DEST}" ]] && "${DEST}" -hide_banner -encoders 2>/dev/null | grep -q libx264; then
    echo "static ffmpeg with libx264 already present at ${DEST}"
    exit 0
fi

OS="$(uname -s)"
if [[ "${OS}" != "Linux" ]]; then
    echo "static build is Linux-only; on ${OS} install ffmpeg via your package"
    echo "manager (e.g. 'brew install ffmpeg') — it includes libx264."
    exit 0
fi

ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64)  JVS_ARCH="amd64" ;;
    aarch64) JVS_ARCH="arm64" ;;
    *) echo "unsupported arch: ${ARCH}" >&2; exit 1 ;;
esac

URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${JVS_ARCH}-static.tar.xz"
mkdir -p "${DEST_DIR}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "downloading ${URL}"
curl -fsSL -o "${TMP}/ffmpeg.tar.xz" "${URL}"
tar -xf "${TMP}/ffmpeg.tar.xz" -C "${TMP}"
# tarball extracts to ffmpeg-<version>-<arch>-static/
FF="$(find "${TMP}" -maxdepth 2 -name ffmpeg -type f | head -1)"
if [[ -z "${FF}" ]]; then
    echo "ffmpeg binary not found in tarball" >&2
    exit 1
fi
mv "${FF}" "${DEST}"
chmod +x "${DEST}"

if "${DEST}" -hide_banner -encoders 2>/dev/null | grep -q libx264; then
    echo "installed ${DEST} (libx264 OK)"
    "${DEST}" -version | head -1
else
    echo "WARNING: installed ffmpeg but libx264 not detected" >&2
    exit 1
fi
