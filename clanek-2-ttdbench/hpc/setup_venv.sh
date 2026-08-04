#!/bin/bash
# Build the cluster venv from scratch. Run from anywhere:
#
#   bash clanek-2-ttdbench/hpc/setup_venv.sh          # foreground
#   nohup bash clanek-2-ttdbench/hpc/setup_venv.sh > ~/venv.log 2>&1 &
#
# Two pip processes writing one venv corrupt it (a half-uninstalled torch is
# indistinguishable from a broken wheel), so this takes a lock and refuses to
# start twice. It always rebuilds from zero rather than repairing: pip's HTTP
# cache makes a rebuild cheap, and a known-good venv is worth more than a
# salvaged one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
VENV="${ROOT}/.venv"
LOCK="${ROOT}/.venv-build.lock"

# mkdir is atomic on NFS; a stale lock is a directory you can simply remove
if ! mkdir "${LOCK}" 2>/dev/null; then
    echo "!! another build holds ${LOCK}" >&2
    echo "   if you are certain no build is running: rmdir ${LOCK}" >&2
    exit 1
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

# Match the relative form: the same venv appears as both /storage/... and
# /auto/... (automount) and pip is usually invoked as `.venv/bin/pip`, so
# matching an absolute ${VENV} path misses the very process we care about.
# The [.] keeps this pattern from matching the pgrep invocation itself.
if pgrep -u "$(id -u)" -f "[.]venv/bin/pip" >/dev/null 2>&1; then
    echo "!! a pip from a .venv is still running -- refusing to touch anything" >&2
    pgrep -u "$(id -u)" -fa "[.]venv/bin/pip" >&2 || true
    echo "   wait for it, or kill it by PID, then re-run" >&2
    exit 1
fi

echo "=== 1/7 wiping ${VENV} ==="
rm -rf "${VENV}"

# MetaCentrum's system python has no ensurepip, hence --without-pip + get-pip
echo "=== 2/7 venv (python $(python3 --version 2>&1 | cut -d' ' -f2)) ==="
python3 -m venv --without-pip "${VENV}"
curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV}/bin/python"

# torch goes in FIRST, from the cu126 index, WITH its dependencies. The konos
# GPUs are GTX 1080 Ti = sm_61 (Pascal) and newer CUDA builds dropped that
# architecture, so the index matters. What also matters: the CUDA runtime now
# ships as separate nvidia-* wheels, so installing torch with --no-deps leaves
# it unable to import at all -- "libcudart.so.12: cannot open shared object
# file". Installing it before ultralytics is what avoids the whole
# install-then-replace dance that made --no-deps look necessary.
echo "=== 3/7 torch + torchvision from cu126 (GTX 1080 Ti is sm_61) ==="
# NOT -q: this pulls ~3-4 GB (torch plus the nvidia-* CUDA runtime wheels) and
# a silent log is indistinguishable from a hung one. --progress-bar off keeps
# it readable when redirected to a file.
"${VENV}/bin/pip" install --progress-bar off torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# Pin what we just installed. ultralytics declares a plain `torch` requirement
# and would happily pull a PyPI build over the cu126 one; a constraints file
# makes that impossible rather than merely unlikely.
echo "=== 4/7 pinning torch/nvidia so nothing replaces them ==="
"${VENV}/bin/pip" freeze | grep -iE '^(torch|torchvision|nvidia-|triton)' \
    > "${ROOT}/.venv-constraints.txt" || true
sed 's/^/   /' "${ROOT}/.venv-constraints.txt" | head -20
# An empty or torch-less constraints file would silently disable the pin and
# let the next install swap the cu126 build for a PyPI one, which only shows up
# on the GPU node. Fail here instead.
if ! grep -qiE '^torch==' "${ROOT}/.venv-constraints.txt"; then
    echo "!! no torch pin in ${ROOT}/.venv-constraints.txt -- pip freeze gave:" >&2
    "${VENV}/bin/pip" freeze | head -20 >&2
    exit 1
fi

# Two constraint files, no overlap between them: .venv-constraints.txt holds
# the cu126 torch stack we just installed, frozen-libs.txt holds the versions
# the frozen run actually used. The second one is what keeps a repair run
# comparable -- ultralytics drives local-gpu and remote, onnxruntime drives
# edge-sim, and a minor bump in either moves detections and timing, so without
# it the equivalence test would be measuring the mediamtx fix AND a library
# upgrade at once.
FROZEN_LIBS="${ROOT}/clanek-2-ttdbench/hpc/frozen-libs.txt"
echo "=== 5/7 ttdbench + ML backends (constrained to the frozen run) ==="
CONSTRAINTS=(-c "${ROOT}/.venv-constraints.txt")
if [[ -f "${FROZEN_LIBS}" ]]; then
    CONSTRAINTS+=(-c "${FROZEN_LIBS}")
else
    echo "   !! ${FROZEN_LIBS} missing -- versions will NOT match the frozen run" >&2
fi
"${VENV}/bin/pip" install -q "${CONSTRAINTS[@]}" -e common/
"${VENV}/bin/pip" install --progress-bar off "${CONSTRAINTS[@]}" ultralytics onnxruntime

echo "=== 6/7 verify imports ==="
# Importing torch is itself the test that the CUDA runtime wheels are present:
# _load_global_deps() dlopens libcudart/libcublasLt at import time and raises
# if the nvidia-* packages are missing.
"${VENV}/bin/python" - <<'PY'
import sys
import torch, torchvision, ultralytics, onnxruntime, ttdbench
print(f"  torch        {torch.__version__}")
print(f"  torch.cuda   {torch.version.cuda}")
print(f"  torchvision  {torchvision.__version__}")
print(f"  ultralytics  {ultralytics.__version__}")
print(f"  onnxruntime  {onnxruntime.__version__}")
print(f"  ttdbench     {ttdbench.__file__}")
# The frontend has no GPU, so is_available() is False here and that is fine --
# this script cannot test the device, only that the stack loads. The real GPU
# check is the two-cell smoke job in docs/DEPLOY-RERUN.md.
print(f"  cuda avail   {torch.cuda.is_available()}  (False on the frontend is expected)")
if not (torch.version.cuda or "").startswith("12.6"):
    sys.exit(f"!! torch is built for CUDA {torch.version.cuda}, need 12.6 -- "
             "anything newer fails on Pascal with 'no kernel image is "
             "available for execution on the device'")
print("  OK")
PY

# The versions must match the frozen run, not merely import. Compared against
# a real header committed alongside this script.
REF="${ROOT}/clanek-2-ttdbench/hpc/frozen-header.jsonl"
if [[ -f "${REF}" ]]; then
    echo "=== 7/7 environment vs frozen run ==="
    "${VENV}/bin/python" "${ROOT}/common/scripts/check_env_matches_frozen.py" "${REF}"
fi

echo "=== venv ready: ${VENV} ==="
