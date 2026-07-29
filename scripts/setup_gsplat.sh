#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"
VENV="${GSPLAT_ROOT}/.venv"
PYTHON="${VENV}/bin/python"
TORCH_VERSION="${GSPLAT_TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${GSPLAT_TORCHVISION_VERSION:-0.24.1}"
TORCH_BACKEND="${GSPLAT_TORCH_BACKEND:-cu130}"
PYCOLMAP_PACKAGE="${GSPLAT_PYCOLMAP_PACKAGE:-pycolmap>=3.13,<4}"

if [[ ! -f "${GSPLAT_ROOT}/setup.py" ]]; then
    echo "gsplat submodule is missing. Run: git submodule update --init --recursive" >&2
    exit 1
fi

git -C "${PROJECT_ROOT}" submodule update --init --recursive third-party/gsplat

if [[ ! -x "${PYTHON}" ]]; then
    uv venv --python 3.11 "${VENV}"
fi

# The pinned upstream example environment currently requires this exact
# PyTorch pair. Keep it isolated inside the upstream checkout.
uv pip install \
    --python "${PYTHON}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --torch-backend "${TORCH_BACKEND}"

# Preinstall build dependencies because gsplat and some example accelerators
# compile against the selected PyTorch environment.
uv pip install \
    --python "${PYTHON}" \
    setuptools wheel ninja "numpy>=2,<3" rich

# Install PyCOLMAP explicitly so the environment can use its CPU wheel alongside
# Torch cu130 without requiring the CUDA 12 runtime from pycolmap-cuda12.
FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${FILTERED_REQUIREMENTS}"' EXIT
grep -Ev '^[[:space:]]*pycolmap([[:space:]]|[<>=!~])' \
    "${GSPLAT_ROOT}/examples/requirements.txt" > "${FILTERED_REQUIREMENTS}"

uv pip install \
    --python "${PYTHON}" \
    --requirement "${FILTERED_REQUIREMENTS}" \
    --no-build-isolation \
    --torch-backend "${TORCH_BACKEND}"
uv pip uninstall --python "${PYTHON}" pycolmap pycolmap-cuda12 || true
uv pip install --python "${PYTHON}" "${PYCOLMAP_PACKAGE}" plyfile

# Installing the checked-out source keeps the experiment pinned to the
# submodule commit instead of silently substituting the PyPI release.
uv pip install \
    --python "${PYTHON}" \
    --editable "${GSPLAT_ROOT}" \
    --no-build-isolation

uv pip check --python "${PYTHON}"
uv run --project "${GSPLAT_ROOT}" --no-sync python - <<'PY'
import torch
import gsplat

print(f"gsplat {gsplat.__version__}")
print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA kernel check: {torch.ones(1, device='cuda').square()}")
else:
    print("CUDA is not available; run the kernel check from a GPU allocation.")
PY

echo "gsplat environment ready: ${VENV}"
