#!/usr/bin/env bash
set -Eeuo pipefail

# Build a CUDA + cuDSS pycolmap (GPU bundle adjustment) from source into a
# self-contained micromamba environment. Layout matches the original
# shenlong-gpu-01 build so scripts/run_object_colmap.sh auto-detects it:
#   <root>/bin/micromamba
#   <root>/mamba_root/envs/gpubuild/bin/python   <- COLMAP_PYTHON
#   <root>/src/{ceres-solver,colmap}             <- sources + build dirs
# Phases are idempotent (marker files under <root>); rerun to resume.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ROOT="${PROJECT_ROOT}/work/_gpu_colmap"
JOBS="32"
CUDA_ARCH="120"
CUDA_VERSION="12.9"
COLMAP_REF="4.2.0"
CERES_REF="master"
PHASES="env,ceres,colmap,pycolmap,verify"
FORCE=0
DRY_RUN=0
NICE_LEVEL="10"

usage() {
    cat <<'USAGE'
Build a GPU (CUDA + cuDSS) pycolmap for GPU bundle adjustment.

Usage:
  bash scripts/build_gpu_colmap.sh [options]

Options:
  --root DIR            Install root (default: work/_gpu_colmap).
  --jobs N              Parallel compile jobs (default: 32).
  --cuda-arch LIST      CMAKE_CUDA_ARCHITECTURES (default: 120 = Blackwell
                        RTX PRO 6000; use e.g. "90;120" for several GPUs).
  --cuda-version V      conda-forge CUDA toolkit version (default: 12.9; the
                        conda-forge libcudss-dev/libcudss pair is chosen to
                        match, Blackwell needs >= 12.8).
  --colmap-ref REF      COLMAP git tag/branch (default: 4.2.0, matching the
                        pycolmap-cuda12 wheel so databases stay compatible).
  --ceres-ref REF       Ceres git ref (default: master; cuDSS support is not in
                        any Ceres release yet).
  --phases LIST         Comma list of env,ceres,colmap,pycolmap,verify
                        (default: all, in order).
  --force               Redo the selected phases even if marked done.
  --nice N              Nice level for compiles (default: 10).
  --dry-run             Print the commands without creating anything.
  -h, --help            Show this help.

Environment:
  Runs entirely without root: micromamba + conda-forge provide the compilers,
  CUDA toolkit, cuDSS, faiss, SuiteSparse, pthreads OpenBLAS. Ceres is patched
  to compile CUDA kernels for the requested architecture (upstream hardcodes
  its list). Expect roughly 1-2 hours; run it under tmux.
USAGE
}

require_value() {
    if [[ -z "${2:-}" ]]; then echo "error: $1 requires a value" >&2; exit 2; fi
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) require_value "$1" "${2:-}"; ROOT="$2"; shift 2 ;;
        --jobs) require_value "$1" "${2:-}"; JOBS="$2"; shift 2 ;;
        --cuda-arch) require_value "$1" "${2:-}"; CUDA_ARCH="$2"; shift 2 ;;
        --cuda-version) require_value "$1" "${2:-}"; CUDA_VERSION="$2"; shift 2 ;;
        --colmap-ref) require_value "$1" "${2:-}"; COLMAP_REF="$2"; shift 2 ;;
        --ceres-ref) require_value "$1" "${2:-}"; CERES_REF="$2"; shift 2 ;;
        --phases) require_value "$1" "${2:-}"; PHASES="$2"; shift 2 ;;
        --nice) require_value "$1" "${2:-}"; NICE_LEVEL="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! "${JOBS}" =~ ^[0-9]+$ ]] || (( 10#${JOBS} < 1 )); then
    echo "error: --jobs must be a positive integer" >&2; exit 2
fi
if [[ ! "${CUDA_ARCH}" =~ ^[0-9]+(a?)(;[0-9]+a?)*$ ]]; then
    echo "error: --cuda-arch must be a ;-separated list of compute capabilities (e.g. 120 or 90;120)" >&2; exit 2
fi
if [[ ! "${CUDA_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "error: --cuda-version must look like 13.2" >&2; exit 2
fi
if [[ ! "${NICE_LEVEL}" =~ ^[0-9]+$ ]] || (( 10#${NICE_LEVEL} > 19 )); then
    echo "error: --nice must be 0..19" >&2; exit 2
fi
IFS=',' read -r -a PHASE_LIST <<< "${PHASES}"
for phase in "${PHASE_LIST[@]}"; do
    case "${phase}" in env|ceres|colmap|pycolmap|verify) ;; *) echo "error: unknown phase: ${phase}" >&2; exit 2 ;; esac
done

ROOT="$(realpath -m -- "${ROOT}")"
MAMBA="${ROOT}/bin/micromamba"
export MAMBA_ROOT_PREFIX="${ROOT}/mamba_root"
ENV_NAME="gpubuild"
PREFIX="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"
SRC="${ROOT}/src"
LOG_DIR="${PROJECT_ROOT}/work/_logs"
LOG_PATH="${LOG_DIR}/build_gpu_colmap.log"

print_command() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run() { print_command "$@"; if (( DRY_RUN == 0 )); then "$@"; fi; }
in_env() { run "${MAMBA}" run -n "${ENV_NAME}" "$@"; }
marker() { printf '%s/.done-%s' "${ROOT}" "$1"; }
phase_selected() { local p; for p in "${PHASE_LIST[@]}"; do [[ "${p}" == "$1" ]] && return 0; done; return 1; }
phase_done() { (( FORCE == 0 )) && [[ -f "$(marker "$1")" ]]; }
mark_done() { (( DRY_RUN == 1 )) || { date -u +%FT%TZ > "$(marker "$1")"; }; }
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

write_env_spec() {
    cat > "$1" <<SPEC
python=3.11
pip
numpy
pybind11
scikit-build-core
ninja
cmake
make
pkg-config
git
cxx-compiler
c-compiler
cuda-version=${CUDA_VERSION}
cuda-nvcc=${CUDA_VERSION}
cuda-cudart-dev
cuda-crt-dev_linux-64
libcublas-dev
libcusparse-dev
libcusolver-dev
libcurand-dev
libnvjitlink-dev
libcudss
libcudss-dev
libblas=*=*openblas
libcblas
liblapack
libopenblas=*=pthreads*
libboost-devel
suitesparse
eigen
eigen-abi-devel
freeimage
glog
gflags
glew
gmp
mpfr
flann
sqlite
vlfeat
lz4-c
metis
openimageio
fmt
tbb-devel
poselib
faiss
libfaiss
libgl-devel
libopengl-devel
libglu
xorg-libx11
xorg-libxext
libxcb
libcurl
openssl
libgomp
SPEC
}

phase_env() {
    log "phase env: micromamba + conda-forge toolchain (CUDA ${CUDA_VERSION}, cuDSS, faiss, SuiteSparse)"
    if [[ ! -x "${MAMBA}" ]]; then
        run mkdir -p "${ROOT}/bin"
        print_command curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest '|' tar -xj -C "${ROOT}" bin/micromamba
        if (( DRY_RUN == 0 )); then
            curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "${ROOT}" bin/micromamba
        fi
    fi
    local spec="${ROOT}/env_spec.txt"
    if (( DRY_RUN == 0 )); then write_env_spec "${spec}"; fi
    print_command write_env_spec "${spec}"
    if [[ -x "${PREFIX}/bin/python" ]] && (( FORCE == 0 )); then
        log "env exists: ${PREFIX} (updating to the spec)"
        run "${MAMBA}" install -y -n "${ENV_NAME}" -c conda-forge --override-channels -f "${spec}"
    else
        run "${MAMBA}" create -y -n "${ENV_NAME}" -c conda-forge --override-channels -f "${spec}"
    fi
    in_env python -m pip install --upgrade pip
    in_env python -c 'import sys; print("python", sys.version.split()[0])'
    in_env nvcc --version
}

patch_ceres_arch() {
    # Upstream resets CMAKE_CUDA_ARCHITECTURES to its own list (max sm_90);
    # append the requested architectures so kernels exist for this GPU.
    local cmake_file="$1"
    print_command patch_ceres_arch "${cmake_file}" "${CUDA_ARCH}"
    (( DRY_RUN == 0 )) || return 0
    python3 - "${cmake_file}" "${CUDA_ARCH}" <<'PY'
import sys
from pathlib import Path
path, archs = Path(sys.argv[1]), sys.argv[2].split(";")
text = path.read_text()
marker = "# 3DGS-Health: requested architectures"
if marker in text:
    print("ceres arch patch already applied")
    sys.exit(0)
anchor = '    list(APPEND CMAKE_CUDA_ARCHITECTURES "90")\n  endif()\n'
if text.count(anchor) != 1:
    sys.exit("ceres CMakeLists.txt layout changed; cannot patch CUDA architectures")
addition = anchor + f"  {marker}\n" + "".join(
    f'  list(APPEND CMAKE_CUDA_ARCHITECTURES "{a}")\n' for a in archs
) + "  list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)\n"
path.write_text(text.replace(anchor, addition))
print("patched ceres CUDA architectures:", archs)
PY
}

phase_ceres() {
    log "phase ceres: Ceres Solver (${CERES_REF}) with CUDA + cuDSS"
    run mkdir -p "${SRC}"
    if [[ ! -d "${SRC}/ceres-solver/.git" ]]; then
        run git clone --depth 1 --branch "${CERES_REF}" https://github.com/ceres-solver/ceres-solver.git "${SRC}/ceres-solver"
    fi
    patch_ceres_arch "${SRC}/ceres-solver/CMakeLists.txt"
    local build="${SRC}/ceres-build"
    (( FORCE == 0 )) || run rm -rf "${build}"
    local cudss_dir=""
    if (( DRY_RUN == 0 )); then
        cudss_dir="$(dirname "$(find "${PREFIX}" -name 'cudss-config.cmake' 2>/dev/null | head -1)")"
        if [[ -z "${cudss_dir}" || "${cudss_dir}" == "." ]]; then
            echo "error: cudss-config.cmake not found under ${PREFIX}; is libcudss installed?" >&2; return 1
        fi
    else
        cudss_dir="${PREFIX}/lib/cmake/cudss"
    fi
    in_env cmake -S "${SRC}/ceres-solver" -B "${build}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_PREFIX_PATH="${PREFIX}" \
        -DCUDAToolkit_ROOT="${PREFIX}" \
        -Dcudss_DIR="${cudss_dir}" \
        -DWITH_CUDA=ON \
        -DWITH_SUITESPARSE=ON \
        -DWITH_EIGENSPARSE=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF
    if (( DRY_RUN == 0 )); then
        if grep -q "cudss_DIR.*NOTFOUND" "${build}/CMakeCache.txt"; then
            echo "error: Ceres configured WITHOUT cuDSS (cudss_DIR NOTFOUND)" >&2; return 1
        fi
        log "cuDSS config: ${cudss_dir}"
    fi
    in_env nice -n "${NICE_LEVEL}" ninja -C "${build}" -j "${JOBS}"
    in_env ninja -C "${build}" install
    if (( DRY_RUN == 0 )); then
        if ! ldd "${PREFIX}/lib/libceres.so" | grep -q cudss; then
            echo "error: installed libceres.so is not linked against cuDSS" >&2; return 1
        fi
        log "libceres.so links cuDSS: $(ldd "${PREFIX}/lib/libceres.so" | grep cudss | awk '{print $1}' | tr '\n' ' ')"
    fi
}

phase_colmap() {
    log "phase colmap: COLMAP ${COLMAP_REF} with CUDA (arch ${CUDA_ARCH}), no GUI/ONNX"
    run mkdir -p "${SRC}"
    if [[ ! -d "${SRC}/colmap/.git" ]]; then
        run git clone --depth 1 --branch "${COLMAP_REF}" https://github.com/colmap/colmap.git "${SRC}/colmap"
    fi
    local build="${SRC}/colmap-build"
    (( FORCE == 0 )) || run rm -rf "${build}"
    in_env cmake -S "${SRC}/colmap" -B "${build}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_PREFIX_PATH="${PREFIX}" \
        -DCUDAToolkit_ROOT="${PREFIX}" \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DGUI_ENABLED=OFF \
        -DOPENGL_ENABLED=OFF \
        -DONNX_ENABLED=OFF \
        -DCGAL_ENABLED=OFF \
        -DLSD_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        -DCCACHE_ENABLED=OFF \
        -DIPO_ENABLED=OFF \
        -DFETCH_FAISS=OFF \
        -DFETCH_POSELIB=OFF \
        -DDOWNLOAD_ENABLED=ON
    in_env nice -n "${NICE_LEVEL}" ninja -C "${build}" -j "${JOBS}"
    in_env ninja -C "${build}" install
    if (( DRY_RUN == 0 )); then
        # COLMAP installs static archives; the CPU-fallback message is compiled
        # in only when Ceres lacks cuDSS, so its absence proves GPU BA support.
        local colmap_lib
        colmap_lib="$(ls "${PREFIX}"/lib/libcolmap_estimators.* 2>/dev/null | head -1 || true)"
        if [[ -z "${colmap_lib}" ]]; then
            echo "error: COLMAP estimators library not found under ${PREFIX}/lib" >&2; return 1
        fi
        if strings -a "${colmap_lib}" | grep -q "compiled without cuDSS support"; then
            echo "error: COLMAP was compiled with CERES_NO_CUDSS (GPU BA would fall back to CPU)" >&2; return 1
        fi
        log "COLMAP estimators built with cuDSS-enabled Ceres: $(basename "${colmap_lib}")"
        if [[ ! -x "${PREFIX}/bin/colmap" ]]; then
            echo "error: COLMAP binary was not installed to ${PREFIX}/bin/colmap" >&2; return 1
        fi
        log "COLMAP installed: ${PREFIX}/bin/colmap ($("${PREFIX}/bin/colmap" help 2>&1 | head -1))"
    fi
}

phase_pycolmap() {
    log "phase pycolmap: Python bindings into ${PREFIX}"
    in_env python "${SRC}/colmap/python/ci/install-build-requirements.py"
    in_env env \
        CMAKE_PREFIX_PATH="${PREFIX}" \
        CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${PREFIX} -DCUDAToolkit_ROOT=${PREFIX} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DCMAKE_BUILD_TYPE=Release" \
        SKBUILD_BUILD_DIR="${SRC}/pycolmap-build" \
        CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
        python -m pip install --no-build-isolation --no-deps -v "${SRC}/colmap"
}

phase_verify() {
    log "phase verify"
    in_env python - <<'PY'
import glob, os, subprocess, sys
import pycolmap
print("pycolmap", pycolmap.__version__, "has_cuda", pycolmap.has_cuda)
assert pycolmap.has_cuda, "pycolmap built without CUDA"
core = glob.glob(os.path.join(os.path.dirname(pycolmap.__file__), "_core*.so"))[0]
ldd = subprocess.run(["ldd", core], capture_output=True, text=True).stdout
ceres = [line.split()[2] for line in ldd.splitlines() if "libceres" in line and "=>" in line]
print("libceres:", ceres)
assert ceres, "pycolmap does not link libceres dynamically"
ceres_ldd = subprocess.run(["ldd", ceres[0]], capture_output=True, text=True).stdout
assert "cudss" in ceres_ldd, "libceres is not linked against cuDSS"
print("cuDSS linked:", [l.split()[0] for l in ceres_ldd.splitlines() if "cudss" in l])
opts = pycolmap.IncrementalPipelineOptions(); opts.ba_use_gpu = True
print("IncrementalPipelineOptions.ba_use_gpu ok; BLAS threads:", end=" ")
import ctypes
try:
    blas = ctypes.CDLL("libopenblas.so.0"); print(blas.openblas_get_num_procs(), "procs, parallel mode", blas.openblas_get_parallel())
except OSError as e:
    print("(no libopenblas.so.0:", e, ")")
print("VERIFY_OK")
PY
}

if (( DRY_RUN == 0 )); then
    mkdir -p "${LOG_DIR}" "${ROOT}"
    exec > >(tee -a "${LOG_PATH}") 2>&1
    log "=== build_gpu_colmap.sh start: root=${ROOT} jobs=${JOBS} arch=${CUDA_ARCH} cuda=${CUDA_VERSION} colmap=${COLMAP_REF} ceres=${CERES_REF} phases=${PHASES} force=${FORCE}"
else
    echo "Dry run: root=${ROOT} jobs=${JOBS} arch=${CUDA_ARCH} cuda=${CUDA_VERSION} colmap=${COLMAP_REF} ceres=${CERES_REF} phases=${PHASES}"
fi
CURRENT_PHASE=""
on_error() {
    local status=$?
    log "phase ${CURRENT_PHASE:-?}: FAILED (exit ${status}) at: ${BASH_COMMAND}"
    if (( DRY_RUN == 0 )); then
        log "BUILD_GPU_COLMAP_FAILED: see ${LOG_PATH}"
        rm -f "${ROOT}/BUILD_DONE"
    fi
    exit "${status}"
}
# Phases run as plain statements (not inside `if`), so every failing command
# aborts through errexit -> ERR trap. Never wrap a phase call in a condition.
trap on_error ERR
for phase in env ceres colmap pycolmap verify; do
    phase_selected "${phase}" || continue
    if phase_done "${phase}" && [[ "${phase}" != "verify" ]]; then
        log "phase ${phase}: already done ($(cat "$(marker "${phase}")")); use --force to redo"
        continue
    fi
    CURRENT_PHASE="${phase}"
    "phase_${phase}"
    mark_done "${phase}"
    log "phase ${phase}: done"
done
CURRENT_PHASE=""
if (( DRY_RUN == 1 )); then
    echo "Dry run complete; nothing was built."
    exit 0
fi
log "BUILD_GPU_COLMAP_DONE: interpreter ${PREFIX}/bin/python"
touch "${ROOT}/BUILD_DONE"
