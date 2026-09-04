# Pthread OpenBLAS shim for pycolmap vocabulary-tree matching

## The problem

The `pycolmap` PyPI wheel statically links a **serial** OpenBLAS
(`openblas_get_num_procs() == 1`). COLMAP's faiss-based vocabulary-tree
retrieval calls BLAS (`sgemm_`) from many OpenMP worker threads at once —
from `faiss::knn_L2sqr` during visual-index assignment and unconditionally
from the ITQ re-ranking transform (`LinearTransform::apply_noalloc`). A serial
OpenBLAS has a buffer table sized for one caller; under concurrent callers it
falls into a racy fallback and dies:

```
OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
BLAS : Bad memory unallocation! : ...
*** SIGSEGV ... blas_memory_alloc ... sgemm_ ... faiss::knn_L2sqr ...
```

Thread-count environment variables cannot fix this (the crash scales with the
number of concurrent callers, and one caller means single-core retrieval).
Because the wheel exports `sgemm_` as an interposable dynamic symbol,
`LD_PRELOAD`-ing a thread-safe OpenBLAS reroutes every BLAS call, including
faiss's internal ones.

## Why it must be the pthread flavor

Both wrong flavors produced real multi-hour crashes on this project's
8,012-image ambulance reconstruction (256-CPU host):

- **Serial build** (statically linked in the wheel): crashed within the first
  few images of index assignment.
- **OpenMP flavor** (`libopenblaso` 0.3.15 extracted from a faiss-cpu wheel):
  nondeterministically racy when invoked from foreign OpenMP teams — one run
  crashed at image 1,144, another at image 5,930, while a 300-second spot
  test passed. Do not use OpenMP-flavor builds here.
- **Pthread flavor** (this shim): designed for arbitrary concurrent callers;
  the build the numpy/scipy ecosystem exercises from thread pools everywhere.
  Validated on 2026-08-13 by running the complete vocabulary-tree pairing
  stage (indexing all 8,012 images, then per-image retrieval + guided SIFT
  matching) against the real feature database with zero warnings and zero
  crashes.

The COLMAP wrapper (`scripts/run_object_colmap.sh`) applies, for pip-wheel
interpreters (the cuDSS build links a threaded BLAS and skips it):

```
LD_PRELOAD=third-party/openblas-threaded/libopenblas-pthread.so.0
LD_LIBRARY_PATH=third-party/openblas-threaded   # resolves vendored libgfortran
OMP_NUM_THREADS<=64      # concurrent callers stay inside MAX_THREADS=64
OPENBLAS_NUM_THREADS=1   # each call computes inline in its calling thread
```

`tests/test_openblas_shim.py` pins these safety properties (pthread flavor,
MAX_THREADS >= 64, unsuffixed exported symbols, correctness under 48
concurrent callers).

## Provenance

Extracted unmodified from Ubuntu 24.04 packages on 2026-08-13 (no root
needed: `apt-get download`, then `dpkg -x`):

| file | package | sha256 |
| --- | --- | --- |
| `libopenblas-pthread.so.0` | `libopenblas0-pthread 0.3.26+ds-1ubuntu0.1` | `bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e` |
| `libgfortran.so.5` | `libgfortran5 14.2.0-4ubuntu2~24.04.1` | `342618ccfebe840446e4ef2b7893f4161b2084845e570f42d68a70bfde58a58d` |

Build config: `OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY
Cooperlake MAX_THREADS=64` (`openblas_get_parallel() == 1` = pthread).
`libgfortran.so.5` is vendored because the host lacks it; its own
dependencies (`libquadmath`, `libgcc_s`) are standard system libraries.

To regenerate: `apt-get download libopenblas0-pthread libgfortran5`, extract
with `dpkg -x`, and copy the real files (dereference symlinks) here.

This shim can be retired once a pycolmap wheel ships a thread-safe BLAS —
check with `ctypes` that `openblas_get_num_procs()` on `pycolmap/_core*.so`
reports more than 1. Checked 2026-09-01: `pycolmap-cuda12` 4.2.0 still
reports 1 and exports `sgemm_`/`dgemm_` via `R_X86_64_JUMP_SLOT`, so the shim
interposes it too.

## Thread budget (third crash mode)

The concurrent-caller count is COLMAP worker threads **plus** OpenMP workers.
With `--num-threads -1` (256 on shenlong-gpu-02) and `OMP_NUM_THREADS=64`,
vocabulary-tree indexing printed "precompiled NUM_THREADS exceeded, adding
auxiliary array" and segfaulted in `sgemm_incopy_COOPERLAKE` within seconds.
Keep both at 32 (`scripts/run_object_colmap.sh` enforces this whenever
the shim is on); the mapping stage may then raise the shim to 48 threads
because bundle adjustment has a single BLAS caller.
