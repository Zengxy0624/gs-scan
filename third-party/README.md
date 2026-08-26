# third-party/

Vendored dependencies. Rules:

1. **Never modify `gsplat/`.** It is a pinned, unmodified checkout of the
   official gsplat repository (git submodule). `build_run_manifest.py` records
   its commit and refuses dirty state — a modified submodule breaks run
   provenance for every training result. Project-side extensions belong in
   `scripts/` (see `train_experiment.py`, which subclasses the example trainer
   without touching the submodule).
2. **`openblas-threaded/` is a deliberate vendored binary, not a leftover.**
   - `libopenblas-pthread.so.0` — Ubuntu's pthread-flavored OpenBLAS 0.3.26,
     preloaded via `LD_PRELOAD` to interpose the crash-prone BLAS variants
     that pycolmap/faiss otherwise use during feature matching.
   - `libgfortran.so.5` — its runtime dependency, resolved via
     `LD_LIBRARY_PATH`.
   - `README.md` inside documents provenance (sha256), the two crash modes
     (serial-BLAS races; thread-metadata slot overflow beyond 64 concurrent
     callers), and the launch-time environment contract
     (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=32`, set before process
     start — the shim reads them in its load-time constructor).

Adding anything new here: include a README with provenance (source, version,
hashes) and the reason it must be vendored rather than installed.
