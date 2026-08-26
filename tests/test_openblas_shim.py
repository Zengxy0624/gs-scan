"""Guard the vendored OpenBLAS shim that vocab-tree matching depends on.

pycolmap's wheel statically links a serial OpenBLAS that segfaults when faiss
retrieval calls it from many OpenMP threads at once, so the pipeline scripts
LD_PRELOAD a thread-safe build from third-party/openblas-threaded. Two shim
mistakes have produced multi-hour reconstruction crashes before: a serial
build (any concurrent callers corrupt its buffer table) and an OpenMP-flavor
build (nondeterministically racy when called from foreign OpenMP teams). These
tests pin the properties that make a shim safe: pthread flavor, an adequate
compiled thread limit, interposable unsuffixed symbols, and survival under
many concurrent callers.
"""

from __future__ import annotations

import ctypes
import re
import threading
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIM_DIR = PROJECT_ROOT / "third-party" / "openblas-threaded"
SHIM = SHIM_DIR / "libopenblas-pthread.so.0"
SCRIPTS = (
    PROJECT_ROOT / "scripts" / "run_ambulance_colmap.sh",
    PROJECT_ROOT / "scripts" / "run_single_video_gs.sh",
)


def load_shim() -> ctypes.CDLL:
    # libgfortran is vendored beside the shim because the host may not
    # provide it; load it first so the shim's dependency resolves.
    vendored_gfortran = SHIM_DIR / "libgfortran.so.5"
    if vendored_gfortran.is_file():
        ctypes.CDLL(str(vendored_gfortran), mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(str(SHIM))


class OpenBlasShimTests(unittest.TestCase):
    def test_shim_and_dependencies_are_vendored(self) -> None:
        self.assertTrue(SHIM.is_file(), f"missing shim: {SHIM}")
        self.assertTrue(
            (SHIM_DIR / "libgfortran.so.5").is_file(),
            "missing vendored libgfortran.so.5 (required by the shim)",
        )

    def test_pipeline_scripts_reference_this_shim(self) -> None:
        for script in SCRIPTS:
            text = script.read_text(encoding="utf-8")
            self.assertIn(SHIM.name, text, script.name)
            self.assertIn("LD_PRELOAD", text, script.name)
            self.assertIn("OPENBLAS_NUM_THREADS=1", text, script.name)

    def test_flavor_is_pthread_with_adequate_thread_limit(self) -> None:
        lib = load_shim()
        lib.openblas_get_config.restype = ctypes.c_char_p
        config = lib.openblas_get_config().decode()

        # 0=serial and 2=OpenMP builds both crashed real reconstructions;
        # only the pthread flavor is safe for arbitrary concurrent callers.
        self.assertEqual(
            lib.openblas_get_parallel(),
            1,
            f"shim must be the pthread flavor, got config: {config}",
        )

        limit = re.search(r"MAX_THREADS=(\d+)", config)
        self.assertIsNotNone(limit, f"no MAX_THREADS in config: {config}")
        self.assertGreaterEqual(
            int(limit.group(1)),
            64,
            "compiled thread limit is below the 64 concurrent callers the "
            f"pipeline allows: {config}",
        )

    def test_interposable_blas_symbols_are_exported(self) -> None:
        lib = load_shim()
        # Unsuffixed names are what make LD_PRELOAD interposition work
        # against pycolmap's statically linked BLAS.
        for symbol in ("sgemm_", "cblas_sgemm", "openblas_set_num_threads"):
            self.assertTrue(hasattr(lib, symbol), symbol)

    def test_concurrent_sgemm_callers_compute_correctly(self) -> None:
        """Hammer the shim from many threads at once.

        This mirrors how faiss calls BLAS from inside OpenMP worker teams.
        A serial or OpenMP-flavor build corrupts its buffer bookkeeping under
        this pattern ("BLAS : Bad memory unallocation!" then SIGSEGV); the
        pthread build must survive with correct results.
        """
        lib = load_shim()
        lib.openblas_set_num_threads(1)  # match the pipeline's env

        size = 64
        identity = (ctypes.c_float * (size * size))()
        for index in range(size):
            identity[index * size + index] = 1.0
        operand = (ctypes.c_float * (size * size))(
            *[float(index % 7) for index in range(size * size)]
        )

        cblas_row_major = ctypes.c_int(101)
        no_transpose = ctypes.c_int(111)
        lib.cblas_sgemm.restype = None
        errors: list[str] = []

        def hammer() -> None:
            result = (ctypes.c_float * (size * size))()
            for _ in range(150):
                lib.cblas_sgemm(
                    cblas_row_major,
                    no_transpose,
                    no_transpose,
                    ctypes.c_int(size),
                    ctypes.c_int(size),
                    ctypes.c_int(size),
                    ctypes.c_float(1.0),
                    identity,
                    ctypes.c_int(size),
                    operand,
                    ctypes.c_int(size),
                    ctypes.c_float(0.0),
                    result,
                    ctypes.c_int(size),
                )
            mismatches = sum(
                1
                for index in range(size * size)
                if result[index] != operand[index]
            )
            if mismatches:
                errors.append(f"{mismatches} wrong elements in I @ B")

        workers = [threading.Thread(target=hammer) for _ in range(48)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
