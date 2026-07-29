from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RunbookSafetyTests(unittest.TestCase):
    def test_cuda_backend_switch_clears_existing_environment(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "uv venv --clear --python 3.11 third-party/gsplat/.venv",
            readme,
        )

    def test_viewer_is_constrained_to_one_gpu(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "CUDA_VISIBLE_DEVICES=0 uv run --project third-party/gsplat --no-sync",
            readme,
        )

    def test_setup_uses_standalone_submodule_path(self) -> None:
        setup = (PROJECT_ROOT / "scripts" / "setup_gsplat.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"', setup)
        self.assertNotIn("third-party/mapping/gsplat", setup)


if __name__ == "__main__":
    unittest.main()
