"""Phone camera poses (c2w, OpenCV axes) in the frame of the gsplat PLY: gsplat's own Parser with normalize=True, as
simple_trainer.py builds it (the other Parser arguments do not change the frame).

From the repo root:
  third-party/gsplat/.venv/bin/python scripts/export_cameras.py work/<project>/<video> \
      outputs/<project>/<video>/cameras.npz
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{REPO}/third-party/gsplat/examples")
from datasets.colmap import Parser  # noqa: E402

data_dir, out = sys.argv[1:3]
parser = Parser(data_dir, normalize=True)
np.savez(out, c2w=parser.camtoworlds)
print(f"wrote {len(parser.camtoworlds)} camera poses to {out}")
