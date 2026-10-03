"""Default filesystem locations for the pi0.5 LoRA workflow.

Each root can be overridden with an environment variable so the scripts work
outside the original training machine:

- PARC_DATA_ROOT:      datasets (LIBERO-plus source, openpi-format output)
- PARC_TRAINING_ROOT:  training checkpoints
- OPENPI_ROOT:         openpi checkout (with its own .venv)
- OPENPI_DATA_HOME:    openpi asset cache (base pi0.5-LIBERO checkpoint)
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("PARC_DATA_ROOT", "/work/PARC2026_data"))
TRAINING_ROOT = Path(os.environ.get("PARC_TRAINING_ROOT", "/work/PARC2026_training"))
OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/tmp/openpi"))
OPENPI_DATA_HOME = Path(os.environ.get("OPENPI_DATA_HOME", "/tmp/openpi-data"))

LEROBOT_HOME = DATA_ROOT / "lerobot"
LIBERO_PLUS_SOURCE = LEROBOT_HOME / "lerobot" / "libero_plus"
CHECKPOINT_ROOT = TRAINING_ROOT / "checkpoints"
BASE_PI05_LIBERO = OPENPI_DATA_HOME / "openpi-assets" / "checkpoints" / "pi05_libero"
