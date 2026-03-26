"""
Pipeline version registry.

- v1: src/pipeline_v1.py (notebook parity — frozen baseline for comparisons)
- v2: src/pipeline_v2.py (v1 + log windows, extra log metrics, interaction flags)

Switch default for day-to-day work by changing DEFAULT_PIPELINE_VERSION.
Compare runs side by side with different --pipeline-version and --out-dir, e.g.:

  python scripts/build_processed_datasets.py --pipeline-version 1 --out-dir data/processed/pipeline_v1
  python scripts/build_processed_datasets.py --pipeline-version 2 --out-dir data/processed/pipeline_v2
"""

from __future__ import annotations

import importlib
from types import ModuleType

DEFAULT_PIPELINE_VERSION = 1

AVAILABLE_PIPELINE_VERSIONS: tuple[int, ...] = (1, 2)


def load_pipeline(version: int | None = None) -> ModuleType:
    """
    Return the pipeline implementation module for the given version.

    The module must define: FILE_*, RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TARGET,
    FEATURES_V3, and a runner such as run_full_pipeline(raw_dir, out_dir) or (legacy) staged runners.
    """
    v = DEFAULT_PIPELINE_VERSION if version is None else version
    if v == 1:
        return importlib.import_module("src.pipeline_v1")
    if v == 2:
        return importlib.import_module("src.pipeline_v2")
    raise ValueError(
        f"Unknown pipeline version {v}. Available: {AVAILABLE_PIPELINE_VERSIONS}. "
        "Implement src/pipeline_v2.py and register it in src/pipeline.py."
    )
