#!/usr/bin/env python3
"""
CLI entry point for the KKBox feature pipeline.

Reads raw CSVs from data/raw/, runs all transformations in memory, and writes
a single model-ready CSV to data/processed/. No intermediate files on disk.

Usage:
  python scripts/build_processed_datasets.py                          # pipeline v1 (default)
  python scripts/build_processed_datasets.py --pipeline-version 2    # pipeline v2
  python scripts/build_processed_datasets.py --raw-dir /path/to/raw --out-dir /path/to/out

Pipeline versions:
  v1  →  src/pipeline_v1.py  →  data/processed/df_model_v1.csv  (26 features, notebook parity)
  v2  →  src/pipeline_v2.py  →  data/processed/df_model_v2.csv  (v1 + log windows + interaction flags)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import AVAILABLE_PIPELINE_VERSIONS, load_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-version",
        type=int,
        default=None,
        metavar="N",
        help=f"Pipeline implementation (default: src.pipeline.DEFAULT_PIPELINE_VERSION). "
        f"Available: {AVAILABLE_PIPELINE_VERSIONS}",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        pl = load_pipeline(args.pipeline_version)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    raw_dir = (args.raw_dir or pl.RAW_DATA_DIR).resolve()
    out_dir = (args.out_dir or pl.PROCESSED_DATA_DIR).resolve()

    ver = getattr(pl, "PIPELINE_VERSION", args.pipeline_version or "?")
    print(f"Using pipeline version {ver}")

    if hasattr(pl, "run_full_pipeline"):
        for f in (pl.FILE_TRAIN, pl.FILE_MEMBERS, pl.FILE_TRANSACTIONS, pl.FILE_USER_LOGS):
            p = raw_dir / f
            if not p.exists():
                print(f"ERROR: missing raw file: {p}", file=sys.stderr)
                sys.exit(1)
        try:
            pl.run_full_pipeline(raw_dir, out_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "ERROR: this pipeline version has no run_full_pipeline(); extend build_processed_datasets.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
