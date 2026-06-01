"""``python -m mesofield.datakit`` — open an interactive datakit shell.

Thin wrapper around :func:`mesofield.datakit.shell.open_shell`; the same
behaviour is exposed as ``mesofield datakit shell``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .shell import open_shell


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m mesofield.datakit")
    parser.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=None,
        help="Experiment directory, or a materialized .pkl/.h5 dataset file. "
        "Omit to open a bare datakit shell.",
    )
    parser.add_argument(
        "--hdf-key",
        default="dataset",
        help="HDF5 key to read when target is an .h5/.hdf5 file (default: dataset).",
    )
    args = parser.parse_args()
    return open_shell(args.target, hdf_key=args.hdf_key)


if __name__ == "__main__":
    raise SystemExit(main())
