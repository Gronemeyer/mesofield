"""Tiny CLI: open an IPython shell with a Dataset for an experiment root."""

from __future__ import annotations

import argparse
from pathlib import Path

from mesofield.datakit.core import Dataset


def main() -> int:
    parser = argparse.ArgumentParser(prog="datakit")
    parser.add_argument("root", type=Path, help="Experiment directory")
    args = parser.parse_args()

    dataset = Dataset.from_directory(args.root)
    namespace = {"dataset": dataset, "inventory": dataset.inventory}
    header = "Embedded datakit shell. Available: dataset, inventory"
    try:
        from IPython import embed
        embed(header=header, user_ns=namespace)
    except ImportError:
        from code import interact
        interact(header, local=namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
