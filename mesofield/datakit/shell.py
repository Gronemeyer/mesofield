"""Interactive shell helper for datakit.

Opens an embedded IPython (falling back to the stdlib ``code`` REPL) with a
datakit object pre-loaded into the namespace. Shared by the
``mesofield datakit shell`` CLI command and ``python -m mesofield.datakit``.

Depending on the ``target`` the shell is seeded with:

- a directory  -> ``dataset`` (a :class:`~mesofield.datakit.Dataset`) plus
  its ``inventory`` and a ``report`` from :func:`~mesofield.datakit.explore`
- a ``.pkl`` / ``.h5`` file -> ``df`` (the materialized DataFrame loaded via
  :func:`~mesofield.datakit.load_dataset`) plus its ``report``
- nothing -> just the datakit package bound as ``datakit``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]


def build_namespace(
    target: Optional[PathLike] = None,
    *,
    hdf_key: str = "dataset",
) -> tuple[dict[str, Any], str]:
    """Build the ``(namespace, header)`` pair for an interactive datakit shell."""
    import mesofield.datakit as datakit
    from .core import Dataset, load_dataset
    from .explore import explore

    namespace: dict[str, Any] = {"datakit": datakit, "explore": explore}

    if target is None:
        header = "Embedded datakit shell. Available: datakit, explore"
        return namespace, header

    p = Path(target).expanduser().resolve()
    if p.is_dir():
        dataset = Dataset.from_directory(p)
        report = explore(dataset, print_output=False)
        namespace.update(dataset=dataset, inventory=dataset.inventory, report=report)
        header = (
            f"Embedded datakit shell for {p}\n"
            "Available: dataset, inventory, report, datakit, explore"
        )
    elif p.is_file():
        df = load_dataset(p, hdf_key=hdf_key)
        report = explore(df, print_output=False)
        namespace.update(df=df, report=report)
        header = (
            f"Embedded datakit shell for {p}\n"
            f"Loaded materialized dataset as 'df' "
            f"({df.shape[0]} rows x {df.shape[1]} cols)\n"
            "Available: df, report, datakit, explore"
        )
    else:
        raise FileNotFoundError(f"Path does not exist: {p}")

    return namespace, header


def open_shell(
    target: Optional[PathLike] = None,
    *,
    hdf_key: str = "dataset",
) -> int:
    """Open an interactive shell pre-loaded with a datakit object.

    Returns a process exit code (``0`` on success).
    """
    namespace, header = build_namespace(target, hdf_key=hdf_key)
    try:
        from IPython import embed

        embed(header=header, user_ns=namespace)
    except ImportError:
        from code import interact

        interact(header, local=namespace)
    return 0
