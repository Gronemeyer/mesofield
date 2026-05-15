"""Intermediate processing stages (DLC, mesomap, lab pipelines).

Each processor is a small subclass of :class:`ProcessorRunner` that
defines ``run(inputs, **params) -> list[Path]``. Calling the runner
wraps that work in a hashing + manifest-writing harness so every
processed file lands with a ``<tool_name>.process.json`` sidecar
recording inputs, parameters, tool version, and upstream provenance.
"""

from mesofield.processing.runner import ProcessorRunner

__all__ = ["ProcessorRunner"]
