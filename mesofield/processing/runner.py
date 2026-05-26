"""Base class for intermediate processing stages.

A processor takes raw acquisition files and produces derived files. The
runner handles the boilerplate: hashing inputs, locating the upstream
AcquisitionManifest, writing a ProcessingManifest sidecar alongside the
outputs, and turning declared outputs into ProducerEntry shapes the
ingest layer can consume.

Typical use:

    from mesofield.processing import ProcessorRunner

    class SpikeSorter(ProcessorRunner):
        tool_name = "my_lab_spikesort"
        tool_version = "0.1.0"

        def run(self, inputs, *, sigma=4.0):
            in_path = inputs[0]
            out = in_path.parent / "spikes.csv"
            # ... do the work; write to `out` ...
            return [out]

    runner = SpikeSorter()
    runner([recording_path], sigma=5.0)
    # → spikes.csv written, plus my_lab_spikesort.process.json next to it
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional, Sequence

from mesokit_schema import (
    AcquisitionManifest,
    InputRef,
    ProcessingManifest,
    ProducerEntry,
    TimeBasis,
)
from mesokit_schema.dataset import hash_file


class ProcessorRunner:
    """Wrap a file-to-file transformation in a ProcessingManifest contract.

    Subclasses must set :attr:`tool_name` and :attr:`tool_version`, and
    override :meth:`run`. Calling the instance executes the work and
    writes the sidecar.
    """

    tool_name: ClassVar[str] = ""
    tool_version: ClassVar[str] = "0.0.0"

    #: Where the manifest sidecar lands. ``"output_dir"`` writes
    #: ``<tool_name>.process.json`` in the directory of the first output;
    #: subclasses can override :meth:`manifest_path` for custom placement.
    manifest_placement: ClassVar[str] = "output_dir"

    # ------------------------------------------------------------------ user hooks

    def run(self, inputs: Sequence[Path], **params: Any) -> list[Path]:
        """Do the actual work. Return the list of files written."""
        raise NotImplementedError("ProcessorRunner subclasses must implement run()")

    def declare_outputs(
        self,
        outputs: Sequence[Path],
        params: dict[str, Any],
        session_root: Optional[Path],
    ) -> list[ProducerEntry]:
        """Turn run() outputs into ProducerEntry instances.

        Default: one entry per output, with data_type=tool_name and
        bids_type/file_type inferred from the path. Override for richer
        declarations (e.g. multiple roles, sidecars).
        """
        entries: list[ProducerEntry] = []
        for path in outputs:
            rel = self._relative_to_session(path, session_root)
            entries.append(
                ProducerEntry(
                    device_id=self.tool_name,
                    device_type="processor",
                    data_type=self.tool_name,
                    bids_type=self._infer_bids_type(path, session_root),
                    file_type=self._infer_file_type(path),
                    output_path=rel,
                    time_basis=TimeBasis(
                        clock_source="derived",
                        description=f"Derived by {self.tool_name} v{self.tool_version}",
                    ),
                )
            )
        return entries

    # ------------------------------------------------------------------ orchestration

    def __call__(
        self,
        inputs: Sequence[Path],
        *,
        upstream: Optional[AcquisitionManifest | Path] = None,
        session_root: Optional[Path] = None,
        **params: Any,
    ) -> tuple[list[Path], ProcessingManifest]:
        """Run + emit the sidecar. Returns (outputs, manifest)."""
        if not self.tool_name:
            raise ValueError(f"{type(self).__name__} must set tool_name")

        input_paths = [Path(p) for p in inputs]
        input_refs = [
            InputRef(path=str(p), content_hash=hash_file(p)) for p in input_paths
        ]

        upstream_manifest, upstream_hash, resolved_session_root = self._resolve_upstream(
            upstream, session_root, input_paths
        )
        if session_root is None:
            session_root = resolved_session_root

        outputs = [Path(p) for p in self.run(input_paths, **params)]
        if not outputs:
            raise RuntimeError(
                f"{self.tool_name}.run() returned no outputs; nothing to declare"
            )

        producer_entries = self.declare_outputs(outputs, params, session_root)

        manifest = ProcessingManifest(
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            tool_invocation=self._invocation(input_paths, params),
            built_at=datetime.now(timezone.utc),
            upstream_acquisition_hash=upstream_hash,
            inputs=input_refs,
            parameters=self._jsonable_params(params),
            outputs=producer_entries,
        )
        sidecar = self.manifest_path(outputs)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        manifest.write(sidecar)
        return outputs, manifest

    # ------------------------------------------------------------------ helpers

    def manifest_path(self, outputs: Sequence[Path]) -> Path:
        if self.manifest_placement == "output_dir":
            return Path(outputs[0]).parent / f"{self.tool_name}.process.json"
        raise ValueError(f"Unknown manifest_placement: {self.manifest_placement!r}")

    def _invocation(self, inputs: Sequence[Path], params: dict[str, Any]) -> str:
        sig = inspect.signature(self.run)
        bound = ", ".join(
            [f"inputs={[p.name for p in inputs]}"]
            + [f"{k}={v!r}" for k, v in params.items()]
        )
        return f"{type(self).__name__}().run({bound})"

    def _jsonable_params(self, params: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in params.items():
            try:
                import json
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = repr(v)
        return out

    @staticmethod
    def _relative_to_session(path: Path, session_root: Optional[Path]) -> str:
        if session_root is None:
            return str(path)
        try:
            return str(Path(path).resolve().relative_to(Path(session_root).resolve()))
        except ValueError:
            return str(path)

    @staticmethod
    def _infer_bids_type(path: Path, session_root: Optional[Path]) -> Optional[str]:
        if session_root is None:
            return path.parent.name or None
        try:
            rel = Path(path).resolve().relative_to(Path(session_root).resolve())
        except ValueError:
            return path.parent.name or None
        # rel is e.g. processed/<bids_type>/<file> or processed/<file>
        parts = rel.parts
        if len(parts) >= 3 and parts[0] == "processed":
            return parts[1]
        if len(parts) == 2:
            return parts[0] if parts[0] != "processed" else None
        return None

    @staticmethod
    def _infer_file_type(path: Path) -> str:
        # Preserve multi-dot extensions like ome.tiff.
        name = path.name
        if "." not in name:
            return ""
        first_dot = name.index(".")
        return name[first_dot + 1:]

    @staticmethod
    def _resolve_upstream(
        upstream: Optional[AcquisitionManifest | Path],
        session_root: Optional[Path],
        input_paths: Sequence[Path],
    ) -> tuple[Optional[AcquisitionManifest], Optional[str], Optional[Path]]:
        """Locate the AcquisitionManifest for the provenance chain.

        Accepts an already-loaded manifest, an explicit path, or `None`
        (in which case we walk up from the first input looking for
        `manifest.json`). Returns (manifest, content_hash, session_root).
        """
        if isinstance(upstream, AcquisitionManifest):
            return upstream, upstream.content_hash(), session_root

        candidate: Optional[Path] = None
        if isinstance(upstream, (str, Path)):
            candidate = Path(upstream)
        elif input_paths:
            for parent in [input_paths[0].resolve().parent, *input_paths[0].resolve().parents]:
                if (parent / "manifest.json").exists():
                    candidate = parent / "manifest.json"
                    break

        if candidate is None or not candidate.exists():
            return None, None, session_root

        try:
            manifest = AcquisitionManifest.read(candidate)
        except Exception:
            return None, None, session_root
        return manifest, manifest.content_hash(), candidate.parent
