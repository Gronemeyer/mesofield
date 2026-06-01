"""``mesofield process`` — batch processing & conversion of recorded data.

Operates on BIDS-formatted experiment trees that have already been acquired:
fluorescence traces, frame montages, video conversion, and manifest
retrofitting.
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from ._richhelp import RichGroup


@click.group('process', cls=RichGroup)
def process():
    """Batch-process and convert recorded experiment data."""


# ---------------------------------------------------------------------------
# Datakit-based meso tiff discovery helpers
# ---------------------------------------------------------------------------


def _discover_meso_entries(experiment_dir, subject_filter=None):
    """Return ManifestEntry objects for mesoscope tiff files.

    Uses datakit's ``discover_manifest`` to scan the BIDS hierarchy,
    filters for ``meso_metadata`` entries (the JSON sidecars), then
    rewrites each entry's path to point at the tiff itself.
    """
    from mesofield.datakit.discover import discover_manifest
    from mesofield.datakit.datamodel import ManifestEntry

    manifest = discover_manifest(Path(experiment_dir))
    tiff_entries = []
    for entry in manifest.entries:
        if entry.tag != "meso_metadata":
            continue
        if subject_filter and entry.subject != subject_filter:
            continue
        # Derive the tiff path from the metadata sidecar path:
        #   ...mesoscope.ome.tiff_frame_metadata.json  →  ...mesoscope.ome.tiff
        tiff_path = entry.path.replace("_frame_metadata.json", "")
        tiff_entries.append(ManifestEntry(
            tag="meso_tiff",
            path=tiff_path,
            origin=entry.origin,
            subject=entry.subject,
            session=entry.session,
            task=entry.task,
        ))
    return tiff_entries


def _discover_meso_tiffs(experiment_dir, subject_filter=None):
    """Return a list of absolute tiff paths and the experiment root."""
    root = Path(experiment_dir).resolve()
    entries = _discover_meso_entries(experiment_dir, subject_filter)
    paths = [str(root / e.path) for e in entries]
    return paths, root


# ---------------------------------------------------------------------------
# trace-meso
# ---------------------------------------------------------------------------


@process.command('trace-meso')
@click.option('--path', help='Path to a single tiff file (bypass discovery)')
@click.option('--dir', help='Experiment directory to discover mesoscope tiffs')
@click.option('--sub', default=None, help='Subject ID to filter (default: all)')
def trace_meso(path, dir, sub):
    """Compute mean fluorescence traces from mesoscope OME-TIFF stacks.

    Uses datakit discovery to find mesoscope tiffs in a BIDS hierarchy,
    or accepts a single --path for quick one-off analysis.
    """
    import pandas as pd
    import mesofield.data.batch as batch

    if path:
        tiff_paths = [path]
        experiment_dir = Path(path).parents[4]
        outdir = Path(experiment_dir) / "processed"
    elif dir:
        tiff_paths, experiment_dir = _discover_meso_tiffs(dir, sub)
        outdir = Path(dir) / "processed" / sub if sub else Path(dir) / "processed"
    else:
        raise click.UsageError("Provide --path or --dir")

    if not tiff_paths:
        click.secho("No mesoscope tiffs found.", fg="yellow")
        return

    os.makedirs(outdir, exist_ok=True)
    click.echo(f"Computing mean traces for {len(tiff_paths)} tiff(s)...")
    results = batch.mean_trace_from_tiff(tiff_paths)

    for tiff_path, trace in results.items():
        df = pd.DataFrame({"Slice": range(len(trace)), "Mean": trace})
        base_name = os.path.splitext(os.path.basename(tiff_path))[0]
        filename = f"{base_name}_meso-mean-trace.csv"
        save_path = os.path.join(outdir, filename)
        df.to_csv(save_path, index=False)
        click.echo(f"  {filename}  ({len(trace)} frames)")

    click.secho(f"Saved {len(results)} trace(s) to {outdir}", fg="green")


# ---------------------------------------------------------------------------
# montage-meso
# ---------------------------------------------------------------------------


@process.command('montage-meso')
@click.option('--dir', required=True, help='Experiment directory containing BIDS formatted /data hierarchy')
@click.option('--sub', default=None, help='Single subject ID to process (default: all subjects)')
@click.option('--frame', default=1, show_default=True, help='0-based frame index to extract from each tiff')
@click.option('--rotate', default=0, show_default=True, type=int, help='Rotate each frame by N degrees (positive=clockwise, negative=counter-clockwise)')
@click.option('--filter', 'ses_filter', default=None, help='Session range START:END (1-based, exclusive end). E.g. 1:10 keeps sessions 1-9.')
def montage_meso(dir, sub, frame, rotate, ses_filter):
    """Extract a single frame from each session's widefield tiff and save a per-subject montage.

    Uses datakit to discover mesoscope tiffs in the BIDS hierarchy.
    Each task within a session becomes its own row.  Columns are sessions,
    so the output image is a grid of (tasks x sessions) panels.
    """
    import numpy as np
    import tifffile
    from PIL import Image, ImageDraw, ImageFont

    # --- Discover tiffs via datakit ---
    entries = _discover_meso_entries(dir, sub)
    if not entries:
        click.secho("No mesoscope tiffs found.", fg="yellow")
        return

    # Build a nested dict: subject -> session -> task -> tiff_path
    tree: dict[str, dict[str, dict[str, str]]] = {}
    root = Path(dir).resolve()
    for entry in entries:
        s, ses, task = entry.subject, entry.session, entry.task or "default"
        tree.setdefault(s, {}).setdefault(ses, {})[task] = str(root / entry.path)

    subjects = [sub] if sub else sorted(tree.keys())

    outdir = os.path.join(dir, "processed")
    os.makedirs(outdir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    label_height = 30

    for subject in subjects:
        if subject not in tree:
            click.echo(f"WARNING: sub-{subject} not found, skipping")
            continue
        sessions = tree[subject]
        ses_keys = sorted(sessions.keys())

        # Apply session range filter
        if ses_filter:
            parts = ses_filter.split(':')
            start = int(parts[0]) if parts[0] else 1
            end = int(parts[1]) if len(parts) > 1 and parts[1] else None
            ses_keys = [k for k in ses_keys if int(k) >= start and (end is None or int(k) < end)]

        # Collect the superset of task names across all sessions (preserving order)
        all_tasks: list[str] = []
        for sk in ses_keys:
            for tk in sessions[sk]:
                if tk not in all_tasks:
                    all_tasks.append(tk)

        # grid[task][ses_key] = normalised 8-bit numpy image
        grid: dict[str, dict[str, np.ndarray]] = {t: {} for t in all_tasks}

        for ses_key in ses_keys:
            session = sessions[ses_key]
            for task in all_tasks:
                if task not in session:
                    click.echo(f"WARNING: sub-{subject} ses-{ses_key} task-{task} missing, skipping")
                    continue

                tiff_path = session[task]
                try:
                    tiff_array = tifffile.memmap(tiff_path)
                    if tiff_array.shape[0] <= frame:
                        click.echo(f"WARNING: {tiff_path} has only {tiff_array.shape[0]} frames, skipping")
                        continue
                    img = np.array(tiff_array[frame])
                except Exception as e:
                    click.echo(f"ERROR reading {tiff_path}: {e}")
                    continue

                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_norm = np.zeros_like(img, dtype=np.uint8)

                if rotate:
                    img_norm = np.array(Image.fromarray(img_norm).rotate(-rotate, expand=True))

                grid[task][ses_key] = img_norm

        if not any(grid[t] for t in all_tasks):
            click.echo(f"WARNING: No frames found for sub-{subject}, skipping")
            continue

        all_imgs = [img for task_imgs in grid.values() for img in task_imgs.values()]
        cell_h = max(img.shape[0] for img in all_imgs)
        cell_w = max(img.shape[1] for img in all_imgs)

        def _resize(img):
            if img.shape[0] == cell_h and img.shape[1] == cell_w:
                return img
            return np.array(Image.fromarray(img).resize((cell_w, cell_h), Image.LANCZOS))

        def _blank():
            return np.zeros((cell_h, cell_w), dtype=np.uint8)

        def _label_header(text, width):
            header = Image.new('L', (width, label_height), color=0)
            draw = ImageDraw.Draw(header)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text(((width - text_w) // 2, 4), text, fill=255, font=font)
            return np.array(header)

        def _vertical_text(text, height, strip_w=40):
            tmp = Image.new('L', (height, strip_w), color=0)
            draw = ImageDraw.Draw(tmp)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((height - tw) // 2, (strip_w - th) // 2), text, fill=255, font=font)
            return tmp

        side_strip_w = 40
        grid_height = label_height + cell_h * len(all_tasks)

        subject_label = f"sub-{subject}"
        left_pil = _vertical_text(subject_label, grid_height, side_strip_w)
        left_strip = np.array(left_pil.rotate(90, expand=True))

        right_pieces = [np.zeros((label_height, side_strip_w), dtype=np.uint8)]
        for task in all_tasks:
            task_label = task if task else "default"
            tmp = _vertical_text(task_label, cell_h, side_strip_w)
            right_pieces.append(np.array(tmp.rotate(-90, expand=True)))
        right_strip = np.vstack(right_pieces)

        columns = []
        for ses_key in ses_keys:
            col_header = _label_header(f"ses-{ses_key}", cell_w)
            panels = [col_header]
            for task in all_tasks:
                img = grid[task].get(ses_key)
                panels.append(_resize(img) if img is not None else _blank())
            columns.append(np.vstack(panels))

        montage = np.hstack([left_strip] + columns + [right_strip])
        montage_img = Image.fromarray(montage)

        filename = f"sub-{subject}_frame-{frame}_montage.png"
        save_path = os.path.join(outdir, filename)
        montage_img.save(save_path)
        n_cells = sum(len(v) for v in grid.values())
        click.echo(f"Saved: {save_path}  ({len(all_tasks)} tasks x {len(ses_keys)} sessions, {n_cells} images)")

    click.secho("Done.", fg="green")


# ---------------------------------------------------------------------------
# pupil-mp4
# ---------------------------------------------------------------------------


@process.command('pupil-mp4')
@click.option('--dir', help='Directory containing the BIDS formatted /data hierarchy')
def pupil_mp4(dir):
    """Convert the pupil videos to mp4 format."""
    from mesofield.data.batch import tiff_to_mp4

    tiff_to_mp4(
        parent_directory=dir,
        fps=30,
        output_format="mp4",
        use_color=False
    )


# ---------------------------------------------------------------------------
# convert-h264
# ---------------------------------------------------------------------------


@process.command('convert-h264')
@click.option('--dir', required=True, help='Directory containing video files to convert')
@click.option('--pattern', default='*.mp4', help='Glob pattern to match files (e.g., "*.mp4", "pupil*.mp4")')
def convert_h264(dir, pattern):
    """Convert video files to H264 format for better compatibility."""
    from mesofield.data.batch import batch_convert_to_h264

    batch_convert_to_h264(
        parent_directory=dir,
        pattern=pattern
    )


# ---------------------------------------------------------------------------
# retrofit-manifest
# ---------------------------------------------------------------------------


@process.command('retrofit-manifest')
@click.argument('path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--force', is_flag=True,
              help='Overwrite existing manifest.json files (default: skip).')
@click.option('--dry-run', is_flag=True,
              help='Print what would be written without touching disk.')
def retrofit_manifest(path, force, dry_run):
    """Reconstruct mesokit-schema AcquisitionManifests for legacy sessions.

    PATH is either a single session directory (``data/sub-X/ses-Y/``) or an
    experiment root that contains many sessions under ``data/``. Hardware
    calibration is not present in legacy acquisitions; producers get an
    empty ``calibration`` dict. Frame-metadata sidecars (``*_frame_metadata.json``)
    are attached to their tiffs. Multi-task sessions write one manifest
    per task as ``manifest_task-<T>.json``.
    """
    from mesofield.utils.retrofit import (
        discover_sessions,
        manifest_filename,
        synthesize_manifests,
    )

    sessions = list(discover_sessions(Path(path)))
    if not sessions:
        click.secho(f"No BIDS sessions found under {path}", fg="yellow")
        raise SystemExit(1)

    written = skipped = empty = 0
    for session in sessions:
        by_task = synthesize_manifests(session)
        if not by_task:
            click.echo(f"empty {session}  (no producer files)")
            empty += 1
            continue
        multi_task = len(by_task) > 1
        for task, manifest in by_task.items():
            out_path = session / manifest_filename(task, multi_task)
            if out_path.exists() and not force:
                click.echo(f"skip  {out_path}  (exists; use --force to overwrite)")
                skipped += 1
                continue
            verb = "would write" if dry_run else "wrote"
            click.echo(
                f"{verb}  {out_path}  ({len(manifest.producers)} producers, task={task})"
            )
            if not dry_run:
                manifest.write(out_path)
                written += 1

    summary = (
        f"\n{'(dry-run) ' if dry_run else ''}"
        f"sessions={len(sessions)} written={written} skipped={skipped} empty={empty}"
    )
    click.echo(summary)
