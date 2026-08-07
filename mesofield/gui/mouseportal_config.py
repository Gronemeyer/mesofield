"""Pure (Qt-free) helpers for editing/validating the MousePortal config block.

Kept separate from the Qt widget so the assemble/validate logic can be unit
tested headlessly. The widget (``mouseportal_controller.py``) is a thin binding
over these functions; persistence goes through
``ExperimentConfig.update_mouseportal``.

The block shape mirrors MousePortal's own config (see
``mouseportal.config``): ``{window, fog, experiment}`` where ``experiment`` is
``{num_blocks, trials_per_block, iti_duration, trial_end_condition,
trial_duration, trial_distance, conditions[], block_conditions[]}``.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Velocity transforms MousePortal knows (mouseportal.transforms._REGISTRY).
KNOWN_TRANSFORMS = (
    "identity", "gain", "invert", "freeze", "offset", "clamp", "noisy", "delay",
)
# Single-parameter transforms editable via one value column in the GUI table.
# (``clamp`` takes two params; edit it in the raw block if needed.)
TRANSFORM_PARAM = {"gain": "gain", "offset": "offset", "noisy": "sigma", "delay": "delay_sec"}
ZERO_PARAM = {"identity", "invert", "freeze"}
TRIAL_END_CONDITIONS = ("duration", "distance", "manual")


def summarize_experiment(block: Dict[str, Any]) -> str:
    """One/two-line human summary of the experiment for status panels."""
    exp = (block or {}).get("experiment", {}) or {}
    if not exp:
        return "No experiment configured."
    nb = exp.get("num_blocks", 1)
    tpb = exp.get("trials_per_block", 0)
    end = exp.get("trial_end_condition", "duration")
    line = f"{nb} block(s) × {tpb} trials  ·  end: {end}"
    if end == "duration" and exp.get("trial_duration") is not None:
        line += f" {exp['trial_duration']}s"
    elif end == "distance" and exp.get("trial_distance") is not None:
        line += f" {exp['trial_distance']}"
    labels = ", ".join(str(c.get("label", "?")) for c in exp.get("conditions", []) or [])
    return f"{line}\nconditions: {labels or '—'}"


def parse_block_sequences(text: str) -> List[Dict[str, List[str]]]:
    """Parse one-block-per-line, comma-separated labels into block_conditions."""
    blocks: List[Dict[str, List[str]]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        labels = [tok.strip() for tok in line.split(",") if tok.strip()]
        blocks.append({"condition_sequence": labels})
    return blocks


def format_block_sequences(block_conditions: List[Dict[str, Any]]) -> str:
    """Render block_conditions back to the one-line-per-block text form."""
    lines = []
    for blk in block_conditions or []:
        seq = blk.get("condition_sequence", []) if isinstance(blk, dict) else []
        lines.append(", ".join(str(s) for s in seq))
    return "\n".join(lines)


def validate_block(block: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems (empty == valid).

    Mirrors MousePortal's ExperimentConfig validation plus cross-checks that
    block sequences reference defined conditions.
    """
    errors: List[str] = []
    exp = (block or {}).get("experiment")
    if not isinstance(exp, dict) or not exp:
        return ["experiment block is missing or empty."]

    nb = exp.get("num_blocks", 1)
    tpb = exp.get("trials_per_block", 1)
    iti = exp.get("iti_duration", 0.0)
    if not isinstance(nb, int) or nb < 1:
        errors.append("num_blocks must be an integer ≥ 1.")
    if not isinstance(tpb, int) or tpb < 1:
        errors.append("trials_per_block must be an integer ≥ 1.")
    try:
        if float(iti) < 0:
            errors.append("iti_duration must be ≥ 0.")
    except (TypeError, ValueError):
        errors.append("iti_duration must be a number.")

    end = exp.get("trial_end_condition", "duration")
    if end not in TRIAL_END_CONDITIONS:
        errors.append(f"trial_end_condition must be one of {TRIAL_END_CONDITIONS}.")

    conditions = exp.get("conditions", []) or []
    labels: List[str] = []
    for i, cond in enumerate(conditions):
        if not isinstance(cond, dict):
            errors.append(f"condition #{i + 1} must be a mapping.")
            continue
        label = cond.get("label")
        if not label:
            errors.append(f"condition #{i + 1} is missing a label.")
        else:
            labels.append(label)
        ttype = cond.get("transform_type", "identity")
        if ttype not in KNOWN_TRANSFORMS:
            errors.append(f"condition '{label}': unknown transform_type '{ttype}'.")
        params = cond.get("transform_params", {}) or {}
        if ttype == "gain":
            gain = params.get("gain", 1.0)
            try:
                if float(gain) < 0:
                    errors.append(f"condition '{label}': gain must be ≥ 0.")
            except (TypeError, ValueError):
                errors.append(f"condition '{label}': gain must be a number.")
    if len(labels) != len(set(labels)):
        errors.append("condition labels must be unique.")

    label_set = set(labels)
    block_conditions = exp.get("block_conditions", []) or []
    if not block_conditions:
        errors.append("at least one block sequence is required.")
    for bi, blk in enumerate(block_conditions):
        seq = blk.get("condition_sequence", []) if isinstance(blk, dict) else []
        if not seq:
            errors.append(f"block #{bi + 1} has an empty sequence.")
        for lbl in seq:
            if lbl not in label_set:
                errors.append(f"block #{bi + 1} references undefined condition '{lbl}'.")
    return errors
