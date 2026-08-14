"""Pure (Qt-free) helpers for editing/validating the MousePortal config block.

Kept separate from the Qt widget so the assemble/validate logic can be unit
tested headlessly. The widget (``mouseportal_controller.py``) is a thin binding
over these functions; persistence goes through
``ExperimentConfig.update_mouseportal``.

The block shape mirrors MousePortal's own config (see ``mouseportal.config``):
``{window, camera, fog, assets, experiment}`` where ``experiment`` is
``{iti_duration, iti_range, random_seed, conditions[], blocks[]}``.

Two things live in exactly one place each:

- **A condition owns how its trials end.** ``trial_end_condition`` plus its
  limit are per-condition and required. There is no session-wide default to
  override, so a trial's behaviour is readable from its condition alone.
- **A block owns its own trial count.** ``len(sequence) * repeat``. There is no
  global trials-per-block, so blocks may differ in length and nothing can
  contradict them.

Display strings produced here live in :data:`TEXT` at the top of the file.
The widget's own labels live in ``mouseportal_controller.TEXT``; between the
two, no user-visible wording is buried in code.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any, Dict, List, Tuple

# ════════════════════════════════════════════════════════════════════════════
# User-facing text produced by this module. Edit here.
# ════════════════════════════════════════════════════════════════════════════
TEXT = {
    # Condition list rows, e.g. "gain_2x (gain 2, 30 s)"
    "condition_row": "{label} ({detail})",
    "condition_unnamed": "(unnamed)",
    "ends_duration": "{value} s",
    "ends_distance": "{value} units",
    "ends_manual": "manual",

    # Block list rows, e.g. "order_A (12 trials, shuffled)"
    "block_row": "{name} ({trials})",
    "block_trials": "{n} trial",
    "block_trials_plural": "{n} trials",
    "block_shuffled": ", shuffled",
    "block_fallback_name": "block {n}",

    # Expanded plan preview
    "plan_empty": "No blocks defined.",
    "plan_line": "{n}. {summary}: {trials}",
    "plan_more": " ...",
    "plan_none": "-",
    "plan_shuffled": "  (order drawn at run time)",
    "plan_total": "= {blocks} block(s), {trials} trials",

    # Status-panel summary
    "summary_empty": "No experiment configured.",
    "summary": "{blocks} block(s), {trials} trials, {length}",
    "summary_length": "~{seconds:.0f} s",
    "summary_length_partial": "at least {seconds:.0f} s ({unknown} unestimated)",
    "summary_conditions": "conditions: {labels}",
    "summary_no_conditions": "-",
}

# How many trials of a block's expanded sequence the plan preview lists before
# it elides the rest.
PLAN_PREVIEW_TRIALS = 8

# Velocity transforms MousePortal knows (mouseportal.transforms._REGISTRY),
# mapped to their parameters and MousePortal's own defaults. The editor builds
# a labelled field per parameter from this table, so the parameter names are on
# screen rather than something the user has to remember and type.
TRANSFORM_PARAMS: Dict[str, Dict[str, float]] = {
    "identity": {},
    "gain": {"gain": 1.0},
    "invert": {},
    "reverse": {"speed": 20.0},
    "freeze": {},
    "offset": {"offset": 5.0},
    "clamp": {"lo": 0.0, "hi": 10.0},
    "noisy": {"sigma": 3.0},
    "delay": {"delay_sec": 0.2},
}
KNOWN_TRANSFORMS = tuple(TRANSFORM_PARAMS)
TRIAL_END_CONDITIONS = ("duration", "distance", "manual")
BLOCK_ORDERS = ("fixed", "shuffle")

# Transforms that ignore the subject's input entirely, so a distance-based end
# rule either cannot be reached or is reached on motion the subject did not
# produce. MousePortal rejects the combination at load; the editor rejects it
# at Save so it never gets that far.
OPEN_LOOP_TRANSFORMS = frozenset({"freeze", "reverse"})

# End rules whose trial length cannot be known ahead of time, so an explicit
# ``expected_duration`` is what makes a run-length estimate possible.
UNTIMED_END_CONDITIONS = frozenset({"distance", "manual"})


def default_seed() -> int:
    """Today's date as ``YYYYMMDD`` — mirrors ``mouseportal.config.default_seed``.

    Note the consequence: two sessions run the same day on this seed draw the
    same ITI lengths and the same shuffled block orders. Change it per subject
    when independent randomisation matters.
    """
    return int(date.today().strftime("%Y%m%d"))


# ─── Condition helpers ──────────────────────────────────────────────────────

def new_condition(label: str = "") -> Dict[str, Any]:
    """A condition mapping with every required field already answered."""
    return {
        "label": label,
        "transform_type": "identity",
        "trial_end_condition": "duration",
        "trial_duration": 30.0,
    }


def condition_summary(cond: Dict[str, Any]) -> str:
    """One-line description of a condition, for the selector list."""
    label = cond.get("label") or TEXT["condition_unnamed"]
    detail = cond.get("transform_type", "identity")
    params = cond.get("transform_params") or {}
    if params:
        detail += " " + " ".join(f"{v:g}" for v in params.values())

    end = cond.get("trial_end_condition")
    if end == "duration":
        detail += ", " + TEXT["ends_duration"].format(
            value=_num(cond.get("trial_duration"))
        )
    elif end == "distance":
        detail += ", " + TEXT["ends_distance"].format(
            value=_num(cond.get("trial_distance"))
        )
    else:
        detail += ", " + TEXT["ends_manual"]
    return TEXT["condition_row"].format(label=label, detail=detail)


def planning_duration(cond: Dict[str, Any]) -> float | None:
    """Seconds a trial of this condition is expected to take, or None.

    A ``duration`` trial is its own estimate. Anything else needs an explicit
    ``expected_duration``; without one the length is genuinely unknown and
    saying so beats inventing a number.
    """
    if cond.get("trial_end_condition") == "duration":
        return _opt_float(cond.get("trial_duration"))
    return _opt_float(cond.get("expected_duration"))


def _opt_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> str:
    v = _opt_float(value)
    return f"{v:g}" if v is not None else "?"


# ─── Block helpers ──────────────────────────────────────────────────────────

def new_block(name: str = "") -> Dict[str, Any]:
    return {"name": name, "sequence": [], "repeat": 1, "order": "fixed"}


def block_trials(block: Dict[str, Any]) -> List[str]:
    """The block's expanded trial order, before any shuffle.

    Shuffling is drawn from the session seed inside MousePortal, so the order
    here is only the pre-shuffle one. That is enough for every number this
    module reports: a shuffle permutes the sequence without changing which
    conditions it contains, so trial counts and total duration are unaffected.
    """
    sequence = block.get("sequence") or []
    if not isinstance(sequence, list):
        return []
    repeat = block.get("repeat", 1)
    repeat = repeat if isinstance(repeat, int) and repeat >= 1 else 1
    return [str(s) for s in sequence] * repeat


def block_summary(block: Dict[str, Any], index: int) -> str:
    """One-line description of a block, for the selector list."""
    name = block.get("name") or TEXT["block_fallback_name"].format(n=index + 1)
    n = len(block_trials(block))
    key = "block_trials" if n == 1 else "block_trials_plural"
    trials = TEXT[key].format(n=n)
    if block.get("order") == "shuffle":
        trials += TEXT["block_shuffled"]
    return TEXT["block_row"].format(name=name, trials=trials)


def describe_plan(experiment: Dict[str, Any]) -> str:
    """Multi-line preview of the expanded session, one line per block."""
    blocks = (experiment or {}).get("blocks") or []
    if not blocks:
        return TEXT["plan_empty"]
    lines: List[str] = []
    for i, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            continue
        trials = block_trials(blk)
        shown = ", ".join(trials[:PLAN_PREVIEW_TRIALS])
        if len(trials) > PLAN_PREVIEW_TRIALS:
            shown += TEXT["plan_more"]
        line = TEXT["plan_line"].format(
            n=i + 1, summary=block_summary(blk, i), trials=shown or TEXT["plan_none"]
        )
        if blk.get("order") == "shuffle":
            line += TEXT["plan_shuffled"]
        lines.append(line)
    total = sum(len(block_trials(b)) for b in blocks if isinstance(b, dict))
    lines.append(TEXT["plan_total"].format(blocks=len(blocks), trials=total))
    return "\n".join(lines)


# ─── Summaries ──────────────────────────────────────────────────────────────

def summarize_experiment(block: Dict[str, Any]) -> str:
    """One/two-line human summary of the experiment for status panels."""
    exp = (block or {}).get("experiment", {}) or {}
    if not exp:
        return TEXT["summary_empty"]
    blocks = exp.get("blocks") or []
    total_trials = sum(len(block_trials(b)) for b in blocks if isinstance(b, dict))
    # The seed-accurate length, not the average: with a random pause the two
    # differ by minutes, and this is the number the recording is sized from.
    seconds, unknown, _ = resolve_session(exp)
    length = (
        TEXT["summary_length_partial"].format(seconds=seconds, unknown=unknown)
        if unknown else TEXT["summary_length"].format(seconds=seconds)
    )
    head = TEXT["summary"].format(
        blocks=len(blocks), trials=total_trials, length=length
    )
    labels = ", ".join(str(c.get("label", "?")) for c in exp.get("conditions", []) or [])
    tail = TEXT["summary_conditions"].format(
        labels=labels or TEXT["summary_no_conditions"]
    )
    return f"{head}\n{tail}"


def total_duration(experiment: Dict[str, Any]) -> Tuple[float, int]:
    """Estimated session length in seconds, and how many trials it could not time.

    Sums each trial's expected duration plus the inter-trial interval that
    follows it. A randomised ITI counts as its mean, so the total is an
    expected length rather than an exact one. Conditions with ``iti_after``
    false contribute no interval.

    The second return value is the number of trials whose length is genuinely
    unknown -- a ``distance`` or ``manual`` condition with no
    ``expected_duration``. Those contribute nothing to the sum, so the total is
    a lower bound and the caller can say so instead of presenting a guess as a
    figure. Trials referencing an undefined condition count as unknown too.

    Shuffled blocks need no special handling: a permutation does not change
    which trials a block contains, so it does not change the sum.
    """
    exp = experiment or {}
    iti = _mean_iti(exp.get("iti_range"), _opt_float(exp.get("iti_duration")) or 0.0)
    conditions = {
        c.get("label"): c for c in exp.get("conditions", []) or [] if isinstance(c, dict)
    }

    total = 0.0
    unknown = 0
    for blk in exp.get("blocks", []) or []:
        if not isinstance(blk, dict):
            continue
        for label in block_trials(blk):
            cond = conditions.get(label)
            if cond is None:
                unknown += 1
                continue
            dur = planning_duration(cond)
            if dur is None:
                unknown += 1
            else:
                total += dur
            if cond.get("iti_after", True):
                total += _mean_iti(cond.get("iti_range"), iti)
    return total, unknown


def resolve_session(experiment: Dict[str, Any], seed: int | None = None):
    """Replay the session the way MousePortal will, for the given seed.

    Returns ``(seconds, unknown, itis)``: the total length, how many trials had
    no knowable length, and the inter-trial interval drawn for each trial in
    session order (``0.0`` where a condition suppresses the pause).

    This mirrors ``mouseportal.experiment.ExperimentStateMachine`` exactly:
    one master RNG seeded from ``random_seed`` yields two child seeds, the
    first for block shuffling and the second for the ITI draws, and the ITI
    stream is only consumed for trials whose condition sets ``iti_after``.
    Reproducing the derivation is what makes the estimate the *actual* session
    length rather than an average -- a 15-45 s random pause over 20 trials
    spans a 10-minute range, so the mean is not a number you can size a
    recording from.

    Any drift from MousePortal's implementation shows up as a wrong estimate,
    not as a wrong session: MousePortal draws its own values at run time and
    records them in the timing sidecar.
    """
    exp = experiment or {}
    if seed is None:
        seed = exp.get("random_seed")
    if seed is None:
        seed = default_seed()

    master = random.Random(int(seed))
    plan_rng = random.Random(master.randrange(2 ** 31))
    iti_rng = random.Random(master.randrange(2 ** 31))

    session_range = exp.get("iti_range")
    session_fixed = _opt_float(exp.get("iti_duration")) or 0.0
    conditions = {
        c.get("label"): c for c in exp.get("conditions", []) or [] if isinstance(c, dict)
    }

    total = 0.0
    unknown = 0
    itis: List[float] = []
    for blk in exp.get("blocks", []) or []:
        if not isinstance(blk, dict):
            continue
        trials = block_trials(blk)
        if blk.get("order") == "shuffle":
            plan_rng.shuffle(trials)
        for label in trials:
            cond = conditions.get(label)
            if cond is None:
                unknown += 1
                itis.append(0.0)
                continue
            dur = planning_duration(cond)
            if dur is None:
                unknown += 1
            else:
                total += dur
            if not cond.get("iti_after", True):
                itis.append(0.0)
                continue
            rng_range = cond.get("iti_range") or session_range
            if rng_range is None:
                pause = session_fixed
            else:
                pause = iti_rng.uniform(float(rng_range[0]), float(rng_range[1]))
            itis.append(pause)
            total += pause
    return total, unknown, itis


def _mean_iti(iti_range: Any, fallback: float) -> float:
    """Mean of an ``[min, max]`` ITI range, or ``fallback`` when unset."""
    if not iti_range:
        return fallback
    return (float(iti_range[0]) + float(iti_range[1])) / 2.0


# ─── Validation ─────────────────────────────────────────────────────────────

def _iti_range_errors(iti_range: Any, where: str) -> List[str]:
    """Validate an ``[min, max]`` ITI range; empty list when unset or valid."""
    if iti_range is None:
        return []
    try:
        lo, hi = (float(v) for v in iti_range)
    except (TypeError, ValueError):
        return [f"{where} must be [min, max] seconds."]
    return [f"{where} must satisfy 0 ≤ min ≤ max."] if lo < 0 or hi < lo else []


def _param_errors(label: str, ttype: str, params: Any) -> List[str]:
    """Check a transform's parameters against what that transform accepts."""
    if not isinstance(params, dict):
        return [f"condition '{label}': transform_params must be a mapping."]
    errors: List[str] = []
    expected = TRANSFORM_PARAMS[ttype]
    unknown = sorted(set(params) - set(expected))
    if unknown:
        errors.append(
            f"condition '{label}': transform '{ttype}' does not take {unknown}. "
            f"Accepts: {sorted(expected) or 'no parameters'}."
        )
    for key, value in params.items():
        if key in expected and not isinstance(value, (int, float)):
            errors.append(f"condition '{label}': {key} must be a number.")
    if ttype == "gain" and isinstance(params.get("gain", 1.0), (int, float)):
        if float(params.get("gain", 1.0)) < 0:
            errors.append(f"condition '{label}': gain must be ≥ 0 (use 'invert' to reverse).")
    if ttype == "clamp":
        lo, hi = params.get("lo", 0.0), params.get("hi", 10.0)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
            errors.append(f"condition '{label}': clamp lo must be ≤ hi.")
    if ttype == "reverse" and isinstance(params.get("speed", 20.0), (int, float)):
        if float(params.get("speed", 20.0)) < 0:
            errors.append(f"condition '{label}': reverse speed must be ≥ 0.")
    return errors


def _end_rule_errors(label: str, ttype: str, cond: Dict[str, Any]) -> List[str]:
    """Check the end rule and that its limit is present and positive."""
    end = cond.get("trial_end_condition")
    if end not in TRIAL_END_CONDITIONS:
        return [
            f"condition '{label}': trial_end_condition must be one of "
            f"{TRIAL_END_CONDITIONS}, got {end!r}."
        ]
    errors: List[str] = []
    if end == "duration":
        value = _opt_float(cond.get("trial_duration"))
        if not value or value <= 0:
            errors.append(f"condition '{label}': 'duration' needs a positive duration.")
    elif end == "distance":
        value = _opt_float(cond.get("trial_distance"))
        if not value or value <= 0:
            errors.append(f"condition '{label}': 'distance' needs a positive distance.")
        # An open-loop transform discards the subject's input, so a distance
        # rule either cannot be reached (freeze) or is reached on motion the
        # subject did not produce (reverse).
        if ttype in OPEN_LOOP_TRANSFORMS:
            errors.append(
                f"condition '{label}': '{ttype}' ignores the subject's input, so a "
                f"'distance' end rule cannot be reached by their running. Use "
                f"'duration' or 'manual'."
            )
    expected = cond.get("expected_duration")
    if expected is not None:
        value = _opt_float(expected)
        if value is None or value <= 0:
            errors.append(f"condition '{label}': expected duration must be positive.")
    return errors


def _condition_errors(conditions: Any) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Validate the condition palette; return errors and the label → condition map."""
    errors: List[str] = []
    by_label: Dict[str, Dict[str, Any]] = {}
    for i, cond in enumerate(conditions or []):
        if not isinstance(cond, dict):
            errors.append(f"condition #{i + 1} must be a mapping.")
            continue
        label = cond.get("label")
        if not label:
            errors.append(f"condition #{i + 1} is missing a label.")
            continue
        if label in by_label:
            errors.append(f"duplicate condition label '{label}'.")
            continue
        by_label[label] = cond

        ttype = cond.get("transform_type")
        if ttype not in TRANSFORM_PARAMS:
            errors.append(
                f"condition '{label}': unknown transform_type {ttype!r}. "
                f"Known: {', '.join(KNOWN_TRANSFORMS)}."
            )
        else:
            errors.extend(_param_errors(label, ttype, cond.get("transform_params") or {}))
            errors.extend(_end_rule_errors(label, ttype, cond))
        errors.extend(
            _iti_range_errors(cond.get("iti_range"), f"condition '{label}': iti_range")
        )
    return errors, by_label


def validate_block(block: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems (empty == valid).

    Mirrors MousePortal's own ``ExperimentConfig`` validation so a config this
    editor accepts is one MousePortal will load. Every check names the block or
    condition at fault; nothing is normalised or defaulted on the way past.
    """
    errors: List[str] = []
    exp = (block or {}).get("experiment")
    if not isinstance(exp, dict) or not exp:
        return ["experiment block is missing or empty."]

    if _opt_float(exp.get("iti_duration")) is None:
        errors.append("iti_duration must be a number.")
    elif float(exp["iti_duration"]) < 0:
        errors.append("iti_duration must be ≥ 0.")
    errors.extend(_iti_range_errors(exp.get("iti_range"), "iti_range"))

    seed = exp.get("random_seed")
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        errors.append("random_seed must be a non-negative integer (or absent for auto).")

    conditions = exp.get("conditions") or []
    if not conditions:
        errors.append("at least one condition is required.")
    cond_errors, by_label = _condition_errors(conditions)
    errors.extend(cond_errors)

    blocks = exp.get("blocks") or []
    if not blocks:
        errors.append("at least one block is required.")
    for i, blk in enumerate(blocks, start=1):
        where = f"block #{i}"
        if not isinstance(blk, dict):
            errors.append(f"{where} must be a mapping.")
            continue
        if blk.get("name"):
            where = f"block '{blk['name']}'"
        sequence = blk.get("sequence")
        if not sequence:
            errors.append(f"{where} has no trials.")
        elif not isinstance(sequence, list):
            errors.append(f"{where}: sequence must be a list of condition labels.")
        else:
            for lbl in sequence:
                if lbl not in by_label:
                    errors.append(f"{where} references undefined condition '{lbl}'.")
        repeat = blk.get("repeat", 1)
        if not isinstance(repeat, int) or repeat < 1:
            errors.append(f"{where}: repeat must be an integer ≥ 1.")
        order = blk.get("order", "fixed")
        if order not in BLOCK_ORDERS:
            errors.append(f"{where}: order must be one of {BLOCK_ORDERS}, got {order!r}.")

    names = [b.get("name") for b in blocks if isinstance(b, dict) and b.get("name")]
    if len(names) != len(set(names)):
        errors.append("block names must be unique (they identify blocks in the data).")
    return errors


def blocks_using(blocks: Any, label: str) -> List[str]:
    """Names (or indices) of blocks whose sequence references *label*."""
    used: List[str] = []
    for i, blk in enumerate(blocks or []):
        if isinstance(blk, dict) and label in (blk.get("sequence") or []):
            used.append(blk.get("name") or f"block {i + 1}")
    return used
