from __future__ import annotations

"""Human-readable description of the MousePortal messaging protocol.

The protocol is organised around the lifecycle of an experimental procedure:
trials are started or stopped, events are marked, stimuli can be updated, and
health/status telemetry flows in both directions normalising and validating 
JSON payloads exchanged between Mesofield and the Panda3D ``runportal`` process.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class Direction(str, Enum):
    HOST_TO_PORTAL = "host_to_portal"
    PORTAL_TO_HOST = "portal_to_host"
    BIDIRECTIONAL = "bidirectional"


class Category(str, Enum):
    TRIAL = "trial"
    EVENT = "event"
    STIMULUS = "stimulus"
    DIAGNOSTIC = "diagnostic"
    SESSION = "session"
    CONTROL = "control"
    TELEMETRY = "telemetry"
    LOGGING = "logging"
    ERROR = "error"
    PLAN = "plan"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    title: str
    description: str
    category: Category
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    one_of: tuple[tuple[str, ...], ...] = ()
    aliases: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MessageSpec:
    description: str
    category: Category
    direction: Direction
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    one_of: tuple[tuple[str, ...], ...] = ()

HOST_TO_PORTAL = Direction.HOST_TO_PORTAL.value
PORTAL_TO_HOST = Direction.PORTAL_TO_HOST.value
BIDIRECTIONAL = Direction.BIDIRECTIONAL.value

COMMANDS: Dict[str, CommandSpec] = {
    "start_trial": CommandSpec(
        title="Start trial",
        description="Begin executing a trial. Provide a mode such as open_loop or closed_loop.",
        category=Category.TRIAL,
        required=(),
        one_of=(("mode", "trial_type"),),
        optional=(
            "segments",
            "schedule",
            "loop",
            "repeat",
            "parameters",
            "event_name",
            "sent_time",
        ),
        aliases={
            "start": {},
            "start_open_loop": {"mode": "open_loop"},
            "start_closed_loop": {"mode": "closed_loop"},
        },
    ),
    "stop_trial": CommandSpec(
        title="Stop trial",
        description="End the active trial and return to idle.",
        category=Category.TRIAL,
        required=(),
        one_of=(),
        optional=("event_name", "sent_time"),
        aliases={"stop": {}},
    ),
    "mark_event": CommandSpec(
        title="Mark event",
        description="Record a behavioural event in the portal log.",
        category=Category.EVENT,
        required=(),
        one_of=(),
        optional=("name", "event", "sent_time", "label"),
        aliases={"event": {}, "add_event": {}},
    ),
    "set_texture": CommandSpec(
        title="Update corridor texture",
        description="Swap the texture applied to a corridor surface.",
        category=Category.STIMULUS,
        required=("face", "texture"),
        one_of=(),
        optional=("surface", "path"),
        aliases={},
    ),
    "get_status": CommandSpec(
        title="Request status",
        description="Ask the portal to emit an immediate status frame.",
        category=Category.DIAGNOSTIC,
        required=(),
        one_of=(),
        optional=("request_id",),
        aliases={"status": {}},
    ),
    "ping": CommandSpec(
        title="Ping",
        description="Measure round-trip time between host and portal.",
        category=Category.DIAGNOSTIC,
        required=(),
        one_of=(),
        optional=("sent_time", "request_id", "client_time"),
        aliases={},
    ),
    "shutdown": CommandSpec(
        title="Shutdown portal",
        description="Terminate the portal application and close sockets.",
        category=Category.SESSION,
        required=(),
        one_of=(),
        optional=(),
        aliases={"end": {}, "quit": {}, "exit": {}},
    ),
    "load_experiment_plan": CommandSpec(
        title="Load experiment plan",
        description="Provide a full experiment definition for later execution.",
        category=Category.PLAN,
        required=("plan",),
        one_of=(),
        optional=("plan_id", "default_mode", "auto_start", "auto_advance", "inter_trial_interval"),
        aliases={"load_plan": {}, "set_plan": {}},
    ),
    "run_experiment": CommandSpec(
        title="Run experiment",
        description="Start executing the previously loaded experiment plan.",
        category=Category.PLAN,
        required=(),
        one_of=(),
        optional=("restart", "overrides", "trial_overrides", "plan_id"),
        aliases={"run_plan": {}, "start_experiment": {}, "start_plan": {}},
    ),
    "pause_experiment": CommandSpec(
        title="Pause experiment",
        description="Pause the active experiment plan without discarding it.",
        category=Category.PLAN,
        required=(),
        one_of=(),
        optional=("reason", "plan_id"),
        aliases={"pause_plan": {}, "hold_experiment": {}},
    ),
    "resume_experiment": CommandSpec(
        title="Resume experiment",
        description="Resume a paused experiment plan from the next trial.",
        category=Category.PLAN,
        required=(),
        one_of=(),
        optional=("overrides", "trial_overrides", "plan_id"),
        aliases={"resume_plan": {}, "continue_experiment": {}},
    ),
    "abort_experiment": CommandSpec(
        title="Abort experiment",
        description="Stop the active experiment and optionally clear the loaded plan.",
        category=Category.PLAN,
        required=(),
        one_of=(),
        optional=("reason", "clear_plan", "plan_id"),
        aliases={"abort_plan": {}, "cancel_experiment": {}},
    ),
}

MESSAGES: Dict[str, MessageSpec] = {
    "command": MessageSpec(
        description="Command issued by the host controller.",
        category=Category.CONTROL,
        direction=Direction.HOST_TO_PORTAL,
        required=("command",),
        one_of=(),
        optional=("request_id", "client_time"),
    ),
    "status": MessageSpec(
        description="Ongoing corridor/trial telemetry from the portal.",
        category=Category.TELEMETRY,
        direction=Direction.PORTAL_TO_HOST,
        required=("time", "position", "velocity", "state", "trial_mode"),
        one_of=(),
        optional=(
            "input_velocity",
            "controller",
            "trial_elapsed",
            "request_id",
            "plan_state",
            "plan_id",
            "remaining_trials",
        ),
    ),
    "event": MessageSpec(
        description="Behavioural event emitted by the portal.",
        category=Category.EVENT,
        direction=Direction.PORTAL_TO_HOST,
        required=("name", "time_received", "position", "trial_mode"),
        one_of=(),
        optional=("delta", "controller", "time_sent"),
    ),
    "ack": MessageSpec(
        description="Acknowledgement of a processed command.",
        category=Category.CONTROL,
        direction=Direction.PORTAL_TO_HOST,
        required=("command", "status"),
        one_of=(),
        optional=("request_id", "sent_time", "message"),
    ),
    "client_status": MessageSpec(
        description="Socket status updates (connecting, connected, disconnected).",
        category=Category.TELEMETRY,
        direction=Direction.BIDIRECTIONAL,
        required=("status",),
        one_of=(),
        optional=("error", "time"),
    ),
    "client_error": MessageSpec(
        description="Protocol or transport error that did not close the socket.",
        category=Category.TELEMETRY,
        direction=Direction.BIDIRECTIONAL,
        required=("error",),
        one_of=(),
        optional=("raw", "time"),
    ),
    "stdout": MessageSpec(
        description="Forwarded stdout line from the portal process.",
        category=Category.LOGGING,
        direction=Direction.PORTAL_TO_HOST,
        required=("text",),
        one_of=(),
        optional=(),
    ),
    "process_exit": MessageSpec(
        description="Final exit code emitted when the portal terminates.",
        category=Category.LOGGING,
        direction=Direction.PORTAL_TO_HOST,
        required=("exit_code",),
        one_of=(),
        optional=("status",),
    ),
    "ping": MessageSpec(
        description="Ping message from the host.",
        category=Category.DIAGNOSTIC,
        direction=Direction.HOST_TO_PORTAL,
        required=(),
        one_of=(),
        optional=("client_time", "request_id"),
    ),
    "pong": MessageSpec(
        description="Response to a ping message.",
        category=Category.DIAGNOSTIC,
        direction=Direction.PORTAL_TO_HOST,
        required=(),
        one_of=(),
        optional=("client_time", "request_id"),
    ),
    "error": MessageSpec(
        description="Fatal protocol error reported by the portal.",
        category=Category.ERROR,
        direction=Direction.PORTAL_TO_HOST,
        required=("message",),
        one_of=(),
        optional=("raw",),
    ),
}


def _normalise_key(value: str) -> str:
    return value.strip().lower()


def _alias_defaults(spec: CommandSpec) -> Dict[str, Dict[str, Any]]:
    aliases = spec.aliases or {}
    return {
        _normalise_key(name): dict(defaults or {})
        for name, defaults in aliases.items()
    }


def resolve_command(command: str | None) -> Tuple[Optional[str], Optional[CommandSpec], Dict[str, Any]]:
    """Return the canonical command name, its spec, and alias default fields."""

    if not isinstance(command, str):
        return None, None, {}
    key = _normalise_key(command)
    spec = COMMANDS.get(key)
    if spec:
        return key, spec, {}
    for canonical, spec in COMMANDS.items():
        aliases = _alias_defaults(spec)
        if key in aliases:
            return canonical, spec, aliases[key]
    return None, None, {}


def lookup_command(command: str | None) -> Tuple[Optional[str], Optional[CommandSpec]]:
    canonical, spec, _ = resolve_command(command)
    return canonical, spec


def coerce_command(command: str) -> str:
    canonical, spec = lookup_command(command)
    if canonical is None or spec is None:
        raise ValueError(f"Unknown MousePortal command: {command}")
    return canonical


def build_command_message(
    command: str,
    *,
    request_id: Optional[str] = None,
    include_type: bool = True,
    payload: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Construct a command message, applying alias defaults when available."""

    canonical, spec, defaults = resolve_command(command)
    if canonical is None or spec is None:
        raise ValueError(f"Unknown MousePortal command: {command}")
    message: Dict[str, Any] = {"command": canonical}
    if include_type:
        message["type"] = "command"
    if request_id is not None:
        message["request_id"] = request_id
    merged: Dict[str, Any] = dict(defaults)
    if payload:
        merged.update(payload)
    if extra:
        merged.update(extra)
    message.update(merged)
    return message


def build_message(
    message_type: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Construct a message with a validated ``type`` field."""

    key = _normalise_key(message_type)
    if key not in MESSAGES:
        raise ValueError(f"Unknown MousePortal message type: {message_type}")
    message: Dict[str, Any] = {"type": key}
    if payload:
        message.update(payload)
    if extra:
        message.update(extra)
    return message


def _missing_required(required: Sequence[str], payload: Mapping[str, Any]) -> Tuple[str, ...]:
    missing: list[str] = []
    for field in required:
        if field not in payload or payload[field] is None:
            missing.append(field)
    return tuple(missing)


def _missing_one_of(groups: Iterable[Sequence[str]], payload: Mapping[str, Any]) -> Tuple[str, ...]:
    missing: list[str] = []
    for group in groups:
        if not any(field in payload and payload[field] is not None for field in group):
            missing.append("|".join(group))
    return tuple(missing)


def validate_command_payload(
    command: str,
    payload: Mapping[str, Any],
) -> Tuple[bool, Tuple[str, ...], Optional[CommandSpec]]:
    """Check whether the payload satisfies the command specification."""

    canonical, spec, defaults = resolve_command(command)
    if canonical is None or spec is None:
        return False, ("unknown_command",), None
    combined = dict(defaults)
    combined.update(payload)
    missing_required = _missing_required(spec.required, combined)
    missing_one_of = _missing_one_of(spec.one_of, combined)
    problems = missing_required + missing_one_of
    return len(problems) == 0, problems, spec


def validate_message(
    payload: Mapping[str, Any],
    direction: Optional[str | Direction] = None,
) -> Tuple[bool, Tuple[str, ...], Optional[MessageSpec]]:
    """Validate a message against the schema catalog."""

    msg_type_value = payload.get("type")
    if not isinstance(msg_type_value, str):
        return False, ("missing:type",), None
    key = _normalise_key(msg_type_value)
    spec = MESSAGES.get(key)
    if spec is None:
        return False, (f"unknown_type:{msg_type_value}",), None

    if isinstance(direction, Direction):
        direction_value = direction.value
    else:
        direction_value = direction

    if direction_value and spec.direction.value not in {direction_value, BIDIRECTIONAL}:
        return False, (f"wrong_direction:{spec.direction.value}",), spec

    missing_required = _missing_required(spec.required, payload)
    missing_one_of = _missing_one_of(spec.one_of, payload)
    problems = missing_required + missing_one_of
    return len(problems) == 0, problems, spec


def first_present(payload: Mapping[str, Any], *candidates: str) -> Optional[Any]:
    """Return the first non-``None`` value among the provided keys."""

    for key in candidates:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None
