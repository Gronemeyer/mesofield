from __future__ import annotations

"""Human-readable description of the MousePortal messaging protocol.

The protocol is organised around the lifecycle of an experimental procedure:
trials are started or stopped, events are marked, stimuli can be updated, and
health/status telemetry flows in both directions normalising and validating 
JSON payloads exchanged between Mesofield and the Panda3D ``runportal`` process.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

HOST_TO_PORTAL = "host_to_portal"
PORTAL_TO_HOST = "portal_to_host"
BIDIRECTIONAL = "bidirectional"

COMMANDS: Dict[str, Dict[str, Any]] = {
    "start_trial": {
        "title": "Start trial",
        "description": "Begin executing a trial. Provide a mode such as open_loop or closed_loop.",
        "category": "trial",
        "required": (),
        "one_of": (("mode", "trial_type"),),
        "optional": (
            "segments",
            "schedule",
            "loop",
            "repeat",
            "parameters",
            "event_name",
            "sent_time",
        ),
        "aliases": {
            "start": {},
            "start_open_loop": {"mode": "open_loop"},
            "start_closed_loop": {"mode": "closed_loop"},
        },
    },
    "stop_trial": {
        "title": "Stop trial",
        "description": "End the active trial and return to idle.",
        "category": "trial",
        "required": (),
        "one_of": (),
        "optional": ("event_name", "sent_time"),
        "aliases": {"stop": {}},
    },
    "mark_event": {
        "title": "Mark event",
        "description": "Record a behavioural event in the portal log.",
        "category": "event",
        "required": (),
        "one_of": (),
        "optional": ("name", "event", "sent_time", "label"),
        "aliases": {"event": {}, "add_event": {}},
    },
    "set_texture": {
        "title": "Update corridor texture",
        "description": "Swap the texture applied to a corridor surface.",
        "category": "stimulus",
        "required": ("face", "texture"),
        "one_of": (),
        "optional": ("surface", "path"),
        "aliases": {},
    },
    "get_status": {
        "title": "Request status",
        "description": "Ask the portal to emit an immediate status frame.",
        "category": "diagnostic",
        "required": (),
        "one_of": (),
        "optional": ("request_id",),
        "aliases": {"status": {}},
    },
    "ping": {
        "title": "Ping",
        "description": "Measure round-trip time between host and portal.",
        "category": "diagnostic",
        "required": (),
        "one_of": (),
        "optional": ("sent_time", "request_id", "client_time"),
        "aliases": {},
    },
    "shutdown": {
        "title": "Shutdown portal",
        "description": "Terminate the portal application and close sockets.",
        "category": "session",
        "required": (),
        "one_of": (),
        "optional": (),
        "aliases": {"end": {}, "quit": {}, "exit": {}},
    },
}

MESSAGES: Dict[str, Dict[str, Any]] = {
    "command": {
        "description": "Command issued by the host controller.",
        "category": "control",
        "direction": HOST_TO_PORTAL,
        "required": ("command",),
        "one_of": (),
        "optional": ("request_id", "client_time"),
    },
    "status": {
        "description": "Ongoing corridor/trial telemetry from the portal.",
        "category": "telemetry",
        "direction": PORTAL_TO_HOST,
        "required": ("time", "position", "velocity", "state", "trial_mode"),
        "one_of": (),
        "optional": ("input_velocity", "controller", "trial_elapsed", "request_id"),
    },
    "event": {
        "description": "Behavioural event emitted by the portal.",
        "category": "event",
        "direction": PORTAL_TO_HOST,
        "required": ("name", "time_received", "position", "trial_mode"),
        "one_of": (),
        "optional": ("delta", "controller", "time_sent"),
    },
    "ack": {
        "description": "Acknowledgement of a processed command.",
        "category": "control",
        "direction": PORTAL_TO_HOST,
        "required": ("command", "status"),
        "one_of": (),
        "optional": ("request_id", "sent_time", "message"),
    },
    "client_status": {
        "description": "Socket status updates (connecting, connected, disconnected).",
        "category": "telemetry",
        "direction": BIDIRECTIONAL,
        "required": ("status",),
        "one_of": (),
        "optional": ("error", "time"),
    },
    "client_error": {
        "description": "Protocol or transport error that did not close the socket.",
        "category": "telemetry",
        "direction": BIDIRECTIONAL,
        "required": ("error",),
        "one_of": (),
        "optional": ("raw", "time"),
    },
    "stdout": {
        "description": "Forwarded stdout line from the portal process.",
        "category": "logging",
        "direction": PORTAL_TO_HOST,
        "required": ("text",),
        "one_of": (),
        "optional": (),
    },
    "process_exit": {
        "description": "Final exit code emitted when the portal terminates.",
        "category": "logging",
        "direction": PORTAL_TO_HOST,
        "required": ("exit_code",),
        "one_of": (),
        "optional": ("status",),
    },
    "ping": {
        "description": "Ping message from the host.",
        "category": "diagnostic",
        "direction": HOST_TO_PORTAL,
        "required": (),
        "one_of": (),
        "optional": ("client_time", "request_id"),
    },
    "pong": {
        "description": "Response to a ping message.",
        "category": "diagnostic",
        "direction": PORTAL_TO_HOST,
        "required": (),
        "one_of": (),
        "optional": ("client_time", "request_id"),
    },
    "error": {
        "description": "Fatal protocol error reported by the portal.",
        "category": "error",
        "direction": PORTAL_TO_HOST,
        "required": ("message",),
        "one_of": (),
        "optional": ("raw",),
    },
}


def _normalise_key(value: str) -> str:
    return value.strip().lower()


def _alias_defaults(spec: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    aliases = spec.get("aliases") or {}
    if isinstance(aliases, dict):
        return {_normalise_key(name): dict(defaults or {}) for name, defaults in aliases.items()}
    if isinstance(aliases, (list, tuple, set)):
        return {_normalise_key(name): {} for name in aliases}
    return {}


def resolve_command(command: str | None) -> Tuple[Optional[str], Optional[Dict[str, Any]], Dict[str, Any]]:
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


def lookup_command(command: str | None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
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
) -> Tuple[bool, Tuple[str, ...], Optional[Dict[str, Any]]]:
    """Check whether the payload satisfies the command specification."""

    canonical, spec, defaults = resolve_command(command)
    if canonical is None or spec is None:
        return False, ("unknown_command",), None
    combined = dict(defaults)
    combined.update(payload)
    missing_required = _missing_required(spec.get("required", ()), combined)
    missing_one_of = _missing_one_of(spec.get("one_of", ()), combined)
    problems = missing_required + missing_one_of
    return len(problems) == 0, problems, spec


def validate_message(
    payload: Mapping[str, Any],
    direction: Optional[str] = None,
) -> Tuple[bool, Tuple[str, ...], Optional[Dict[str, Any]]]:
    """Validate a message against the schema catalog."""

    msg_type_value = payload.get("type")
    if not isinstance(msg_type_value, str):
        return False, ("missing:type",), None
    key = _normalise_key(msg_type_value)
    spec = MESSAGES.get(key)
    if spec is None:
        return False, (f"unknown_type:{msg_type_value}",), None

    if direction and spec.get("direction") not in {direction, BIDIRECTIONAL}:
        return False, (f"wrong_direction:{spec['direction']}",), spec

    missing_required = _missing_required(spec.get("required", ()), payload)
    missing_one_of = _missing_one_of(spec.get("one_of", ()), payload)
    problems = missing_required + missing_one_of
    return len(problems) == 0, problems, spec


def first_present(payload: Mapping[str, Any], *candidates: str) -> Optional[Any]:
    """Return the first non-``None`` value among the provided keys."""

    for key in candidates:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None
