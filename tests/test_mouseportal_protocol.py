from __future__ import annotations

from mesofield.gui import portal_protocol


def test_build_command_message_normalises_aliases() -> None:
    message = portal_protocol.build_command_message("end", request_id="req-1")
    assert message["type"] == "command"
    assert message["command"] == "shutdown"
    assert message["request_id"] == "req-1"


def test_validate_command_payload_requires_mode_for_start_trial() -> None:
    message = portal_protocol.build_command_message("start_trial", request_id="req-2")
    ok, issues, _ = portal_protocol.validate_command_payload("start_trial", message)
    assert not ok
    assert any(token in {"trial_type|mode", "mode|trial_type"} for token in issues)

    with_mode = portal_protocol.build_command_message(
        "start_trial",
        request_id="req-3",
        payload={"trial_type": "closed_loop"},
    )
    ok2, issues2, _ = portal_protocol.validate_command_payload("start_trial", with_mode)
    assert ok2, issues2


def test_build_status_message_validates() -> None:
    status = portal_protocol.build_message(
        "status",
        time=1.23,
        position=4.5,
        velocity=1.2,
        state="Running",
        trial_mode="closed_loop",
    )
    ok, issues, _ = portal_protocol.validate_message(status, portal_protocol.PORTAL_TO_HOST)
    assert ok, issues


def test_validate_message_unknown_type() -> None:
    ok, issues, spec = portal_protocol.validate_message({"type": "something_else"})
    assert not ok
    assert issues[0].startswith("unsupported_type") or issues[0].startswith("unknown_type")
    assert spec is None


def test_first_present_returns_first_value() -> None:
    payload = {"mode": "open_loop", "trial_type": None}
    value = portal_protocol.first_present(payload, "trial_type", "mode")
    assert value == "open_loop"
