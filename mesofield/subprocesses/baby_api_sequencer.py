"""Baby API experiment engine utilities.

The original prototype has been reshaped to align with the integration plan for
MousePortal.  The engine remains generic and relies on registries for routines
and block policies so that downstream projects can extend behaviour without
touching the core runtime.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Type


# ---------------------------------------------------------------------------
# registries


_ROUTINES: Dict[str, Type["Routine"]] = {}
_POLICIES: Dict[str, Type["BlockPolicy"]] = {}


def routine(name: str) -> Callable[[Type["Routine"]], Type["Routine"]]:
    def _wrap(cls: Type["Routine"]) -> Type["Routine"]:
        key = name.strip().lower()
        _ROUTINES[key] = cls
        cls.kind = key  # type: ignore[attr-defined]
        return cls

    return _wrap


def policy(name: str) -> Callable[[Type["BlockPolicy"]], Type["BlockPolicy"]]:
    def _wrap(cls: Type["BlockPolicy"]) -> Type["BlockPolicy"]:
        key = name.strip().lower()
        _POLICIES[key] = cls
        cls.kind = key  # type: ignore[attr-defined]
        return cls

    return _wrap


# ---------------------------------------------------------------------------
# helpers


def _parse_duration(expr: str) -> float:
    s = expr.strip().lower()
    if s.endswith("ms"):
        return float(s[:-2]) / 1000.0
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60.0
    return float(s)


def sample_range(rng: random.Random, expr: str) -> float:
    if ".." in expr:
        a, b = expr.split("..", 1)
        return rng.uniform(_parse_duration(a), _parse_duration(b))
    return _parse_duration(expr)


# ---------------------------------------------------------------------------
# event log


@dataclass
class Event:
    abs_time: float
    rel_time: float
    level: str
    label: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 1
        return payload


class EventLog:
    def __init__(self) -> None:
        self._trial_t0 = 0.0
        self._events: List[Event] = []
        self._callback: Optional[Callable[[Event], None]] = None

    def set_trial_t0(self, t: float) -> None:
        self._trial_t0 = t

    def add(self, *, now: float, level: str, label: str, **data: Any) -> Event:
        event = Event(now, now - self._trial_t0, level, label, data)
        self._events.append(event)
        if self._callback:
            self._callback(event)
        return event

    def set_callback(self, cb: Optional[Callable[[Event], None]]) -> None:
        self._callback = cb

    def clear(self) -> None:
        self._events.clear()
        self._trial_t0 = 0.0

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([event.to_dict() for event in self._events], f, indent=2)


# ---------------------------------------------------------------------------
# spec dataclasses


@dataclass(frozen=True)
class TrialSpec:
    name: str
    routine_type: str
    cfg: Dict[str, Any]


@dataclass(frozen=True)
class BlockSpec:
    name: str
    policy: str
    repeats: int
    trials: List[TrialSpec]


@dataclass(frozen=True)
class ExperimentSpec:
    rng_seed: int
    blocks: List[BlockSpec]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentSpec":
        blocks: List[BlockSpec] = []
        for block in data.get("blocks", []):
            trials = [
                TrialSpec(
                    name=trial["name"],
                    routine_type=trial["routine"],
                    cfg=dict(trial.get("config", {})),
                )
                for trial in block.get("trials", [])
            ]
            blocks.append(
                BlockSpec(
                    name=block["name"],
                    policy=block.get("policy", "sequential"),
                    repeats=int(block.get("repeats", 1)),
                    trials=trials,
                )
            )
        return cls(rng_seed=int(data.get("rng_seed", 0)), blocks=blocks)


@dataclass(frozen=True)
class TrialInstance:
    block_name: str
    trial_name: str
    spec: TrialSpec


# ---------------------------------------------------------------------------
# policies


class BlockPolicy:
    kind: str

    def __init__(self, block: BlockSpec, rng: random.Random):
        self.block = block
        self.rng = rng

    def sequence(self) -> Iterable[TrialInstance]:  # pragma: no cover - interface
        raise NotImplementedError


@policy("sequential")
class SequentialPolicy(BlockPolicy):
    def sequence(self) -> Iterable[TrialInstance]:
        for _ in range(self.block.repeats):
            for trial in self.block.trials:
                yield TrialInstance(self.block.name, trial.name, trial)


@policy("random_no_replacement")
class RandomNRPolicy(BlockPolicy):
    def sequence(self) -> Iterable[TrialInstance]:
        for _ in range(self.block.repeats):
            idxs = list(range(len(self.block.trials)))
            self.rng.shuffle(idxs)
            for idx in idxs:
                trial = self.block.trials[idx]
                yield TrialInstance(self.block.name, trial.name, trial)


# ---------------------------------------------------------------------------
# routines


class Routine:
    kind: str

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self, eng: "Engine", inst: TrialInstance) -> Generator[None, None, None]:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------
# engine


class Engine:
    def __init__(
        self,
        spec: ExperimentSpec,
        *,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.spec = spec
        self.rng = random.Random(spec.rng_seed)
        self.time_fn = time_fn or time.perf_counter
        self.log = EventLog()
        self.capabilities: Dict[str, bool] = {}
        self._control_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None
        self._event_cb: Optional[Callable[[Event], None]] = None
        self._trial_iter: Iterator[TrialInstance] = iter([])
        self._current_trial: Optional[Generator[None, None, None]] = None
        self._current_instance: Optional[TrialInstance] = None
        self._current_block: Optional[str] = None
        self._active = False
        self._t0 = 0.0

    # configuration --------------------------------------------------
    def set_control_callback(self, cb: Optional[Callable[[str, Dict[str, Any]], Any]]) -> None:
        self._control_cb = cb

    def set_event_callback(self, cb: Optional[Callable[[Event], None]]) -> None:
        self._event_cb = cb
        self.log.set_callback(cb)

    # lifecycle ------------------------------------------------------
    def start(self) -> None:
        self.log.clear()
        self._t0 = self.time_fn()
        self._trial_iter = iter(self._yield_trials())
        self._current_trial = None
        self._current_instance = None
        self._current_block = None
        self._active = True
        self.log_event("engine", "engine_start")

    def stop(self) -> EventLog:
        if self._active:
            self.log_event("engine", "engine_stop")
        self._active = False
        return self.log

    @property
    def running(self) -> bool:
        return self._active

    # utilities ------------------------------------------------------
    def now(self) -> float:
        return self.time_fn() - self._t0

    def wait(self, seconds: float) -> Generator[None, None, None]:
        target = self.now() + seconds
        while self.now() < target:
            yield None

    def log_event(self, level: str, label: str, **data: Any) -> Event:
        return self.log.add(now=self.now(), level=level, label=label, **data)

    def control(self, name: str, **params: Any) -> bool:
        supported = self.capabilities.get(name, False)
        if not supported:
            self.log_event("warning", "control_unsupported", control=name, params=params)
            return False
        if self._control_cb is None:
            self.log_event("warning", "control_unwired", control=name, params=params)
            return False
        try:
            self._control_cb(name, params)
            self.log_event("control", name, **params)
            return True
        except Exception as exc:  # pragma: no cover - runtime errors
            self.log_event("error", "control_failed", control=name, error=str(exc), params=params)
            return False

    # main loop ------------------------------------------------------
    def step(self) -> bool:
        if not self._active:
            return False

        if self._current_trial is None:
            try:
                inst = next(self._trial_iter)
            except StopIteration:
                if self._current_block is not None:
                    self.log_event("block", "block_end", block=self._current_block)
                self.stop()
                return False
            if inst.block_name != self._current_block:
                if self._current_block is not None:
                    self.log_event("block", "block_end", block=self._current_block)
                self._current_block = inst.block_name
                self.log_event("block", "block_start", block=self._current_block)
            self.log.set_trial_t0(self.now())
            self.log_event("trial", "trial_start", block=inst.block_name, trial=inst.trial_name)
            routine_cls = _ROUTINES[inst.spec.routine_type.lower()]
            self._current_instance = inst
            self._current_trial = routine_cls(inst.spec.cfg).run(self, inst)

        assert self._current_trial is not None
        try:
            next(self._current_trial)
        except StopIteration:
            inst = self._current_instance
            if inst is not None:
                self.log_event("trial", "trial_end", block=inst.block_name, trial=inst.trial_name)
            self._current_trial = None
            self._current_instance = None

        return self._active

    # internals ------------------------------------------------------
    def _yield_trials(self) -> Iterable[TrialInstance]:
        for block in self.spec.blocks:
            policy_cls = _POLICIES.get(block.policy.lower())
            if policy_cls is None:
                raise KeyError(f"Unknown policy '{block.policy}'")
            for inst in policy_cls(block, self.rng).sequence():
                yield inst


# ---------------------------------------------------------------------------
# routines


@routine("open_loop")
class OpenLoopRoutine(Routine):
    """Open loop routine with random glitches (gain changes and input reversals)."""
    
    def run(self, eng: "Engine", inst: TrialInstance) -> Generator[None, None, None]:
        # Get trial duration
        duration_min = float(self.cfg.get("duration_min_s", 10.0))
        duration_max = float(self.cfg.get("duration_max_s", 60.0))
        trial_duration = eng.rng.uniform(duration_min, duration_max)
        
        # Get glitch configuration
        glitch_cfg = self.cfg.get("glitch", {})
        count_min = int(glitch_cfg.get("count_min", 2))
        count_max = int(glitch_cfg.get("count_max", 5))
        glitch_duration_min = float(glitch_cfg.get("duration_min_s", 0.3))
        glitch_duration_max = float(glitch_cfg.get("duration_max_s", 1.5))
        gap_min = float(glitch_cfg.get("inter_glitch_gap_min_s", 0.2))
        gap_max = float(glitch_cfg.get("inter_glitch_gap_max_s", 1.0))
        gain_min = float(glitch_cfg.get("gain_min", 0.5))
        gain_max = float(glitch_cfg.get("gain_max", 1.5))
        reverse_prob = float(glitch_cfg.get("reverse_probability", 0.4))
        
        # Generate glitch schedule
        num_glitches = eng.rng.randint(count_min, count_max)
        glitch_times = []
        current_time = 0.0
        
        for _ in range(num_glitches):
            # Add gap before next glitch
            gap = eng.rng.uniform(gap_min, gap_max)
            current_time += gap
            
            # Check if we have time for a glitch
            glitch_duration = eng.rng.uniform(glitch_duration_min, glitch_duration_max)
            if current_time + glitch_duration > trial_duration:
                break
                
            # Add the glitch
            glitch_gain = eng.rng.uniform(gain_min, gain_max)
            glitch_reverse = eng.rng.random() < reverse_prob
            glitch_times.append((current_time, glitch_duration, glitch_gain, glitch_reverse))
            current_time += glitch_duration
        
        # Log trial start
        eng.log_event("info", "trial_start", trial=inst.trial_name, duration_s=trial_duration, glitches=len(glitch_times))
        
        # Execute trial with glitches
        start_time = eng.time_fn()
        glitch_index = 0
        in_glitch = False
        glitch_end_time = 0.0
        
        while True:
            current_time = eng.time_fn() - start_time
            
            # Check if trial is complete
            if current_time >= trial_duration:
                break
                
            # Check for glitch start
            if not in_glitch and glitch_index < len(glitch_times):
                glitch_start, glitch_dur, glitch_gain, glitch_reverse = glitch_times[glitch_index]
                if current_time >= glitch_start:
                    # Start glitch
                    eng.control("set_gain", gain=glitch_gain)
                    eng.control("set_reverse", flag=glitch_reverse)
                    eng.log_event("info", "glitch_start", gain=glitch_gain, reverse=glitch_reverse)
                    in_glitch = True
                    glitch_end_time = glitch_start + glitch_dur
            
            # Check for glitch end
            if in_glitch and current_time >= glitch_end_time:
                # End glitch - restore defaults
                eng.control("set_gain", gain=1.0)
                eng.control("set_reverse", flag=False)
                eng.log_event("info", "glitch_end")
                in_glitch = False
                glitch_index += 1
            
            yield
        
        # End glitch if still active
        if in_glitch:
            eng.control("set_gain", gain=1.0)
            eng.control("set_reverse", flag=False)
            eng.log_event("info", "glitch_end")
        
        eng.log_event("info", "trial_end", trial=inst.trial_name)


@routine("closed_loop")
class ClosedLoopRoutine(Routine):
    """Closed loop routine - normal corridor movement without glitches."""
    
    def run(self, eng: "Engine", inst: TrialInstance) -> Generator[None, None, None]:
        # Get trial duration
        duration_min = float(self.cfg.get("duration_min_s", 60.0))
        duration_max = float(self.cfg.get("duration_max_s", 300.0))
        trial_duration = eng.rng.uniform(duration_min, duration_max)
        
        # Ensure normal gain and input settings
        eng.control("set_gain", gain=1.0)
        eng.control("set_reverse", flag=False)
        
        # Log trial start
        eng.log_event("info", "trial_start", trial=inst.trial_name, duration_s=trial_duration)
        
        # Wait for duration
        yield from eng.wait(trial_duration)
        
        eng.log_event("info", "trial_end", trial=inst.trial_name)


__all__ = [
    "Engine",
    "Event",
    "EventLog",
    "ExperimentSpec",
    "TrialSpec",
    "BlockSpec",
    "Routine",
    "BlockPolicy",
    "routine",
    "policy",
    "sample_range",
]
