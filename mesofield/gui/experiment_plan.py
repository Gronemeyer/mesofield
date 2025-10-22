"""Compatibility wrapper for experiment planning helpers.

The experiment scheduling logic now lives inside ``runportal.py`` so that the
Panda3D application can operate in isolation from external modules when
launched via the MousePortal subprocess.  The symbols remain re-exported here
to avoid breaking older imports while keeping the single source of truth inside
``runportal``.
"""

from __future__ import annotations

from .runportal import ExperimentPlan, TrialDefinition

__all__ = ["ExperimentPlan", "TrialDefinition"]
