"""Scaffolding for new experiments and machine-level rig configurations.

Both halves are "get a machine/experiment ready to use" concerns:

- :func:`scaffold_experiment` generates a runnable experiment directory.
- :mod:`mesofield.scaffold.rigs` keeps a per-machine store of canonical
  ``hardware.yaml`` files that ``mesofield init`` copies into new experiments.
"""

from mesofield.scaffold.experiment import (
    scaffold_experiment,
    hardware_yaml_template,
)
from mesofield.scaffold.rigs import (
    rigs_dir,
    list_rigs,
    rig_path,
    add_rig,
    new_rig,
    remove_rig,
)

__all__ = [
    "scaffold_experiment",
    "hardware_yaml_template",
    "rigs_dir",
    "list_rigs",
    "rig_path",
    "add_rig",
    "new_rig",
    "remove_rig",
]
