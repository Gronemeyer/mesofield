"""Sample custom Procedure used by tests/test_procedure_discovery.py.

Declares a trivial Procedure subclass so the discovery path
(experiment.json -> procedure_file/procedure_class -> imported subclass)
has something concrete to resolve to. No behavior beyond the base class.
"""

from __future__ import annotations

from mesofield.base import Procedure


class SampleProcedure(Procedure):
    """Marker subclass; inherits the generic run/cleanup unchanged."""
