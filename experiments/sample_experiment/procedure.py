"""Sample custom Procedure for the sample_experiment.

Discovered at runtime via the ``procedure_file`` and ``procedure_class``
fields in ``experiment.json``.  See
:func:`mesofield.base.load_procedure_from_config`.
"""

from mesofield.base import Procedure


class SampleProcedure(Procedure):
    """Trivial subclass that demonstrates the extension hooks.

    Replace the body of :meth:`prerun` / :meth:`on_started` /
    :meth:`on_finished` with experiment-specific logic.  Multi-camera
    sync, encoder/PsychoPy/NIDAQ orchestration, and cleanup are all
    handled by the base ``Procedure``.
    """

    def prerun(self) -> None:
        self.logger.info("SampleProcedure.prerun: nothing custom to do")

    def on_started(self) -> None:
        self.logger.info("SampleProcedure.on_started")

    def on_finished(self) -> None:
        self.logger.info("SampleProcedure.on_finished")
