"""
Example custom procedures demonstrating how to create user-defined experimental workflows.

This module provides examples of how to subclass :class:`mesofield.base.Procedure`
to create custom experimental protocols that integrate with the Mesofield GUI.
"""

import time
import logging
from typing import Optional, Dict, Any

from mesofield.base import Procedure


class SimpleBehaviorProcedure(Procedure):
    """
    Example procedure for a simple behavior experiment.
    
    This procedure extends :class:`Procedure` to add custom behavior-specific logic
    while maintaining the standard camera and encoder recording workflow.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        if not self.config.has("protocol"):
            self.config.set("protocol", "simple_behavior_001")
        if not self.config.has("experimenter"):
            self.config.set("experimenter", "researcher")
        self.config.notes.extend(["Simple behavior paradigm", "Mouse running on treadmill"])
        
        # Behavior-specific parameters
        self.trial_duration = 300  # 5 minutes in seconds
        self.baseline_duration = 60  # 1 minute baseline
        self.stimulus_duration = 180  # 3 minutes stimulus
        
    def setup_experiment(self):
        """Setup behavior-specific experimental parameters."""
        super().setup_experiment()
        
        # Additional behavior setup
        self.logger.info("Setting up behavior experiment parameters")
        self.logger.info(f"Trial duration: {self.trial_duration}s")
        self.logger.info(f"Baseline: {self.baseline_duration}s, Stimulus: {self.stimulus_duration}s")
        
        # Configure any behavior-specific hardware here
        # Example: setup stimulus delivery system, reward system, etc.
        
    def run_trial(self):
        """Run a single behavior trial with baseline and stimulus periods."""
        self.logger.info("Starting behavior trial")
        
        # Baseline period
        self.logger.info("Baseline period started")
        time.sleep(self.baseline_duration)
        
        # Stimulus period
        self.logger.info("Stimulus period started")
        # Here you would trigger your behavioral stimulus
        # Example: deliver visual stimulus, play sound, etc.
        time.sleep(self.stimulus_duration)
        
        # Post-stimulus period
        remaining_time = self.trial_duration - self.baseline_duration - self.stimulus_duration
        if remaining_time > 0:
            self.logger.info("Post-stimulus period")
            time.sleep(remaining_time)
        
        self.logger.info("Behavior trial completed")


class MultiTrialProcedure(Procedure):
    """
    Example procedure that runs multiple trials with custom logic.
    
    This demonstrates how to create a completely custom procedure from :class:`Procedure`
    with full control over the experimental workflow.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        if not self.config.has("protocol"):
            self.config.set("protocol", "multi_trial_001")
        if not self.config.has("experimenter"):
            self.config.set("experimenter", "researcher")
        self.config.notes.extend(["Multiple trial experiment", "Custom inter-trial intervals"])
        
        # Trial parameters
        self.num_trials = 5
        self.trial_duration = 120  # 2 minutes per trial
        self.inter_trial_interval = 30  # 30 seconds between trials
        self.current_trial = 0
        
    def setup_experiment(self):
        """Setup multi-trial experiment."""
        super().setup_experiment()
        self.logger.info(f"Setting up {self.num_trials} trials")
        self.logger.info(f"Trial duration: {self.trial_duration}s")
        self.logger.info(f"Inter-trial interval: {self.inter_trial_interval}s")
        
    def run(self):
        """Run the complete multi-trial experiment."""
        try:
            self.setup_experiment()
            
            # Start data collection systems
            self.start_cameras()
            self.start_encoder()
            
            # Run multiple trials
            for trial_num in range(1, self.num_trials + 1):
                self.current_trial = trial_num
                self.logger.info(f"Starting trial {trial_num}/{self.num_trials}")
                
                # Run single trial
                self.run_single_trial()
                
                # Inter-trial interval (except after last trial)
                if trial_num < self.num_trials:
                    self.logger.info(f"Inter-trial interval ({self.inter_trial_interval}s)")
                    time.sleep(self.inter_trial_interval)
            
            self.logger.info("All trials completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during multi-trial experiment: {e}")
            raise
        finally:
            self.cleanup()
    
    def run_single_trial(self):
        """Run a single trial of the experiment."""
        start_time = time.time()
        
        # Mark trial start
        self.logger.info(f"Trial {self.current_trial} started")
        
        # Custom trial logic here
        # Example: present stimuli, collect behavioral responses, etc.
        time.sleep(self.trial_duration)
        
        # Mark trial end
        duration = time.time() - start_time
        self.logger.info(f"Trial {self.current_trial} completed in {duration:.1f}s")


class OptoStimulationProcedure(Procedure):
    """
    Example procedure for optogenetic stimulation experiments.
    
    This demonstrates how to integrate hardware control (LED stimulation)
    with the standard Mesofield recording workflow.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        if not self.config.has("protocol"):
            self.config.set("protocol", "opto_stim_001")
        if not self.config.has("experimenter"):
            self.config.set("experimenter", "researcher")
        self.config.notes.extend(["Optogenetic stimulation", "Blue light, 473nm", "10Hz pulses"])
        
        # Stimulation parameters
        self.stim_frequency = 10  # Hz
        self.stim_duration = 5  # seconds
        self.stim_start_time = 60  # start stimulation after 60s baseline
        self.total_duration = 300  # 5 minutes total
        
    def setup_experiment(self):
        """Setup optogenetic stimulation parameters."""
        super().setup_experiment()
        
        self.logger.info("Configuring optogenetic stimulation")
        self.logger.info(f"Stimulation frequency: {self.stim_frequency} Hz")
        self.logger.info(f"Stimulation duration: {self.stim_duration}s")
        self.logger.info(f"Stimulation starts at: {self.stim_start_time}s")
        
        # Configure LED stimulation hardware
        # This would typically involve setting up the Arduino-Switch or similar device
        
    def run_trial(self):
        """Run optogenetic stimulation trial."""
        start_time = time.time()
        
        # Baseline period
        self.logger.info("Baseline recording started")
        time.sleep(self.stim_start_time)
        
        # Stimulation period
        self.logger.info("Starting optogenetic stimulation")
        self.deliver_optogenetic_stimulation()
        
        # Post-stimulation recording
        remaining_time = self.total_duration - self.stim_start_time - self.stim_duration
        if remaining_time > 0:
            self.logger.info("Post-stimulation recording")
            time.sleep(remaining_time)
        
        total_time = time.time() - start_time
        self.logger.info(f"Optogenetic trial completed in {total_time:.1f}s")
    
    def deliver_optogenetic_stimulation(self):
        """Deliver optogenetic stimulation pulses."""
        # This is a simplified example - in practice you'd control actual hardware
        pulse_interval = 1.0 / self.stim_frequency
        num_pulses = int(self.stim_duration * self.stim_frequency)
        
        self.logger.info(f"Delivering {num_pulses} pulses at {self.stim_frequency} Hz")
        
        for pulse in range(num_pulses):
            # Turn on LED
            self.logger.debug(f"Pulse {pulse + 1}/{num_pulses}")
            # Here you would send commands to your LED controller
            # Example: self.config.hardware.arduino_switch.pulse_led()
            
            time.sleep(pulse_interval)
        
        self.logger.info("Optogenetic stimulation completed")


# Factory function for easy procedure creation
def create_custom_procedure(procedure_type: str, config_path: Optional[str], **kwargs) -> Procedure:
    """
    Factory function to create custom procedures by name.
    
    Args:
        procedure_type: Name of the procedure type
        config_path: Path to experiment JSON configuration
        **kwargs: Additional keyword arguments for procedure configuration
    
    Returns:
        Configured procedure instance
    """
    procedure_classes = {
        'simple_behavior': SimpleBehaviorProcedure,
        'multi_trial': MultiTrialProcedure,
        'opto_stimulation': OptoStimulationProcedure,
    }
    
    if procedure_type not in procedure_classes:
        available = ', '.join(procedure_classes.keys())
        raise ValueError(f"Unknown procedure type '{procedure_type}'. Available: {available}")
    
    ProcedureClass = procedure_classes[procedure_type]
    procedure = ProcedureClass(config_path)
    for key, value in kwargs.items():
        procedure.config.set(key, value)
    return procedure


if __name__ == "__main__":
    # Example usage
    # Path to experiment configuration JSON
    config_path = "experiment.json"
    
    # Create and run a simple behavior procedure
    procedure = create_custom_procedure(
        'simple_behavior',
        config_path,
        protocol="test_behavior_001",
        experimenter="test_user"
    )
    
    print(f"Created procedure: {procedure.__class__.__name__}")
    print(f"Protocol: {procedure.config.get('protocol')}")
    
    # In practice, you would call procedure.run() to execute the experiment
    # procedure.run()
