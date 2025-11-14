# Custom Procedure Implementation Guide

This guide explains how to implement custom procedures in Mesofield to define your own experimental workflows that integrate seamlessly with the GUI and hardware control systems.

## Overview

The Mesofield procedure system allows you to:
- Define custom experimental workflows by subclassing existing procedure classes
- Control hardware initialization, data collection, and experimental timing
- Integrate with the Mesofield GUI for easy experiment execution
- Maintain consistent data organization and logging

## Getting Started

### 1. Basic Procedure Structure

All custom procedures should inherit from the `Procedure` class:

```python
from typing import Optional, Dict, Any

from mesofield.base import Procedure
from mesofield.config import ExperimentConfig


class MyCustomProcedure(Procedure):
    def __init__(self, config: ExperimentConfig, *, overrides: Optional[Dict[str, Any]] = None):
        super().__init__(config, overrides=overrides)
        # Add your custom initialization here

    def setup_experiment(self) -> None:
        """Override to add custom setup logic."""
        # Your custom setup code here

    def run_trial(self) -> None:
        """Override to define your experimental trial logic."""
        # Your trial implementation here
```

### 2. Choosing the Right Base Class

**Use `Procedure` when:**

- You want the standard neuroscience workflow (camera recording + encoder tracking)
- You need to add custom logic to an existing experimental framework
- You want automatic hardware initialization and cleanup or full control over the workflow

## Implementation Examples

See `custom_procedures.py` for complete working examples of:

- `SimpleBehaviorProcedure`: Basic behavior experiment with baseline and stimulus periods
- `MultiTrialProcedure`: Multiple trial experiment with custom timing
- `OptoStimulationProcedure`: Optogenetic stimulation with LED control

## Hardware Integration

### Camera Control

```python
for camera in self.hardware.cameras:
    camera.start()
```

### Encoder/Treadmill Control

```python
self.hardware.encoder.start_recording()
```

### LED/Stimulation Control

```python
def setup_stimulation(self):
    """Configure LED stimulation hardware."""
    if hasattr(self.config.hardware, 'arduino_switch'):
        # Configure stimulation parameters
        pass
```

## GUI Integration

### Launching with Custom Procedures

Use the command line interface to launch Mesofield with your custom procedure:

```bash
python -m mesofield --procedure-class my_module.MyCustomProcedure --experiment-id exp_001
```

### Procedure Configuration

Configure procedures through the GUI using the "Configure Procedure" button when available.

## Data Organization

### Automatic Data Paths

```python
# These paths are created based on your ExperimentConfig settings
camera_path = self.config.make_path("meso", "ome.tiff", bids_type="func")
encoder_path = self.config.make_path('treadmill_data', 'csv', 'beh')
log_path = self.config.make_path('experiment_log', 'log')
```

## Logging and Monitoring

### Using the Built-in Logger

```python
def run_trial(self):
    self.logger.info("Trial started")  # Info level
    self.logger.debug("Debug information")  # Debug level
    self.logger.warning("Warning message")  # Warning level
    self.logger.error("Error occurred")  # Error level
```

## Error Handling

### Robust Procedure Implementation

```python
def run(self):
    try:
        self.setup_experiment()
        self.prerun()

        if self.data is None:
            raise RuntimeError("Data manager not initialized")

        self.hardware.encoder.start_recording()
        for cam in self.hardware.cameras:
            cam.start()

        # Your experiment logic
        self.run_experiment()

    except KeyboardInterrupt:
        self.logger.info("Experiment interrupted by user")
    except Exception as exc:
        self.logger.error(f"Experiment failed: {exc}")
        raise
    finally:
        # Always cleanup, even if errors occur
        self.stopped_time = datetime.now()
        self._cleanup_procedure()
```

## Best Practices

1. **Always call `super().__init__()`** in your constructor
2. **Use the logger** for all output instead of print statements
3. **Implement proper error handling** with try/except blocks
4. **Test with short durations** before running long experiments
5. **Use meaningful experiment IDs** and notes
6. **Document your parameters** clearly for other users
7. **Organize your code** into logical methods (setup, run_trial, cleanup)

For complete examples and advanced usage, see the `custom_procedures.py` file in this directory.
