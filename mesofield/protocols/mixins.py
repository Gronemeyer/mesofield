"""
Mixin classes for implementing hardware protocols with different threading models.

This module provides mixin classes that can be used to implement hardware device
protocols in conjunction with different threading models (threading, QThread, asyncio).
"""

from __future__ import annotations

import threading


class ThreadedHardwareDevice:
    """
    Mixin for implementing the HardwareDevice protocol with Python's threading.

    This mixin provides the basic structure for a hardware device that uses
    Python's threading module. It handles the thread creation, starting, and stopping.
    
    Usage:
    ```python
    class MySensor(ThreadingHardwareDeviceMixin):
        device_type = "sensor"
        device_id = "my_sensor"
        
        def __init__(self, config=None):
            super().__init__()
            self.config = config or {}
            
        def initialize(self):
            # Initialize your device
            pass
            
        def _run(self):
            # This is called in the thread context
            while not self._stop_event.is_set():
                # Do work
                pass
                
        def get_status(self):
            return {"active": not self._stop_event.is_set()}
    ```
    """
    
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._active = False
    
    def start(self) -> bool:
        """Start the device thread."""
        if self._active:
            return True
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        self._active = True
        return True
    
    def stop(self) -> bool:
        """Stop the device thread."""
        if not self._active:
            return True
            
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._active = False
        return True
    
    def close(self) -> None:
        """Close the device and clean up resources."""
        self.stop()
    
    def _run(self) -> None:
        """
        Main thread method to be overridden by subclasses.
        
        This method runs in a separate thread when start() is called.
        It should check self._stop_event periodically and exit if it's set.
        """
        raise NotImplementedError("Subclasses must implement _run()")


class AsyncioHardwareDevice:
    """
    Mixin for implementing the HardwareDevice protocol with asyncio.
    
    This mixin provides the basic structure for a hardware device that uses
    Python's asyncio module. It handles the task creation, starting, and cancellation.
    
    Usage:
    ```python
    class MySensor(AsyncioHardwareDeviceMixin):
        device_type = "sensor"
        device_id = "my_sensor"
        
        def __init__(self, loop=None, config=None):
            super().__init__(loop)
            self.config = config or {}
            
        def initialize(self):
            # Initialize your device
            pass
            
        async def _run(self):
            # This is the coroutine that runs as a task
            while True:
                # Do work
                if self._should_stop():
                    break
                await asyncio.sleep(0.01)
                
        def get_status(self):
            return {"active": self._task is not None and not self._task.done()}
    ```
    """
    
    def __init__(self, loop=None):
        import asyncio
        self._loop = loop or asyncio.get_event_loop()
        self._task = None
        self._stop_requested = False
    
    def start(self) -> bool:
        """Start the device task."""
        import asyncio
        if self._task is not None and not self._task.done():
            return True
            
        self._stop_requested = False
        self._task = asyncio.create_task(self._run())
        return True
    
    def stop(self) -> bool:
        """Stop the device task."""
        if self._task is None or self._task.done():
            return True
            
        self._stop_requested = True
        self._task.cancel()
        return True
    
    def close(self) -> None:
        """Close the device and clean up resources."""
        self.stop()
    
    def _should_stop(self) -> bool:
        """Check if the task should stop."""
        return self._stop_requested
    
    async def _run(self) -> None:
        """
        Main coroutine to be overridden by subclasses.
        
        This coroutine runs as a task when start() is called.
        It should check self._should_stop() periodically and exit if it returns True.
        """
        raise NotImplementedError("Subclasses must implement _run()")