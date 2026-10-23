# API Reference

Mesofield is organised into a small set of subpackages. Each one is documented below; start with the package you're working in.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} `base` & `protocols`
:link: mesofield.base
:link-type: doc

The `Procedure` orchestrator and the device protocols every hardware adapter implements.
:::

:::{grid-item-card} `config` & `signals`
:link: mesofield.config
:link-type: doc

The `ExperimentConfig` registry and the lightweight psygnal-based event bus.
:::

:::{grid-item-card} `hardware` & `engines`
:link: mesofield.hardware
:link-type: doc

`HardwareManager` lifecycle and the MicroManager MDA engine integrations.
:::

:::{grid-item-card} `devices`
:link: mesofield.devices
:link-type: doc

Concrete hardware adapters: cameras, DAQs, encoders, treadmills, PsychoPy.
:::

:::{grid-item-card} `data` & `datakit`
:link: mesofield.data
:link-type: doc

Writers, batch managers, and the post-acquisition data exploration toolkit.
:::

:::{grid-item-card} `gui`
:link: mesofield.gui
:link-type: doc

The PyQt6 widgets and controllers that make up the desktop app.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

mesofield.base
mesofield.config
mesofield.protocols
mesofield.signals
mesofield.hardware
mesofield.engines
mesofield.devices
mesofield.data
mesofield.datakit
mesofield.processing
mesofield.processors
mesofield.scaffold
mesofield.gui
mesofield.utils
```
