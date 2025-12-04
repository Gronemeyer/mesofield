# MousePortal configuration templates

This folder contains a ready-to-run portal configuration (`portal_config.yaml`) and
an easier starter (`friendly-template.yaml`). The workflow is now:

1. **ExperimentConfig JSON** – enables the `mouse_portal` plugin and points to the
   unified YAML file.
2. **portal_config.yaml** – owns *all* runtime wiring: ZeroMQ endpoints, runtime
   artifact paths, scene parameters, and the full experiment sequence definition.

## Minimal ExperimentConfig entry

Only two things remain in the JSON: the plugin declaration and the path to the
unified YAML. Everything else lives in the YAML file.

```json
"Plugins": {
  "mouse_portal": {
    "enabled": true,
    "module": "mesofield.plugins.mouseportal",
    "config": {
      "unified_config_path": "C:/dev/mesofield/mesofield/subprocesses/portal_config.yaml"
    }
  }
}
```

## portal_config.yaml anatomy

- `metadata` documents the scenario and version.
- `visual_diagram` provides a quick ASCII schematic so collaborators can grok the
  corridor at a glance.
- `plugin` gathers the values that previously lived in JSON (device id, topic,
  ready attempts, ZeroMQ endpoints, runtime path, etc.).
- The remaining top-level keys are passed directly to the MousePortal renderer.
- `engine` describes the block/trial sequence and is normalised into a runtime
  spec automatically when the plugin starts.

Use `friendly-template.yaml` as a starting point for new experiments – it keeps
the structure of the demo but trims it down so you can adapt it quickly.
