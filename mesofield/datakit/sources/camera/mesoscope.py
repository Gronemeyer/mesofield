"""Mesoscope camera metadata data source."""

from mesofield.datakit.sources.camera.metadata_json import MetadataJSON


class MesoMetadataSource(MetadataJSON):
    """Load mesoscope camera metadata JSON files as a table."""
    tag = "meso_metadata"
    patterns = ("**/*_mesoscope.ome.tiff_frame_metadata.json",
                "**/*_meso.ome.tiff_frame_metadata.json")
    camera_tag = "meso_metadata"
    flatten_payload = False
    drop_columns = (
        "camera_metadata",
        "property_values",
        "version",
        "format",
        "ROI-X-start",
        "ROI-Y-start",
        "mda_event",
        "Height",
        "Width",
        "camera_device",
        "pixel_size_um",
        "images_remaining_in_buffer",
        "PixelType",
        "hardware_triggered",
        # Redundant timestamp columns: time_elapsed_s + time_absolute already
        # cover the same clock at smaller dtypes.
        "ElapsedTime-ms",
        "TimeReceivedByCore",
    )
