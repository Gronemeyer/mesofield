"""Pupil camera metadata data source."""

from mesofield.datakit.sources.camera.metadata_json import MetadataJSON


class PupilMetadataSource(MetadataJSON):
    """Load pupil camera metadata JSON files as a table."""
    tag = "pupil_metadata"
    patterns = ("**/*_pupil.mp4_frame_metadata.json",
                "**/*_pupil.ome.tiff_frame_metadata.json")
    camera_tag = "pupil_metadata"
    flatten_payload = False
    drop_columns = (
        "camera_metadata",
        "property_values",
        "version",
        "format",
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
    allow_fallback_entry_key = True