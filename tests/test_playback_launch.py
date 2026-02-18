from mesofield.playback.launch import discover_playback_context


def _write_dataqueue(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """queue_elapsed,packet_ts,device_ts,device_id,payload
0.0,2024-01-01T00:00:00,0.0,test_device,"0"
0.5,2024-01-01T00:00:00,0.5,test_device,"1"
"""
    )


def test_discover_playback_context_fallback(tmp_path):
    dq_path = tmp_path / "nest" / "dataqueue.csv"
    _write_dataqueue(dq_path)

    context = discover_playback_context(tmp_path)

    assert context.dataqueue_path == dq_path
    assert context.selection == ("unknown", "unknown", "unknown")
    assert context.playback.duration == 0.5
