from datetime import datetime

from mesofield.playback import DataqueuePlayback, PlaybackEvent


def _write_queue_csv(path):
    rows = [
        [0.0, datetime(2024, 1, 1, 12, 0, 0), "0.0", "enc", "0.1"],
        [0.1, datetime(2024, 1, 1, 12, 0, 0, 100000), "0.1", "enc", "0.2"],
        [0.25, datetime(2024, 1, 1, 12, 0, 0, 250000), "0.25", "other", "{'a': 1}"],
    ]
    path.write_text(
        "\n".join(
            ["queue_elapsed,packet_ts,device_ts,device_id,payload"]
            + [",".join(map(str, row)) for row in rows]
        )
    )


def test_dataqueue_playback_runs_through_events(tmp_path):
    csv_path = tmp_path / "dataqueue.csv"
    _write_queue_csv(csv_path)

    playback = DataqueuePlayback.from_csv(csv_path, speed=1000)
    seen = []
    playback.add_listener(lambda evt: seen.append((evt.device_id, evt.payload)))

    playback.start(blocking=True)

    assert seen == [("enc", 0.1), ("enc", 0.2), ("other", {"a": 1})]


def test_scrub_uses_fractional_position():
    events = [
        PlaybackEvent(elapsed=0.0, packet_ts=datetime.now(), device_ts=None, device_id="enc", payload=0),
        PlaybackEvent(elapsed=0.2, packet_ts=datetime.now(), device_ts=None, device_id="enc", payload=1),
        PlaybackEvent(elapsed=0.4, packet_ts=datetime.now(), device_ts=None, device_id="enc", payload=2),
    ]

    playback = DataqueuePlayback(events, speed=1000)
    playback.scrub(fraction=0.5)

    seen = []
    playback.add_listener(lambda evt: seen.append(evt.payload))
    playback.start(blocking=True)

    assert seen == [1, 2]


def test_device_ts_accepts_iso_datetime(tmp_path):
    csv_path = tmp_path / "dataqueue.csv"
    csv_path.write_text(
        "\n".join(
            ["queue_elapsed,packet_ts,device_ts,device_id,payload"]
            + [
                "0.0,2024-01-01T00:00:00,2025-12-05 17:09:41.742987,enc,0",
                "0.1,2024-01-01T00:00:00,2025-12-05 17:09:41.842987,enc,1",
            ]
        )
    )

    playback = DataqueuePlayback.from_csv(csv_path)
    seen = []
    playback.add_listener(lambda evt: seen.append(evt.device_ts))
    playback.start(blocking=True)

    assert len(seen) == 2
    assert all(isinstance(ts, datetime) for ts in seen)
