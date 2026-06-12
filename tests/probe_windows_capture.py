"""Standalone Windows USB-webcam capture probe — no mesofield, no GUI.

Run on the Windows box:  python tests/probe_windows_capture.py

For each (backend, capture-FOURCC) combination it opens index 0, tries to read
10 frames, and reports how many actually arrived. The first row with frames>0
is the configuration mesofield needs to use. This isolates *capture* (live
view) from the *writer* codec, which is a separate concern.
"""

import cv2

INDEX = 0

# (label, cv2 backend constant)
BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),   # usually the most reliable for USB webcams
    ("MSMF", cv2.CAP_MSMF),     # Windows default; common source of silent no-frame
    ("ANY", cv2.CAP_ANY),
]

# Capture pixel formats to force via CAP_PROP_FOURCC. None = leave as-is.
CAP_FOURCCS = [None, "MJPG", "YUY2"]


def probe(backend_name, backend_id, cap_fourcc):
    cap = cv2.VideoCapture(INDEX, backend_id)
    if not cap.isOpened():
        return "not opened"
    try:
        if cap_fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*cap_fourcc))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got = 0
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                got += 1
        return f"opened {w}x{h}  frames={got}/10"
    finally:
        cap.release()


def main():
    print(f"OpenCV {cv2.__version__}  index={INDEX}\n")
    print(f"{'backend':<8} {'cap_fourcc':<10} result")
    print("-" * 48)
    for bname, bid in BACKENDS:
        for fcc in CAP_FOURCCS:
            try:
                res = probe(bname, bid, fcc)
            except Exception as exc:
                res = f"ERROR {exc}"
            print(f"{bname:<8} {str(fcc):<10} {res}")
        print()


if __name__ == "__main__":
    main()
