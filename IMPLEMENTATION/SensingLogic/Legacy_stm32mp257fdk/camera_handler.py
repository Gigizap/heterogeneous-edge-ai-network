"""
camera_handler.py
Handles all camera open / frame-grab / release logic.
"""

import os
import cv2
import numpy as np

os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
os.environ.setdefault("WAYLAND_DISPLAY",  "wayland-0")

_GST_PIPELINE = (
    "libcamerasrc name=cs src::stream-role=view-finder cs.src ! "
    "video/x-raw, format=RGB16, width=640, height=480 ! "
    "tee name=t "
    "t. ! queue ! waylandsink "
    "t. ! queue ! videoconvert ! video/x-raw, format=BGR ! "
    "appsink name=appsink drop=true max-buffers=1 sync=false"
)

CAPTURE_FRAMES     = 5    # frames to sample when picking the sharpest
MIN_BRIGHTNESS     = 20.0 # frames below this are considered black/corrupt
MAX_WARMUP_FRAMES  = 60   # hard cap to avoid infinite loop (~2s at 30fps)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Try GStreamer first, fall back to V4L2 index."""
    cap = cv2.VideoCapture(_GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (GStreamer + V4L2 index {index})")

    # Drain frames until we get a non-black one (DCMIPP drops first ~20 frames)
    for i in range(MAX_WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.mean() >= MIN_BRIGHTNESS:
            print(f"[camera] ready after {i+1} warmup frames (brightness={frame.mean():.1f})", flush=True)
            return cap

    print(f"[camera] WARNING: still dim after {MAX_WARMUP_FRAMES} frames - proceeding anyway", flush=True)
    return cap


def grab_best_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Read CAPTURE_FRAMES frames and return the sharpest one as BGR.
    Raises RuntimeError if no frame could be read.
    """
    best_frame, best_sharpness = None, -1.0
    for _ in range(CAPTURE_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue
        grey      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(grey, cv2.CV_64F).var()
        if sharpness > best_sharpness:
            best_sharpness, best_frame = sharpness, frame.copy()
    if best_frame is None:
        raise RuntimeError("No frame could be read from camera")
    return best_frame


def release_camera(cap: cv2.VideoCapture):
    if cap is not None:
        cap.release()