"""
object_detection_pipeline.py

NPU inference for SSD-MobileNet object detection on STM32MPU X-LINUX-AI.

Public API
----------
  detect_objects_now()                     -> list[dict]  (single shot)
  stream_objects(callback, stop_event)     -> None        (continuous)
"""

import sys

import cv2
import numpy as np
import threading
from pathlib import Path

from SensingLogic.camera_handler import open_camera, grab_best_frame, release_camera

sys.path.append("/usr/local/x-linux-ai/object-detection")  # wherever ssd_mobilenet_pp.py actually lives
from ssd_mobilenet_pp import NeuralNetwork

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/usr/local/x-linux-ai/object-detection")
MDL_V2     = BASE / "models/coco_ssd_mobilenet/ssd_mobilenet_v2_fpnlite_10_256_int8_per_tensor.nb"
LABEL_FILE = BASE / "models/coco_ssd_mobilenet/labels_coco_dataset_80.txt"

# ── Tuneable ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65
IOU_THRESHOLD        = 0.45
INPUT_MEAN           = 127.5
INPUT_STD            = 127.5

# ── Model singleton ───────────────────────────────────────────────────────────
_nn: NeuralNetwork | None = None
_nn_lock = threading.Lock()

def _load_model(model_path: Path = MDL_V2) -> NeuralNetwork:
    global _nn
    with _nn_lock:
        if _nn is None:
            _nn = NeuralNetwork(
                model_file=str(model_path),
                label_file=str(LABEL_FILE),
                input_mean=INPUT_MEAN,
                input_std=INPUT_STD,
                confidence_thresh=CONFIDENCE_THRESHOLD,
                iou_threshold=IOU_THRESHOLD,
            )
    return _nn


# ── Core inference ────────────────────────────────────────────────────────────
def _run_inference(frame_bgr: np.ndarray) -> list[dict]:
    """
    Run a single inference on a BGR frame.
    Returns a list of detections, each:
      {"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}
    """
    nn = _load_model()
    nn_w, nn_h, _ = nn.get_img_size()

    h, w = frame_bgr.shape[:2]

    # Resize to model input size, convert to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(frame_rgb, (nn_w, nn_h))

    inference_time = nn.launch_inference(resized)
    locations, classes, scores = nn.get_results()

    print(
        f"[detect] frame {w}x{h}  inference={inference_time*1000:.1f}ms",
        flush=True,
    )

    if len(locations) == 0:
        return []

    results = []
    for idx in range(len(scores[0])):
        score = float(scores[0][idx])
        if score < CONFIDENCE_THRESHOLD:
            continue

        label = nn.get_label(idx, classes)

        # Normalised → pixel coords
        if nn.model_type == "ssd_mobilenet_v2":
            y1n, x1n, y2n, x2n = locations[0][idx]
        else:  # v1: [y1, x1, y2, x2] already normalised
            y1n, x1n, y2n, x2n = locations[0][idx]

        bbox = [
            int(np.clip(x1n * w, 0, w)),
            int(np.clip(y1n * h, 0, h)),
            int(np.clip(x2n * w, 0, w)),
            int(np.clip(y2n * h, 0, h)),
        ]

        results.append({
            "label":      label,
            "confidence": round(score, 4),
            "bbox":       bbox,          # [x1, y1, x2, y2] in pixels
        })

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def detect_objects_now() -> list[dict]:
    """
    Open the camera, grab the sharpest frame, run object detection, close the camera.

    Returns a list of detected objects:
      [{"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}, ...]

    Returns [] if nothing is detected above the confidence threshold.
    """
    cap = open_camera()
    try:
        frame = grab_best_frame(cap)
        return _run_inference(frame)
    finally:
        release_camera(cap)


def stream_objects(
    callback,
    stop_event: threading.Event | None = None,
    *,
    model_path: Path = MDL_V2,
) -> None:
    """
    Open the camera once and run inference continuously until stop_event is set
    (or KeyboardInterrupt).

    Each frame's results are passed to `callback(detections, frame_bgr)` where:
      - detections  list[dict]   same schema as detect_objects_now()
      - frame_bgr   np.ndarray   the raw BGR frame that was inferred on

    Usage - fire and forget in a background thread
    -----------------------------------------------
    stop = threading.Event()

    def on_frame(detections, frame):
        for d in detections:
            print(d["label"], d["confidence"])

    t = threading.Thread(target=stream_objects, args=(on_frame, stop), daemon=True)
    t.start()
    ...
    stop.set()   # stop gracefully from anywhere

    Usage - blocking (e.g. in __main__)
    ------------------------------------
    stream_objects(on_frame)          # runs until Ctrl-C
    """
    _load_model(model_path)

    cap = open_camera()
    print("[stream] camera open - starting continuous inference", flush=True)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                print("[stream] stop event received - exiting", flush=True)
                break

            try:
                frame = grab_best_frame(cap)
            except RuntimeError as e:
                print(f"[stream] frame grab failed: {e} - retrying", flush=True)
                continue

            detections = _run_inference(frame)
            callback(detections, frame)

    except KeyboardInterrupt:
        print("[stream] KeyboardInterrupt - exiting", flush=True)
    finally:
        release_camera(cap)
        print("[stream] camera released", flush=True)

def run_till_detect(reply, stop_event: threading.Event, target_object: str):
    def callback(detections, _frame):
        labels = [d["label"] for d in detections]
        if target_object in labels:
            match = next(d for d in detections if d["label"] == target_object)
            reply(f"I just spotted a {target_object} (confidence: {match['confidence']:.0%})")
            stop_event.set()  # tells stream_objects to stop the loop
    return callback  # <-- this is what stream_objects will call each frame



# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("=== Single-shot detection ===")
    hits = detect_objects_now()
    if hits:
        for h in hits:
            print(f"  {h['label']:20s}  conf={h['confidence']:.2f}  bbox={h['bbox']}")
    else:
        print("  nothing detected")

    print("\n=== Streaming (5 seconds) ===")
    stop = threading.Event()
    frame_count = [0]

    def _print_detections(detections, _frame):
        frame_count[0] += 1
        print(f"  frame {frame_count[0]:03d} → {len(detections)} object(s)")
        for d in detections:
            print(f"    {d['label']:20s}  conf={d['confidence']:.2f}  bbox={d['bbox']}")

    t = threading.Thread(
        target=stream_objects,
        args=(_print_detections, stop),
        daemon=True,
    )
    t.start()
    time.sleep(5)
    stop.set()
    t.join(timeout=3)
    print(f"  done - processed {frame_count[0]} frames")