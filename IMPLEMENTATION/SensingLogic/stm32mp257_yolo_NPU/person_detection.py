#!/usr/bin/env python3
"""
person_detection.py  -  STM32MP257F-DK NPU sensing skill

YOLOv8n person detector running on the STM32MP257F-DK NPU via the stai_mpu
runtime (hardware acceleration). Adapted from the live benchmark script into the
single-frame sensing-skill contract (same shape as stm32mp257_yolo_CPU):

  - registered in tool_config.json as command "person_detection" -> detect_people
  - detect_people(**kwargs) grabs ONE frame, runs detection, and RETURNS a
    human-readable string (NOT a live preview loop / FPS benchmark).

Camera is the on-board ISP via a GStreamer pipeline (libcamerasrc). Heavy
imports (cv2, numpy, stai_mpu) are deferred into the functions so importing this
module never fails on a machine without those libraries.
"""

import time
import logging
from pathlib import Path

# Child of the "SensingLogic" logger that sensing_agent configures, so every line
# here is stamped with this agent's ID (see SensingLogic/sensing_agent.py).
log = logging.getLogger(__name__)

# Model lives next to this file; resolved absolutely so cwd doesn't matter.
MODEL_PATH   = str(Path(__file__).parent / "yolov8n_320_quant_pt_uf_od_coco-person-st.nb")
INPUT_SIZE   = 320
CONF_THRES   = 0.25
IOU_THRES    = 0.45
CAM_W, CAM_H = 640, 480

# Fixed input quantization for the .nb model (from the board's export):
# float[0..1] -> int8 via round(x / scale) + zero_point
_IN_SCALE = 0.003921568859368563
_IN_ZP    = -128

# RGB16 forces the ISP to give color (not grayscale R8), then convert to BGR.
GST_PIPELINE = (
    "libcamerasrc name=cs src::stream-role=view-finder cs.src ! "
    "video/x-raw,format=RGB16,width={},height={} ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
).format(CAM_W, CAM_H)

# Lazy singletons so the model + camera are acquired once across calls.
_MODEL = None   # stai_mpu_network
_CAP   = None   # cv2.VideoCapture


# ----------------------------------------------------------------------------
# Pre/post-processing (from the benchmark script)
# ----------------------------------------------------------------------------

def _preprocess(frame):
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    r = min(INPUT_SIZE / h, INPUT_SIZE / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (INPUT_SIZE - new_w) / 2, (INPUT_SIZE - new_h) / 2
    img = cv2.resize(frame, (new_w, new_h))
    img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                             int(round(dw - 0.1)), int(round(dw + 0.1)),
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.clip(np.round(img / _IN_SCALE) + _IN_ZP, -128, 127).astype(np.int8)
    return np.expand_dims(img, 0), r, dw, dh


def _nms(boxes, scores, iou_thres):
    import numpy as np
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]; y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def _postprocess(raw, r, dw, dh):
    import numpy as np
    preds = np.squeeze(np.asarray(raw, dtype=np.float32))   # model outputs float 0..1
    if preds.shape[0] == 5:
        preds = preds.T
    confs = preds[:, 4]
    keep = confs > CONF_THRES
    preds, confs = preds[keep], confs[keep]
    if len(confs) == 0:
        return [], []
    boxes = preds[:, :4].copy() * INPUT_SIZE          # normalized 0..1 -> pixels
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / r
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / r
    boxes[:, 2] /= r
    boxes[:, 3] /= r
    keep_idx = _nms(boxes, confs, IOU_THRES)
    if len(keep_idx) == 0:
        return [], []
    keep_idx = np.array(keep_idx)
    return boxes[keep_idx], confs[keep_idx]


# ----------------------------------------------------------------------------
# Model + camera (lazy singletons)
# ----------------------------------------------------------------------------

def _get_model():
    global _MODEL
    if _MODEL is None:
        from stai_mpu import stai_mpu_network
        _MODEL = stai_mpu_network(model_path=MODEL_PATH, use_hw_acceleration=True)
        log.info("stai_mpu person model loaded: %s", MODEL_PATH)
    return _MODEL


def _get_camera():
    global _CAP
    if _CAP is None:
        import cv2
        cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("stm32: cannot open camera (GStreamer pipeline)")
        for _ in range(3):                # discard a few frames so the ISP settles
            cap.read()
            time.sleep(0.05)
        _CAP = cap
    return _CAP


def _shutdown():
    """Best-effort camera release on process exit."""
    global _CAP
    if _CAP is not None:
        try:
            _CAP.release()
        except Exception:
            pass
        _CAP = None


import atexit
atexit.register(_shutdown)


# ----------------------------------------------------------------------------
# Person detection skill
# ----------------------------------------------------------------------------

def detect_people(**kwargs):
    """
    Capture one frame, run YOLOv8n person detection on the STM32 NPU (stai_mpu),
    and return a human-readable string with the number of people detected.
    """
    log.info("detect_people: running single-frame person detection")
    model = _get_model()
    cap = _get_camera()

    ret, frame = cap.read()
    if not ret or frame is None:
        return "stm32 detection says: camera read failed"

    blob, r, dw, dh = _preprocess(frame)
    model.set_input(0, blob)
    model.run()
    boxes, confs = _postprocess(model.get_output(0), r, dw, dh)

    n = len(confs)
    if n == 0:
        return "detection says: no people detected now"

    scores = ", ".join(f"{float(c):.2f}" for c in confs)
    return f"detection says: detected {n} person(s) (confidence: {scores})"


# ----------------------------------------------------------------------------
# Person monitoring skill (threaded / long-running)
# ----------------------------------------------------------------------------

def await_person(reply, stop_event, poll_interval=0.0, **kwargs):
    """
    Background monitor: keep grabbing frames and running YOLOv8n person
    detection on the STM32 NPU (stai_mpu) until a person appears (or the job is
    stopped). When a person is first detected, alert the user via reply() and stop.

    Registered in tool_config.json as a threaded skill, so sensing_agent.py launches
    it as:  Thread(target=await_person, args=(reply, stop), kwargs=arguments)
    The first two positional args are therefore the reply callback and the stop
    threading.Event; any tool arguments arrive as keywords.

    Same singletons (model + camera) and pre/post-processing as detect_people -
    this just runs them in a loop instead of once.
    """
    log.info("await_person: monitor started - watching the camera for a person")
    try:
        model = _get_model()
        cap = _get_camera()
    except Exception as e:
        log.error("await_person: could not start - %s", e)
        reply(f"stm32 monitor says: could not start (setup failed: {e})")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        blob, r, dw, dh = _preprocess(frame)
        model.set_input(0, blob)
        model.run()
        _boxes, confs = _postprocess(model.get_output(0), r, dw, dh)

        if len(confs) > 0:
            best = max(float(c) for c in confs)
            log.info("await_person: person detected (%d person(s), best conf %.2f) - alerting user",
                     len(confs), best)
            reply(f"stm32 monitor says: a person just appeared! "
                  f"({len(confs)} person(s), confidence: {best:.2f})")
            stop_event.set()
            return

        if poll_interval:
            time.sleep(poll_interval)

    log.info("await_person: monitor stopped")
