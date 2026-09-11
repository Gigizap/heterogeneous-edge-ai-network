#!/usr/bin/env python3
"""
people_counter.py

Meeting-room people counter for the Raspberry Pi 5, CPU preset.

The underlying model (yolov8n.onnx) is a general-purpose 80-class object
detector, same as raspberrypi5_yolo_CPU, so this skill runs full detection
and then keeps only the "person" class to produce a head count.

- detect_people_number : capture one frame, run YOLOv8n on the CPU,
                         return how many people are in the meeting room
"""

import time
import logging

# Child of the "SensingLogic" logger sensing_agent configures, so every line here is
# stamped with this agent's ID (see SensingLogic/sensing_agent.py).
log = logging.getLogger(__name__)

from pathlib import Path

MODEL_PATH   = str(Path(__file__).parent / "yolov8n.onnx")
INPUT_SIZE   = 640
CONF_THRES   = 0.25
IOU_THRES    = 0.45
NUM_CLASSES  = 80
PERSON_CLASS = 0  # "person" is index 0 in the COCO class list
CAM_SIZE     = (1280, 720)
SAMPLE_SECS  = 1.5   # sampling window; the reply is the most frequent per-frame count

# Lazy singletons so the model/camera are only loaded once across calls
_SESSION = None
_INPUT_NAME = None
_PICAM2 = None


def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    import cv2
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _preprocess(frame):
    import cv2
    import numpy as np
    img, r, (dw, dh) = _letterbox(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return np.ascontiguousarray(img), r, dw, dh


def _mode(counts):
    """Most frequent count, ties broken toward the lower value."""
    return max(set(counts), key=lambda c: (counts.count(c), -c))


def _postprocess_people(output, r, dw, dh):
    """Same decode as the general object detector, filtered to the person class."""
    import cv2
    import numpy as np
    preds = np.squeeze(output).T
    scores = preds[:, 4:4 + NUM_CLASSES]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    keep = (confidences > CONF_THRES) & (class_ids == PERSON_CLASS)
    preds, confidences = preds[keep], confidences[keep]
    if preds.shape[0] == 0:
        return [], []

    boxes = preds[:, :4].copy()
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / r
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / r
    boxes[:, 2] = boxes[:, 2] / r
    boxes[:, 3] = boxes[:, 3] / r

    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                            CONF_THRES, IOU_THRES)
    if len(idxs) == 0:
        return [], []
    idxs = np.array(idxs).flatten()
    return boxes[idxs], confidences[idxs]


def _get_session():
    global _SESSION, _INPUT_NAME
    if _SESSION is None:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        _INPUT_NAME = _SESSION.get_inputs()[0].name
    return _SESSION, _INPUT_NAME


def _get_camera():
    global _PICAM2
    if _PICAM2 is None:
        from picamera2 import Picamera2
        _PICAM2 = Picamera2()
        config = _PICAM2.create_preview_configuration(
            main={"size": CAM_SIZE, "format": "RGB888"}
        )
        _PICAM2.configure(config)
        _PICAM2.start()
        time.sleep(1.0)  # let auto-exposure settle
    return _PICAM2


def detect_people_number(**kwargs):
    """
    Sample the Pi camera for SAMPLE_SECS, run YOLOv8n object detection on the
    CPU on every frame, and return the most frequent people count. Voting over
    frames absorbs the per-frame flicker of the detector.
    """
    log.info("detect_people_number: sampling people count for %.1fs", SAMPLE_SECS)
    import cv2

    session, input_name = _get_session()
    picam2 = _get_camera()

    counts = []
    deadline = time.monotonic() + SAMPLE_SECS
    while time.monotonic() < deadline:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        blob, r, dw, dh = _preprocess(frame)
        outputs = session.run(None, {input_name: blob})
        _boxes, confs = _postprocess_people(outputs[0], r, dw, dh)
        counts.append(len(confs))

    if not counts:
        log.warning("detect_people_number: no frames captured")
        return "detection failed: camera read error"

    n = _mode(counts)
    log.debug("detect_people_number: %d frames, counts=%s -> %d", len(counts), counts, n)
    status = "room occupied" if n > 0 else "room free"
    return f"{status}: {n} people in the meeting room"
