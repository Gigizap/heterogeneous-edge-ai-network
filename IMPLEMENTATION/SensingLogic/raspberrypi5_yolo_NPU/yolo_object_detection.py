#!/usr/bin/env python3
"""
yolo_object_detection.py  —  Hailo-NPU sensing skill

Mirror of the CPU preset's detect_object skill (raspberrypi5_yolo_CPU), but the
inference runs on the Raspberry Pi 5 + Hailo-10H (AI HAT+ 2) NPU instead of the
ONNX CPU runtime.

Model : yolov8n.hef  (full 80-class COCO, 640x640, NMS baked into the HEF)
Camera: Raspberry Pi Camera Module (Picamera2)

Contract expected by SensingLogic/sensing_agent.py:
  - registered in tool_config.json as command "object_detection" -> detect_object
  - detect_object(**kwargs) captures ONE frame, runs detection, and RETURNS a
    human-readable string (NOT a live preview loop). The leader broadcasts the
    tool call and collects this string as the reply.

Heavy imports (hailo_platform, picamera2, cv2, numpy) are deferred into the
functions so that importing this module (or scanning the preset) never fails on
a machine without the Hailo SDK / camera.
"""

import atexit
import time
import logging
from pathlib import Path

# Child of the "SensingLogic" logger sensing_agent configures, so every line here is
# stamped with this agent's ID (see SensingLogic/sensing_agent.py).
log = logging.getLogger(__name__)

# yolov8n.hef lives next to this file; resolved absolutely so the path is correct
# regardless of the process working directory.
HEF_PATH   = str(Path(__file__).parent / "yolov8n.hef")
IMGSZ      = 640
CONF_THRES = 0.25
CAM_SIZE   = (1280, 720)

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

# Lazy singletons so the NPU model + camera are acquired once across calls.
_HAILO  = None   # tuple kept alive so nothing is garbage-collected / released early
_PICAM2 = None


# ----------------------------------------------------------------------------
# Hailo NPU + camera (lazy singletons)
# ----------------------------------------------------------------------------

def _get_hailo():
    """
    Acquire the Hailo VDevice + configured infer model once and keep it alive.

    Unlike the benchmark script (which uses `with VDevice(...) as ...`), a skill
    is called repeatedly, so we enter the context managers manually and stash
    every handle in the module-level _HAILO tuple. Returns
    (configured_infer_model, bindings, input_buf).
    """
    global _HAILO
    if _HAILO is None:
        import numpy as np
        from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        vdevice = VDevice(params)
        infer_model = vdevice.create_infer_model(HEF_PATH)
        infer_model.set_batch_size(1)
        infer_model.input().set_format_type(FormatType.UINT8)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        cm = infer_model.configure()        # context manager — keep it open
        configured = cm.__enter__()
        bindings = configured.create_bindings()

        input_buf  = np.empty((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        output_buf = np.empty(infer_model.output().shape, dtype=np.float32)
        bindings.input().set_buffer(input_buf)
        bindings.output().set_buffer(output_buf)

        _HAILO = (vdevice, infer_model, cm, configured, bindings, input_buf)
        log.info("Hailo object model loaded: %s", HEF_PATH)

    return _HAILO[3], _HAILO[4], _HAILO[5]   # configured, bindings, input_buf


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


@atexit.register
def _shutdown():
    """Best-effort release of the Hailo device + camera on process exit."""
    global _HAILO, _PICAM2
    if _HAILO is not None:
        vdevice, infer_model, cm, configured, bindings, input_buf = _HAILO
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
        try:
            vdevice.release()
        except Exception:
            pass
        _HAILO = None
    if _PICAM2 is not None:
        try:
            _PICAM2.stop()
        except Exception:
            pass
        _PICAM2 = None


# ----------------------------------------------------------------------------
# Object detection skill
# ----------------------------------------------------------------------------

def detect_object(**kwargs):
    """
    Capture one frame from the Pi camera, run YOLOv8n detection on the Hailo NPU,
    and return a human-readable string of the objects detected.

    The HEF has NMS baked in, so the output buffer is grouped by class:
    raw[class_idx] is an array of detections, each [ymin, xmin, ymax, xmax, score].
    """
    log.info("detect_object: running single-frame object detection")
    import cv2
    import numpy as np

    configured, bindings, input_buf = _get_hailo()
    picam2 = _get_camera()

    frame = picam2.capture_array()
    resized = cv2.resize(frame, (IMGSZ, IMGSZ))
    np.copyto(input_buf, resized)

    configured.run([bindings], timeout=10000)
    raw = bindings.output().get_buffer()

    detections = []
    for cls_idx in range(len(raw)):
        dets = raw[cls_idx]
        if dets is None or len(dets) == 0:
            continue
        for det in dets:
            score = float(det[4])
            if score < CONF_THRES:
                continue
            detections.append({"label": COCO_NAMES[cls_idx], "confidence": round(score, 3)})

    if not detections:
        return "raspberry detection says: no objects detected"

    parts = [f"{d['label']} ({d['confidence']:.2f})" for d in detections]
    return (
        "raspberry detection says: detected "
        + str(len(detections))
        + " object(s): "
        + ", ".join(parts)
    )
