def test_raspberry(**kwargs):
    log.info("test_raspberry: liveness check")
    random_computation = sum(i * i for i in range(10000))  # Simulate some workload
    string_to_return = "raspberry test says: function executed successfully: " + str(random_computation)
    return string_to_return
#!/usr/bin/env python3
"""
benchmark_functions.py

Skill functions for the Raspberry Pi benchmark/agent harness.

- test_raspberry      : simple liveness/sanity check
- detect_object       : capture one frame, run YOLOv8n object detection on CPU,
                        return the list of detected objects
"""

import time
import logging

# Child of the "SensingLogic" logger sensing_agent configures, so every line here is
# stamped with this agent's ID (see SensingLogic/sensing_agent.py).
log = logging.getLogger(__name__)


def test_raspberry(**kwargs):
    log.info("test_raspberry: liveness check")
    random_computation = sum(i * i for i in range(10000))  # Simulate some workload
    string_to_return = "raspberry test says: function executed successfully: " + str(random_computation)
    return string_to_return


# ----------------------------------------------------------------------------
# Object detection
# ----------------------------------------------------------------------------

MODEL_PATH  = "SensingLogic/raspberrypi5_yolo_CPU/yolov8n.onnx"
INPUT_SIZE  = 640
CONF_THRES  = 0.25
IOU_THRES   = 0.45
NUM_CLASSES = 80
CAM_SIZE    = (1280, 720)

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


def _postprocess(output, r, dw, dh):
    import cv2
    import numpy as np
    preds = np.squeeze(output).T
    scores = preds[:, 4:4 + NUM_CLASSES]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    keep = confidences > CONF_THRES
    preds, class_ids, confidences = preds[keep], class_ids[keep], confidences[keep]
    if preds.shape[0] == 0:
        return [], [], []

    boxes = preds[:, :4].copy()
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / r
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / r
    boxes[:, 2] = boxes[:, 2] / r
    boxes[:, 3] = boxes[:, 3] / r

    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                            CONF_THRES, IOU_THRES)
    if len(idxs) == 0:
        return [], [], []
    idxs = np.array(idxs).flatten()
    return boxes[idxs], confidences[idxs], class_ids[idxs]


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


def detect_object(**kwargs):
    """
    Capture one frame from the Pi camera, run YOLOv8n detection on the CPU,
    and return a human-readable string of the objects detected.
    """
    log.info("detect_object: running single-frame object detection")
    import cv2

    session, input_name = _get_session()
    picam2 = _get_camera()

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    blob, r, dw, dh = _preprocess(frame)
    outputs = session.run(None, {input_name: blob})
    boxes, confs, class_ids = _postprocess(outputs[0], r, dw, dh)

    detections = []
    for conf, cid in zip(confs, class_ids):
        detections.append({"label": COCO_NAMES[int(cid)], "confidence": round(float(conf), 3)})

    if not detections:
        return "no objects detected"

    # Build a readable summary: "person (0.91), chair (0.87)"
    parts = [f"{d['label']} ({d['confidence']:.2f})" for d in detections]
    string_to_return = (
        "detected "
        + str(len(detections))
        + " object(s): "
        + ", ".join(parts)
    )
    return string_to_return

