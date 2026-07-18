#!/usr/bin/env python3
"""
Live object detection on Raspberry Pi 5 CPU
Model : yolov8n.onnx  (full 80-class COCO, 640x640, decoded output [1,84,8400])
Camera: Raspberry Pi Camera Module 2 (Picamera2)

Counts frames against wall-clock time. Saves the average FPS either when
20 minutes elapse OR when you stop (q / Ctrl+C), whichever comes first,
using everything counted up to that moment.

Run:   python3 detect_cpu.py        Quit: q in window, or Ctrl+C
"""

import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

MODEL_PATH  = "yolov8n.onnx"
INPUT_SIZE  = 640
CONF_THRES  = 0.25
IOU_THRES   = 0.45
NUM_CLASSES = 80
CAM_SIZE    = (1280, 720)

WINDOW_SECONDS = 20 * 60
_TAG        = os.environ.get("BENCH_TAG", "solo")
RESULT_FILE = f"avg_fps_cpu_yolo_{_TAG}.txt"

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


def letterbox(img, new_shape=640, color=(114, 114, 114)):
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


def preprocess(frame):
    img, r, (dw, dh) = letterbox(frame, INPUT_SIZE)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return np.ascontiguousarray(img), r, dw, dh


def postprocess(output, r, dw, dh):
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


def save_avg(window_frames, elapsed):
    avg = window_frames / elapsed if elapsed > 0 else 0.0
    with open(RESULT_FILE, "w") as f:
        f.write(f"avg_fps_first_20min={avg:.3f}\n")
        f.write(f"total_frames={window_frames}\n")
        f.write(f"window_seconds={elapsed:.2f}\n")
    print(f"\n[detect_cpu] >>> avg = {avg:.2f} FPS over {elapsed:.1f}s "
          f"saved to {RESULT_FILE}")


def main():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAM_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    print("Pi 5 CPU live detection. Press q or Ctrl+C to stop.")

    fps = 0.0
    alpha = 0.9
    prev = time.time()

    start = time.time()
    window_frames = 0
    saved = False

    try:
        while True:
            frame = picam2.capture_array()
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            blob, r, dw, dh = preprocess(frame)
            outputs = session.run(None, {input_name: blob})
            boxes, confs, class_ids = postprocess(outputs[0], r, dw, dh)

            for box, conf, cid in zip(boxes, confs, class_ids):
                x, y, w, h = box.astype(int)
                label = f"{COCO_NAMES[cid]} {conf:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, max(y - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps = inst if fps == 0 else alpha * fps + (1 - alpha) * inst
            cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if not saved:
                window_frames += 1
                if now - start >= WINDOW_SECONDS:
                    save_avg(window_frames, now - start)
                    saved = True

            cv2.imshow("YOLOv8n CPU", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if not saved:                       # stopped early -> save what we have
                    save_avg(window_frames, time.time() - start)
                    saved = True
                break
    except KeyboardInterrupt:
        if not saved:                               # stopped early -> save what we have
            save_avg(window_frames, time.time() - start)
            saved = True
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
