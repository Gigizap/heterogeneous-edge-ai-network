#!/usr/bin/env python3
"""YOLOv8n person detection — STM32MP257F-DK NPU"""

import os, sys, time
import cv2
import numpy as np
from stai_mpu import stai_mpu_network

MODEL_PATH  = "yolov8n_320_quant_pt_uf_od_coco-person-st.nb"
INPUT_SIZE  = 320
CONF_THRES  = 0.25
IOU_THRES   = 0.45
CAM_W, CAM_H = 640, 480

WINDOW_SECONDS = 20 * 60
TAG = os.environ.get("BENCH_TAG", "")
RESULT_FILE = f"avg_fps_npu_yolo_{TAG}.txt" if TAG else "avg_fps_npu_yolo.txt"

# RGB16 forces the ISP to give color (not grayscale R8)
GST_PIPELINE = (
    "libcamerasrc name=cs src::stream-role=view-finder cs.src ! "
    "video/x-raw,format=RGB16,width={},height={} ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
).format(CAM_W, CAM_H)


def preprocess(frame):
    h, w = frame.shape[:2]
    r = min(INPUT_SIZE / h, INPUT_SIZE / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (INPUT_SIZE - new_w) / 2, (INPUT_SIZE - new_h) / 2
    img = cv2.resize(frame, (new_w, new_h))
    img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                             int(round(dw - 0.1)), int(round(dw + 0.1)),
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.clip(np.round(img / 0.003921568859368563) + (-128), -128, 127).astype(np.int8)
    return np.expand_dims(img, 0), r, dw, dh


def nms(boxes, scores, iou_thres):
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


def postprocess(raw, r, dw, dh):
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
    keep_idx = nms(boxes, confs, IOU_THRES)
    if len(keep_idx) == 0:
        return [], []
    keep_idx = np.array(keep_idx)
    return boxes[keep_idx], confs[keep_idx]


def save_avg(frames, elapsed):
    avg = frames / elapsed if elapsed > 0 else 0.0
    with open(RESULT_FILE, "w") as f:
        f.write(f"avg_fps={avg:.3f}\nframes={frames}\nseconds={elapsed:.2f}\n")
    print(f"\n>>> {avg:.2f} FPS over {elapsed:.1f}s → {RESULT_FILE}")


def main():
    model = stai_mpu_network(model_path=MODEL_PATH, use_hw_acceleration=True)
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit("Cannot open camera")

    print(f"NPU detection running. Results → {RESULT_FILE}. Press q or Ctrl+C to stop.")
    fps, prev, start, n, saved = 0.0, time.time(), time.time(), 0, False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            blob, r, dw, dh = preprocess(frame)
            model.set_input(0, blob)
            model.run()
            boxes, confs = postprocess(model.get_output(0), r, dw, dh)

            for box, c in zip(boxes, confs):
                x, y, w, h = box.astype(int)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"person {c:.2f}", (x, max(y-6,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            now = time.time()
            fps = 0.9*fps + 0.1/(max(now-prev, 1e-6)) if fps else 1/(max(now-prev,1e-6))
            prev = now
            cv2.putText(frame, f"FPS:{fps:5.1f} NPU", (10,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if not saved:
                n += 1
                if now - start >= WINDOW_SECONDS:
                    save_avg(n, now - start); saved = True

            cv2.imshow("YOLOv8n NPU", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if not saved: save_avg(n, time.time()-start)
                break
    except KeyboardInterrupt:
        if not saved: save_avg(n, time.time()-start)
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()