#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2

from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType

HEF_PATH   = "yolov8n.hef"
IMGSZ      = 640
CONF_THRES = 0.25
CAM_SIZE   = (1280, 720)
SWAP_RB    = True

WINDOW_SECONDS = 20 * 60
_TAG        = os.environ.get("BENCH_TAG", "solo")
RESULT_FILE = f"avg_fps_hailo_yolo_{_TAG}.txt"

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

def parse_nms_by_class(raw, ow, oh):
    boxes, scores, class_ids = [], [], []
    for cls_idx in range(len(raw)):
        dets = raw[cls_idx]
        if dets is None or len(dets) == 0:
            continue
        for det in dets:
            score = float(det[4])
            if score < CONF_THRES:
                continue
            ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
            x1 = int(xmin * ow); y1 = int(ymin * oh)
            x2 = int(xmax * ow); y2 = int(ymax * oh)
            boxes.append((x1, y1, x2, y2))
            scores.append(score)
            class_ids.append(cls_idx)
    return boxes, scores, class_ids

def save_avg(window_frames, elapsed):
    avg = window_frames / elapsed if elapsed > 0 else 0.0
    with open(RESULT_FILE, "w") as f:
        f.write(f"avg_fps_first_20min={avg:.3f}\n")
        f.write(f"total_frames={window_frames}\n")
        f.write(f"window_seconds={elapsed:.2f}\n")
    print(f"\n[detect_hailo] >>> avg = {avg:.2f} FPS over {elapsed:.1f}s "
          f"saved to {RESULT_FILE}")

def main():
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(HEF_PATH)
        infer_model.set_batch_size(1)
        infer_model.input().set_format_type(FormatType.UINT8)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        with infer_model.configure() as configured_infer_model:
            bindings = configured_infer_model.create_bindings()

            input_buf = np.empty((IMGSZ, IMGSZ, 3), dtype=np.uint8)
            out_shape = infer_model.output().shape
            output_buf = np.empty(out_shape, dtype=np.float32)
            bindings.input().set_buffer(input_buf)
            bindings.output().set_buffer(output_buf)

            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": CAM_SIZE, "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()

            print("Hailo-10H live detection (async API). Press q or Ctrl+C to stop.")

            fps = 0.0
            alpha = 0.9
            prev = time.time()

            start = time.time()
            window_frames = 0
            saved = False

            try:
                while True:
                    frame = picam2.capture_array()
                    oh, ow = frame.shape[:2]

                    resized = cv2.resize(frame, (IMGSZ, IMGSZ))
                    np.copyto(input_buf, resized)

                    configured_infer_model.run([bindings], timeout=10000)
                    raw = bindings.output().get_buffer()

                    boxes, scores, class_ids = parse_nms_by_class(raw, ow, oh)

                    bgr = frame if SWAP_RB else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    for (x1, y1, x2, y2), sc, cid in zip(boxes, scores, class_ids):
                        label = f"{COCO_NAMES[cid]} {sc:.2f}"
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(bgr, label, (x1, max(y1 - 6, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    now = time.time()
                    inst = 1.0 / max(now - prev, 1e-6)
                    prev = now
                    fps = inst if fps == 0 else alpha * fps + (1 - alpha) * inst
                    cv2.putText(bgr, f"FPS: {fps:5.1f}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    if not saved:
                        window_frames += 1
                        if now - start >= WINDOW_SECONDS:
                            save_avg(window_frames, now - start)
                            saved = True

                    cv2.imshow("YOLOv8n Hailo-10H", bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        if not saved:
                            save_avg(window_frames, time.time() - start)
                            saved = True
                        break
            except KeyboardInterrupt:
                if not saved:
                    save_avg(window_frames, time.time() - start)
                    saved = True
            finally:
                picam2.stop()
                cv2.destroyAllWindows()
                cv2.waitKey(1)

if __name__ == "__main__":
    main()
