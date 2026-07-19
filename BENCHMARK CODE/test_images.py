#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("ERROR: OpenCV is required.  Install it with:  pip install opencv-python")

ROOT = Path(__file__).resolve().parent
_REPO = ROOT.parent

ONNX_MODEL_PATH = _REPO / "IMPLEMENTATION" / "SensingLogic" / "raspberrypi5_yolo_CPU" / "yolov8n.onnx"
TFLITE_MODEL_PATH = (
    _REPO / "IMPLEMENTATION" / "SensingLogic" / "stm32mp257_yolo_CPU"
    / "yolov8n_320_quant_pt_uf_od_coco-person-st.tflite"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ONNX_INPUT_SIZE = 640
ONNX_CONF_THRES = 0.25
ONNX_IOU_THRES = 0.45
ONNX_NUM_CLASSES = 80
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

TFLITE_INPUT_SIZE = 320
TFLITE_CONF_THRES = 0.25
TFLITE_IOU_THRES = 0.45

def _onnx_letterbox(img, new_shape=640, color=(114, 114, 114)):
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

def _onnx_preprocess(frame):
    img, r, (dw, dh) = _onnx_letterbox(frame, ONNX_INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return np.ascontiguousarray(img), r, dw, dh

def _onnx_postprocess(output, r, dw, dh, conf_thres):
    preds = np.squeeze(output).T
    scores = preds[:, 4:4 + ONNX_NUM_CLASSES]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    keep = confidences > conf_thres
    preds, class_ids, confidences = preds[keep], class_ids[keep], confidences[keep]
    if preds.shape[0] == 0:
        return [], [], []

    boxes = preds[:, :4].copy()
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / r
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / r
    boxes[:, 2] = boxes[:, 2] / r
    boxes[:, 3] = boxes[:, 3] / r

    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                            conf_thres, ONNX_IOU_THRES)
    if len(idxs) == 0:
        return [], [], []
    idxs = np.array(idxs).flatten()
    return boxes[idxs], confidences[idxs], class_ids[idxs]

class OnnxDetector:
    name = "onnx"

    def __init__(self, conf_thres):
        import onnxruntime as ort
        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")
        self.conf_thres = conf_thres
        self.session = ort.InferenceSession(str(ONNX_MODEL_PATH),
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        blob, r, dw, dh = _onnx_preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        boxes, confs, class_ids = _onnx_postprocess(outputs[0], r, dw, dh, self.conf_thres)
        dets = []
        for box, conf, cid in zip(boxes, confs, class_ids):
            dets.append({
                "label": COCO_NAMES[int(cid)],
                "confidence": float(conf),
                "box": [float(v) for v in box],
            })
        return dets

def _tflite_preprocess(frame, dtype, scale, zp):
    h, w = frame.shape[:2]
    r = min(TFLITE_INPUT_SIZE / h, TFLITE_INPUT_SIZE / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (TFLITE_INPUT_SIZE - new_w) / 2, (TFLITE_INPUT_SIZE - new_h) / 2
    img = cv2.resize(frame, (new_w, new_h))
    img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                             int(round(dw - 0.1)), int(round(dw + 0.1)),
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if dtype == np.uint8:
        img = np.clip(np.round(img / scale) + zp, 0, 255).astype(np.uint8)
    elif dtype == np.int8:
        img = np.clip(np.round(img / scale) + zp, -128, 127).astype(np.int8)
    return np.expand_dims(img, 0), r, dw, dh

def _tflite_nms(boxes, scores, iou_thres):
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

def _tflite_postprocess(raw, r, dw, dh, conf_thres):
    preds = np.squeeze(raw.astype(np.float32))
    if preds.shape[0] == 5:
        preds = preds.T
    confs = preds[:, 4]
    keep = confs > conf_thres
    preds, confs = preds[keep], confs[keep]
    if len(confs) == 0:
        return [], []
    boxes = preds[:, :4].copy() * TFLITE_INPUT_SIZE
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / r
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / r
    boxes[:, 2] /= r
    boxes[:, 3] /= r
    keep_idx = _tflite_nms(boxes, confs, TFLITE_IOU_THRES)
    if len(keep_idx) == 0:
        return [], []
    keep_idx = np.array(keep_idx)
    return boxes[keep_idx], confs[keep_idx]

def _load_tflite_interpreter(model_path):
    errors = []
    for loader in (
        lambda: __import__("ai_edge_litert.interpreter", fromlist=["Interpreter"]).Interpreter,
        lambda: __import__("tflite_runtime.interpreter", fromlist=["Interpreter"]).Interpreter,
        lambda: __import__("tensorflow", fromlist=["lite"]).lite.Interpreter,
    ):
        try:
            Interpreter = loader()
        except Exception as e:
            errors.append(str(e))
            continue
        return Interpreter(model_path=str(model_path))
    raise ImportError(
        "No TFLite backend available. Install one of:\n"
        "    pip install ai-edge-litert   (recommended on a laptop)\n"
        "    pip install tflite-runtime\n"
        "    pip install tensorflow\n"
        "Import errors were:\n - " + "\n - ".join(errors)
    )

class TfliteDetector:
    name = "tflite"

    def __init__(self, conf_thres):
        if not TFLITE_MODEL_PATH.exists():
            raise FileNotFoundError(f"TFLite model not found: {TFLITE_MODEL_PATH}")
        self.conf_thres = conf_thres
        self.interp = _load_tflite_interpreter(TFLITE_MODEL_PATH)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.dtype = self.inp["dtype"]
        self.scale, self.zp = self.inp["quantization"]

    def detect(self, frame):
        blob, r, dw, dh = _tflite_preprocess(frame, self.dtype, self.scale, self.zp)
        self.interp.set_tensor(self.inp["index"], blob)
        self.interp.invoke()
        boxes, confs = _tflite_postprocess(self.interp.get_tensor(self.out["index"]),
                                           r, dw, dh, self.conf_thres)
        dets = []
        for box, conf in zip(boxes, confs):
            dets.append({
                "label": "person",
                "confidence": float(conf),
                "box": [float(v) for v in box],
            })
        return dets

def _color_for(label):
    h = (hash(label) % 180)
    hsv = np.uint8([[[h, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_detections(image, dets, title):
    out = image.copy()
    H, W = out.shape[:2]
    for d in dets:
        x, y, w, h = d["box"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        color = _color_for(d["label"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 2, ty), color, -1)
        cv2.putText(out, text, (x1 + 1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W, 22), (0, 0, 0), -1)
    cv2.putText(out, f"{title}: {len(dets)} detection(s)", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def summarize(dets):
    if not dets:
        return "nothing detected"
    parts = [f"{d['label']} ({d['confidence']:.2f})" for d in dets]
    return f"{len(dets)} detection(s): " + ", ".join(parts)

def build_detectors(which, conf):
    detectors = []
    want_onnx = which in ("both", "onnx")
    want_tflite = which in ("both", "tflite")

    if want_onnx:
        try:
            detectors.append(OnnxDetector(ONNX_CONF_THRES if conf is None else conf))
            print(f"[ok] ONNX model loaded:   {ONNX_MODEL_PATH.name}")
        except Exception as e:
            print(f"[skip] ONNX model unavailable: {e}")
    if want_tflite:
        try:
            detectors.append(TfliteDetector(TFLITE_CONF_THRES if conf is None else conf))
            print(f"[ok] TFLite model loaded: {TFLITE_MODEL_PATH.name}")
        except Exception as e:
            print(f"[skip] TFLite model unavailable: {e}")
    return detectors

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="test",
                    help="folder with input images (default: ./test)")
    ap.add_argument("--out", default=None,
                    help="output folder for annotated images (default: <folder>/results)")
    ap.add_argument("--models", choices=["both", "onnx", "tflite"], default="both",
                    help="which model(s) to run (default: both)")
    ap.add_argument("--conf", type=float, default=None,
                    help="confidence threshold override (default: model's own 0.25)")
    ap.add_argument("--show", action="store_true",
                    help="also display annotated images in a window")
    args = ap.parse_args()

    in_dir = (ROOT / args.folder) if not Path(args.folder).is_absolute() else Path(args.folder)
    out_dir = Path(args.out) if args.out else (in_dir / "results")

    if not in_dir.exists():
        in_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created input folder: {in_dir}")
        print("Drop some images in there and run this script again.")
        return

    images = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {in_dir}")
        print(f"Supported extensions: {', '.join(sorted(IMAGE_EXTS))}")
        return

    detectors = build_detectors(args.models, args.conf)
    if not detectors:
        print("No model backends available - install onnxruntime and/or a TFLite runtime "
              "(see the header of this file).")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nInput : {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Images: {len(images)}\n" + "-" * 60)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"\n{img_path.name}: could not read image, skipping")
            continue
        print(f"\n{img_path.name}  ({frame.shape[1]}x{frame.shape[0]})")
        for det in detectors:
            dets = det.detect(frame)
            print(f"  [{det.name:6}] {summarize(dets)}")
            annotated = draw_detections(frame, dets, det.name.upper())
            out_path = out_dir / f"{img_path.stem}__{det.name}{img_path.suffix}"
            cv2.imwrite(str(out_path), annotated)
            if args.show:
                cv2.imshow(f"{img_path.name} [{det.name}]", annotated)

    if args.show:
        print("\nPress any key in an image window to close them all.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "-" * 60)
    print(f"Done. Annotated images written to: {out_dir}")

if __name__ == "__main__":
    main()
