"""
face_recognition_pipeline.py

NPU inference for face detection + recognition on STM32MPU X-LINUX-AI.

Public API
----------
  add_face_now(name)       -> str
  recognize_faces_now()    -> list[dict]
"""

import cv2
import numpy as np
from pathlib import Path

from stai_mpu import stai_mpu_network
from SensingLogic.camera_handler import open_camera, grab_best_frame, release_camera

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/usr/local/x-linux-ai/face-recognition")
DB_DIR  = BASE / "database"
MDL_DET = BASE / "models/blazeface/blazeface_128x128_quant.nb"
MDL_REC = BASE / "models/facenet/facenet512_160x160_quant.nb"

# ── Quantisation params - BlazeFace ───────────────────────────────────────────
BF_IN_SCALE  = 0.007843;  BF_IN_ZP  = 127
BF_SC0_SCALE = 0.595617;  BF_SC0_ZP = 252
BF_SC1_SCALE = 114.7795;  BF_SC1_ZP = 255
BF_BX0_SCALE = 2.103926;  BF_BX0_ZP = 113
BF_BX1_SCALE = 58.178192; BF_BX1_ZP = 60

# ── Quantisation params - FaceNet ─────────────────────────────────────────────
FN_IN_SCALE  = 0.003922; FN_IN_ZP  = 0
FN_OUT_SCALE = 0.034521; FN_OUT_ZP = 131

# ── Tuneable ──────────────────────────────────────────────────────────────────
DETECTION_THRESHOLD = 0.65
NMS_IOU_THRESHOLD   = 0.3
RECO_THRESHOLD      = 0.65
FACE_CROP_PAD       = 0.25

# ── BlazeFace anchors ─────────────────────────────────────────────────────────
def _generate_anchors(input_size=128):
    anchors = []
    for stride, n in zip([8, 16], [2, 6]):
        grid = input_size // stride
        for row in range(grid):
            for col in range(grid):
                cx = (col + 0.5) / grid
                cy = (row + 0.5) / grid
                for _ in range(n):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)

ANCHORS = _generate_anchors(128)

# ── Model singletons ──────────────────────────────────────────────────────────
_det_model = None
_rec_model = None

def _load_models():
    global _det_model, _rec_model
    if _det_model is None:
        _det_model = stai_mpu_network(model_path=str(MDL_DET), use_hw_acceleration=True)
    if _rec_model is None:
        _rec_model = stai_mpu_network(model_path=str(MDL_REC), use_hw_acceleration=True)


# ── Quant helpers ─────────────────────────────────────────────────────────────
def _dequant(arr, scale, zp):
    return (arr.astype(np.float32) - zp) * scale

def _quant_uint8(arr, scale, zp):
    return np.clip(np.round(arr / scale + zp), 0, 255).astype(np.uint8)


# ── BlazeFace detection ───────────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def _nms(x1, y1, x2, y2, scores, iou_thr):
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        order = order[1:][iou < iou_thr]
    return keep

def _detect_faces(image_rgb: np.ndarray) -> list:
    h, w = image_rgb.shape[:2]

    # ── frame sanity check ────────────────────────────────────────────────────
    mean_brightness = image_rgb.mean()
    print(f"[detect] frame {w}x{h}  brightness={mean_brightness:.1f}", flush=True)
    if mean_brightness < 5.0:
        print("[detect] WARNING: frame looks black - camera may not be ready", flush=True)

    resized = cv2.resize(image_rgb, (128, 128))
    inp = _quant_uint8(resized.astype(np.float32) / 255.0, BF_IN_SCALE, BF_IN_ZP)[np.newaxis]

    _det_model.set_input(0, inp)
    _det_model.run()

    scores = np.concatenate([
        _dequant(_det_model.get_output(0), BF_SC0_SCALE, BF_SC0_ZP).reshape(-1),
        _dequant(_det_model.get_output(1), BF_SC1_SCALE, BF_SC1_ZP).reshape(-1),
    ])
    boxes = np.concatenate([
        _dequant(_det_model.get_output(2), BF_BX0_SCALE, BF_BX0_ZP).reshape(512, 16),
        _dequant(_det_model.get_output(3), BF_BX1_SCALE, BF_BX1_ZP).reshape(384, 16),
    ])
    scores = _sigmoid(scores)

    top5 = sorted(scores.tolist())[-5:][::-1]
    print(f"[detect] max_score={top5[0]:.4f}  top-5={[round(s,4) for s in top5]}  threshold={DETECTION_THRESHOLD}", flush=True)

    acx = ANCHORS[:, 0];  acy = ANCHORS[:, 1]
    cx  = acx + boxes[:, 1] / 128.0
    cy  = acy + boxes[:, 0] / 128.0
    bw  = boxes[:, 3] / 128.0  # BlazeFace encodes w/h directly, not log-space
    bh  = boxes[:, 2] / 128.0

    mask = scores >= DETECTION_THRESHOLD
    if not np.any(mask):
        return []
    cx, cy, bw, bh, scores = cx[mask], cy[mask], bw[mask], bh[mask], scores[mask]

    x1 = np.clip((cx - bw / 2) * w, 0, w).astype(int)
    y1 = np.clip((cy - bh / 2) * h, 0, h).astype(int)
    x2 = np.clip((cx + bw / 2) * w, 0, w).astype(int)
    y2 = np.clip((cy + bh / 2) * h, 0, h).astype(int)

    keep = _nms(x1, y1, x2, y2, scores, NMS_IOU_THRESHOLD)
    return [{"bbox": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
             "score": round(float(scores[i]), 4)} for i in keep]


# ── FaceNet embedding ─────────────────────────────────────────────────────────
def _get_embedding(face_crop_rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_crop_rgb, (160, 160))
    inp     = _quant_uint8(resized.astype(np.float32) / 255.0, FN_IN_SCALE, FN_IN_ZP)[np.newaxis]

    _rec_model.set_input(0, inp)
    _rec_model.run()

    emb  = _dequant(_rec_model.get_output(0), FN_OUT_SCALE, FN_OUT_ZP).flatten()
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)


# ── Crop helper ───────────────────────────────────────────────────────────────
def _crop_face(image: np.ndarray, bbox: list) -> np.ndarray:
    h, w   = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pw, ph = int((x2 - x1) * FACE_CROP_PAD), int((y2 - y1) * FACE_CROP_PAD)
    return image[max(0, y1-ph):min(h, y2+ph), max(0, x1-pw):min(w, x2+pw)]


# ── Face database ─────────────────────────────────────────────────────────────
def _db_path(name: str) -> Path:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return DB_DIR / (name.strip().replace(" ", "_") + ".npy")

def _load_db() -> dict:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return {p.stem.replace("_", " "): np.load(str(p)) for p in sorted(DB_DIR.glob("*.npy"))}

def _db_match(embedding: np.ndarray):
    """Return (name, similarity) of best match, or (None, None) if below threshold."""
    db = _load_db()
    if not db:
        return None, None
    best_name, best_sim = None, -1.0
    for name, stored in db.items():
        sim = float(np.dot(embedding, stored) /
                    (np.linalg.norm(embedding) * np.linalg.norm(stored) + 1e-8))
        if sim > best_sim:
            best_sim, best_name = sim, name
    return (best_name, round(best_sim, 4)) if best_sim >= RECO_THRESHOLD else (None, None)


# ── Public API ────────────────────────────────────────────────────────────────

def add_face_now(name: str) -> str:
    """
    Open the camera, grab the sharpest frame, detect a face and register it.

    Returns:
      "face added"                on success
      "name already registered"   if the same name is already in the DB
      "you look like '<other>'"   if the face matches someone else
      "no face detected"          if BlazeFace finds nothing
      "multiple faces detected"   if more than one face is in frame
    """
    _load_models()

    if _db_path(name).exists():
        return "name already registered"

    cap = open_camera()
    try:
        frame     = grab_best_frame(cap)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces     = _detect_faces(image_rgb)

        if not faces:
            return "no face detected"
        if len(faces) > 1:
            return f"multiple faces detected: {len(faces)}, {faces}"

        crop = _crop_face(image_rgb, faces[0]["bbox"])
        emb  = _get_embedding(crop)

        existing_name, existing_conf = _db_match(emb)
        if existing_name is not None:
            return f"you look like '{existing_name}'"

        np.save(str(_db_path(name)), emb)
        return "face added"
    finally:
        release_camera(cap)


def eliminate_face(name: str) -> str:
    """
    Remove a face from the database by name.

    Returns:
      "face removed"      on success
      "name not found"    if no entry with that name exists
    """
    p = _db_path(name)
    if not p.exists():
        return "name not found"
    p.unlink()
    return "face removed"


def list_saved_faces() -> list[str]:
    """Return a list of all registered names, alphabetically sorted."""
    return sorted(_load_db().keys())


def recognize_faces_now() -> list[dict]:
    """
    Open the camera, grab the sharpest frame, return a list of recognised faces.

    Each entry:
      {"identity": str, "confidence": float | None, "bbox": [x1, y1, x2, y2]}

    identity is "unknown" when no DB match exceeds the threshold.
    Returns [] if no face is detected.
    """
    _load_models()

    cap = open_camera()
    try:
        frame     = grab_best_frame(cap)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces     = _detect_faces(image_rgb)

        if not faces:
            return "no face detected"

        results = []
        for f in faces:
            crop = _crop_face(image_rgb, f["bbox"])
            if crop.size == 0:
                continue
            emb        = _get_embedding(crop)
            name, conf = _db_match(emb)
            results.append({
                "identity":   name if name else "unknown",
                "confidence": conf,
                "bbox":       f["bbox"],
            })
        return results
    finally:
        release_camera(cap)