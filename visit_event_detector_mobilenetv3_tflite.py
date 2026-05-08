#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visit_event_detector_mobilenetv3_tflite.py

Flower YOLO -> ROI motion -> MobileNetV3 TFLite insect classifier -> record.
"""

from pathlib import Path
from datetime import datetime
from collections import deque
import time
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as e:
        raise ImportError(
            "No TFLite interpreter found. Try:\n"
            "  source ~/visit_detect/ulty_imx_env/bin/activate\n"
            "  pip install tflite-runtime\n"
        ) from e


FLOWER_MODEL_PATH = "/home/zuizui0223/visit_detect/cirsium_best.pt"
INSECT3_TFLITE_PATH = "/home/zuizui0223/visit_detect/mobilenetv3_run_20260507_172636/insect3_mobilenetv3small.tflite"
LABELS_PATH = "/home/zuizui0223/visit_detect/mobilenetv3_run_20260507_172636/labels.txt"

OUTPUT_DIR = Path("/home/zuizui0223/visit_detect/events_mobilenetv3_tflite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "event_log.csv"
DEBUG_CROP_DIR = OUTPUT_DIR / "debug_crops"
DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)

CAM_SIZE = (640, 480)
SHOW_WINDOWS = False
SAVE_FPS = 20.0
MAX_RECORD_SEC = 60.0

FLOWER_DETECT_EVERY = 120
MAX_FLOWER_MISSING = 4
FLOWER_STABLE_REQUIRED = 2
ROI_IOU_UPDATE_THRES = 0.15
FLOWER_CONF_THRES = 0.65
FLOWER_EXPAND_RATIO = 0.10

MIN_FLOWER_AREA_RATIO = 0.002
MAX_FLOWER_AREA_RATIO = 0.35
MIN_FLOWER_ASPECT = 0.30
MAX_FLOWER_ASPECT = 3.50

WARMUP_FRAMES = 40
MOTION_RATIO_THRES = 0.008
MIN_MOTION_COMPONENT_AREA = 80
MIN_ROI_WIDTH = 40
MIN_ROI_HEIGHT = 40

MOG_HISTORY = 120
MOG_VAR_THRESHOLD = 40

CLASSIFIER_IMAGE_SIZE = (224, 224)
ALLOWED_INSECT_CLASSES = {"diptera", "hymenoptera", "lepidoptera"}
INSECT_CONF_THRES = 0.70
CLASSIFIER_ACTIVE_SEC = 2.5
CLASSIFY_EVERY = 10

VOTE_WINDOW = 8
VOTE_REQUIRED_HITS = 3
STOP_AFTER_NO_INSECT_LIKE_SEC = 2.0
CROP_EXPAND_RATIO = 1.00

SAVE_DEBUG_CROPS = True
SAVE_DEBUG_CROP_EVERY_SEC = 1.0


state = {
    "frame_count": 0,
    "last_flower_det": None,
    "pending_flower_det": None,
    "flower_stable_count": 0,
    "flower_missing_count": 0,
    "last_motion_time": 0.0,
    "last_motion_ratio": 0.0,
    "last_motion_boxes": 0,
    "last_insect_time": 0.0,
    "last_insect_label": "none",
    "last_insect_score": 0.0,
    "last_crop_box": None,
    "recent_votes": deque(maxlen=VOTE_WINDOW),
    "recording": False,
    "writer": None,
    "current_video_path": None,
    "record_start_time": 0.0,
    "last_debug_crop_time": 0.0,
}


def read_labels(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"labels.txt not found: {path}")
    labels = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            labels.append(parts[1])
        else:
            labels.append(parts[0])
    return np.array(labels, dtype=object)


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-6)


def expand_box(box, ratio, w, h):
    if len(box) == 5:
        x1, y1, x2, y2, _ = box
    else:
        x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(bw * ratio)
    py = int(bh * ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def draw_text(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)


def write_log(event, extra=""):
    new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "time", "event", "frame", "flower_conf",
                "motion_ratio", "motion_boxes",
                "insect_label", "insect_score", "vote_hits",
                "recording", "extra"
            ])
        flower_conf = ""
        if state["last_flower_det"] is not None:
            flower_conf = f"{state['last_flower_det'].get('conf', 0):.3f}"
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            event,
            state["frame_count"],
            flower_conf,
            f"{state['last_motion_ratio']:.5f}",
            state["last_motion_boxes"],
            state["last_insect_label"],
            f"{state['last_insect_score']:.3f}",
            sum(state["recent_votes"]),
            state["recording"],
            extra
        ])


def reset_insect_state():
    state["last_insect_label"] = "none"
    state["last_insect_score"] = 0.0
    state["last_crop_box"] = None
    state["recent_votes"].clear()


if not os.path.exists(FLOWER_MODEL_PATH):
    raise FileNotFoundError(FLOWER_MODEL_PATH)
if not os.path.exists(INSECT3_TFLITE_PATH):
    raise FileNotFoundError(INSECT3_TFLITE_PATH)

print("[INFO] loading flower YOLO:", FLOWER_MODEL_PATH)
flower_model = YOLO(FLOWER_MODEL_PATH)

print("[INFO] loading TFLite:", INSECT3_TFLITE_PATH)
interpreter = Interpreter(model_path=INSECT3_TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]
print("[INFO] TFLite input:", input_details[0]["shape"], input_dtype)

insect3_classes = read_labels(LABELS_PATH)
print("[INFO] labels:", insect3_classes)

print("[INFO] starting camera")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": CAM_SIZE, "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2.0)

frame_rgb = picam2.capture_array()
H, W = frame_rgb.shape[:2]
print(f"[INFO] camera size: {W} x {H}")

back_sub = cv2.createBackgroundSubtractorMOG2(
    history=MOG_HISTORY,
    varThreshold=MOG_VAR_THRESHOLD,
    detectShadows=False
)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def detect_flower(frame_bgr):
    results = flower_model.predict(source=frame_bgr, conf=FLOWER_CONF_THRES, save=False, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    candidates = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
        conf = float(box.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        area_ratio = (bw * bh) / float(W * H)
        aspect = bw / float(bh)
        if area_ratio < MIN_FLOWER_AREA_RATIO or area_ratio > MAX_FLOWER_AREA_RATIO:
            continue
        if aspect < MIN_FLOWER_ASPECT or aspect > MAX_FLOWER_ASPECT:
            continue
        candidates.append((conf, x1, y1, x2, y2, area_ratio, aspect))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda z: z[0])
    conf, x1, y1, x2, y2, area_ratio, aspect = candidates[0]
    rx1, ry1, rx2, ry2 = expand_box((x1, y1, x2, y2), FLOWER_EXPAND_RATIO, W, H)

    return {
        "bbox": (x1, y1, x2, y2),
        "roi": (rx1, ry1, rx2, ry2),
        "conf": conf,
        "area_ratio": area_ratio,
        "aspect": aspect,
    }


def update_flower_roi(det):
    if det is None:
        return
    if state["last_flower_det"] is None:
        state["pending_flower_det"] = det
        state["flower_stable_count"] += 1
        if state["flower_stable_count"] >= FLOWER_STABLE_REQUIRED:
            state["last_flower_det"] = state["pending_flower_det"]
            write_log("flower_acquired")
        return

    iou = box_iou(state["last_flower_det"]["bbox"], det["bbox"])
    if iou >= ROI_IOU_UPDATE_THRES:
        state["last_flower_det"] = det
        state["flower_stable_count"] = min(state["flower_stable_count"] + 1, FLOWER_STABLE_REQUIRED)
    else:
        write_log("flower_candidate_rejected", f"iou={iou:.3f}")


def calc_motion_in_roi(frame_bgr, roi_box):
    rx1, ry1, rx2, ry2 = roi_box
    if rx2 - rx1 < MIN_ROI_WIDTH or ry2 - ry1 < MIN_ROI_HEIGHT:
        return 0.0, None, []

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 0.0, None, []

    roi_h, roi_w = roi.shape[:2]
    small_w = 240
    small_h = max(1, int(roi_h * (small_w / max(1, roi_w))))
    small_h = min(240, small_h)
    small = cv2.resize(roi, (small_w, small_h))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    fg = back_sub.apply(gray)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    clean = fg.copy()
    boxes = []
    scale_x = roi_w / small_w
    scale_y = roi_h / small_h

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_MOTION_COMPONENT_AREA:
            clean[labels == i] = 0
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        gx1 = int(rx1 + x * scale_x)
        gy1 = int(ry1 + y * scale_y)
        gx2 = int(rx1 + (x + w) * scale_x)
        gy2 = int(ry1 + (y + h) * scale_y)
        gx1, gy1, gx2, gy2 = clamp_box(gx1, gy1, gx2, gy2, W, H)
        boxes.append((gx1, gy1, gx2, gy2, area))

    motion_ratio = cv2.countNonZero(clean) / float(clean.size)
    return motion_ratio, clean, boxes


def classify_crop_tflite(frame_bgr, crop_box):
    x1, y1, x2, y2 = crop_box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "none", 0.0

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, CLASSIFIER_IMAGE_SIZE)

    x = crop_rgb.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    if input_dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        if scale and scale > 0:
            x = x / scale + zero_point
        x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x = x.astype(input_dtype)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)[0]

    if output_details[0]["dtype"] == np.uint8:
        scale, zero_point = output_details[0]["quantization"]
        pred = (pred.astype("float32") - zero_point) * scale

    cls_id = int(np.argmax(pred))
    score = float(np.max(pred))
    label = str(insect3_classes[cls_id]) if cls_id < len(insect3_classes) else str(cls_id)
    return label, score


def save_debug_crop(frame_bgr, crop_box, label, score, reason):
    if not SAVE_DEBUG_CROPS:
        return
    now = time.time()
    if now - state["last_debug_crop_time"] < SAVE_DEBUG_CROP_EVERY_SEC:
        return

    x1, y1, x2, y2 = crop_box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return

    state["last_debug_crop_time"] = now
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = DEBUG_CROP_DIR / f"{ts}_{reason}_{label}_{score:.2f}.jpg"
    cv2.imwrite(str(out), crop)


def start_recording(vis_frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"insect_event_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, SAVE_FPS, (W, H))

    state["writer"] = writer
    state["recording"] = True
    state["current_video_path"] = path
    state["record_start_time"] = time.time()

    write_log("record_start", str(path))
    print(f"[START] {state['last_insect_label']} {state['last_insect_score']:.2f} -> {path}")


def stop_recording(reason):
    if state["writer"] is not None:
        state["writer"].release()
    write_log("record_stop", reason)
    print(f"[STOP] {state['current_video_path']} {reason}")

    state["writer"] = None
    state["recording"] = False
    state["current_video_path"] = None
    state["record_start_time"] = 0.0


def make_vis(frame_bgr, motion_boxes, motion_ratio, classifier_active, insect_confirmed):
    vis = frame_bgr.copy()

    if state["last_flower_det"] is not None:
        fx1, fy1, fx2, fy2 = state["last_flower_det"]["bbox"]
        rx1, ry1, rx2, ry2 = state["last_flower_det"]["roi"]
        cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        draw_text(vis, f"flower {state['last_flower_det']['conf']:.2f}", 30)

    for mb in motion_boxes:
        x1, y1, x2, y2, area = mb
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

    if state["last_crop_box"] is not None:
        x1, y1, x2, y2 = state["last_crop_box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    draw_text(vis, f"motion {motion_ratio:.4f}", 60)
    draw_text(vis, f"classifier_active {classifier_active}", 90, (0,255,255) if classifier_active else (180,180,180))
    draw_text(vis, f"insect {state['last_insect_score']:.2f} ({state['last_insect_label']})", 120, (255,0,0) if insect_confirmed else (180,180,180))
    draw_text(vis, f"votes {sum(state['recent_votes'])}/{VOTE_WINDOW}", 150)
    draw_text(vis, f"recording {state['recording']}", 180, (0,0,255) if state["recording"] else (180,180,180))
    return vis


try:
    print("[INFO] main loop started")
    print("[INFO] output:", OUTPUT_DIR)
    write_log("program_start")

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        state["frame_count"] += 1
        now = time.time()

        run_flower = state["last_flower_det"] is None or state["frame_count"] % FLOWER_DETECT_EVERY == 1
        if run_flower:
            det = detect_flower(frame_bgr)

            if det is not None:
                state["flower_missing_count"] = 0
                update_flower_roi(det)
            else:
                state["flower_missing_count"] += 1
                state["flower_stable_count"] = 0

                if state["flower_missing_count"] >= MAX_FLOWER_MISSING:
                    if state["last_flower_det"] is not None:
                        write_log("flower_lost")
                    state["last_flower_det"] = None
                    reset_insect_state()
                    if state["recording"]:
                        stop_recording("[flower lost]")

        if state["last_flower_det"] is None:
            if state["frame_count"] % 120 == 0:
                print("[INFO] flower not detected")
            if state["recording"]:
                stop_recording("[no flower]")

            if SHOW_WINDOWS:
                vis = frame_bgr.copy()
                draw_text(vis, "flower not detected", 30, (0, 0, 255))
                cv2.imshow("visit detector", vis)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            continue

        motion_ratio, motion_mask, motion_boxes = calc_motion_in_roi(frame_bgr, state["last_flower_det"]["roi"])
        state["last_motion_ratio"] = motion_ratio
        state["last_motion_boxes"] = len(motion_boxes)

        if state["frame_count"] > WARMUP_FRAMES and motion_ratio > MOTION_RATIO_THRES and len(motion_boxes) > 0:
            state["last_motion_time"] = now

        classifier_active = (now - state["last_motion_time"]) <= CLASSIFIER_ACTIVE_SEC

        if not classifier_active and not state["recording"]:
            if len(state["recent_votes"]) > 0:
                state["recent_votes"].append(0)

        insect_confirmed = False
        if classifier_active and len(motion_boxes) > 0 and state["frame_count"] % CLASSIFY_EVERY == 1:
            biggest = max(motion_boxes, key=lambda b: b[4])
            crop_box = expand_box(biggest, CROP_EXPAND_RATIO, W, H)
            state["last_crop_box"] = crop_box

            label, score = classify_crop_tflite(frame_bgr, crop_box)
            state["last_insect_label"] = label
            state["last_insect_score"] = score

            insect_like = label in ALLOWED_INSECT_CLASSES and score >= INSECT_CONF_THRES
            state["recent_votes"].append(1 if insect_like else 0)
            save_debug_crop(frame_bgr, crop_box, label, score, "hit" if insect_like else "low")

            if insect_like:
                state["last_insect_time"] = now
                write_log("insect_like", f"{label}:{score:.3f}")

        if sum(state["recent_votes"]) >= VOTE_REQUIRED_HITS:
            insect_confirmed = True
            state["last_insect_time"] = now
            if not state["recording"]:
                vis = make_vis(frame_bgr, motion_boxes, motion_ratio, classifier_active, insect_confirmed)
                start_recording(vis)

        need_visual = SHOW_WINDOWS or state["recording"]
        vis = make_vis(frame_bgr, motion_boxes, motion_ratio, classifier_active, insect_confirmed) if need_visual else frame_bgr

        if state["recording"] and state["writer"] is not None:
            state["writer"].write(vis)

            if now - state["last_insect_time"] > STOP_AFTER_NO_INSECT_LIKE_SEC:
                stop_recording("[no insect_like]")
                reset_insect_state()
            elif now - state["record_start_time"] > MAX_RECORD_SEC:
                stop_recording("[max duration]")
                reset_insect_state()

        if SHOW_WINDOWS:
            cv2.imshow("visit detector", vis)
            if motion_mask is not None:
                cv2.imshow("motion mask", motion_mask)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C received")

finally:
    if state["recording"]:
        stop_recording("[program end]")
    write_log("program_end")
    try:
        picam2.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("[INFO] finished")
