import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

from ultralytics import YOLO
import mediapipe as mp


# ----------------------------
# Download Face Landmarker Task model (if missing)
# ----------------------------
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

def ensure_face_landmarker_model(model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 1000:
        return
    print(f"Downloading face_landmarker.task -> {model_path}")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)


# ----------------------------
# Text logging
# ----------------------------
def append_text_log(
    filepath: Path,
    student_name: str,
    student_id: str,
    period: str,
    start_dt: datetime,
    end_dt: datetime,
    totals: dict,
):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    total = sum(float(v) for v in totals.values())
    att = float(totals.get("attentive", 0.0))
    slp = float(totals.get("sleeping", 0.0))
    phn = float(totals.get("phone", 0.0))
    unk = float(totals.get("unknown", 0.0))

    def pct(x):
        return (x / total * 100.0) if total > 0 else 0.0

    block = []
    block.append("=" * 30)
    block.append(f"Student: {student_name} | ID: {student_id} | Period: {period}")
    block.append(f"Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    block.append(f"End:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    block.append(f"Total: {total:.2f}s")
    block.append(f"  attentive: {att:.2f}s ({pct(att):.1f}%)")
    block.append(f"  sleeping : {slp:.2f}s ({pct(slp):.1f}%)")
    block.append(f"  phone    : {phn:.2f}s ({pct(phn):.1f}%)")
    block.append(f"  unknown  : {unk:.2f}s ({pct(unk):.1f}%)")
    block.append("=" * 30)
    block.append("")

    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n".join(block))


# ----------------------------
# Load your classifier from artifacts
# ----------------------------
def load_classifier(ckpt_path: Path, labels_path: Path, device: torch.device):
    with open(labels_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)

    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "resnet18")

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, idx_to_class


def fmt_seconds(s: float) -> str:
    s = max(0.0, float(s))
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"


# ----------------------------
# YOLO phone detection
# ----------------------------
def yolo_detect_phone(yolo_model, frame_bgr, conf=0.30):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(frame_rgb, verbose=False, conf=conf)
    r = results[0]
    names = r.names

    if r.boxes is None or len(r.boxes) == 0:
        return False

    for b in r.boxes:
        cls_id = int(b.cls.item())
        name = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, "")
        if str(name).lower() in {"cell phone", "phone", "mobile phone"}:
            return True
    return False


# ----------------------------
# MediaPipe Tasks FaceLandmarker (VIDEO mode) + blink score
# ----------------------------
def make_face_landmarker(task_path: Path):
    BaseOptions = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    RunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(options)


def get_blink_score(face_result):
    """
    Returns (blink_avg, left, right) or None.
    Handles mediapipe versions where face_blendshapes[0] is:
      - object with .categories
      - or a list of Category
    """
    blends = getattr(face_result, "face_blendshapes", None)
    if not blends:
        return None

    first = blends[0]
    if hasattr(first, "categories"):
        cats = first.categories
    elif isinstance(first, list):
        cats = first
    else:
        return None

    left = right = None
    for c in cats:
        name = getattr(c, "category_name", None)
        score = getattr(c, "score", None)
        if name == "eyeBlinkLeft":
            left = float(score)
        elif name == "eyeBlinkRight":
            right = float(score)

    if left is None or right is None:
        return None
    return (left + right) / 2.0, left, right


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Your classifier files
    parser.add_argument("--ckpt", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--labels", type=str, default="artifacts/labels.json")

    # Webcam
    parser.add_argument("--camera", type=int, default=0)

    # Logging identity
    parser.add_argument("--student_name", type=str, default="Noel Karki")
    parser.add_argument("--student_id", type=str, default="STU-032")
    parser.add_argument("--period", type=str, default="Day1-Period1")
    parser.add_argument("--log_txt", type=str, default="logs/attention_log.txt")

    # Classifier smoothing
    parser.add_argument("--smooth_window", type=int, default=20)
    parser.add_argument("--min_conf", type=float, default=0.60)

    # YOLO phone
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.30)
    parser.add_argument("--yolo_every", type=int, default=2)
    parser.add_argument("--phone_vote_window", type=int, default=12)
    parser.add_argument("--phone_vote_ratio", type=float, default=0.7)

    # Sleep detection via blink
    parser.add_argument("--blink_thresh", type=float, default=0.60, help=">= means eyes closed")
    parser.add_argument("--sleep_hold_sec", type=float, default=1.2, help="closed this long => sleeping")
    parser.add_argument("--wake_hold_sec", type=float, default=0.5, help="open this long => awake")
    parser.add_argument("--sleep_every", type=int, default=2)

    # Guards
    parser.add_argument("--dark_thresh", type=float, default=15.0)
    parser.add_argument("--face_task_path", type=str, default="models/face_landmarker.task")

    args = parser.parse_args()

    session_start = datetime.now()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Classifier device:", device)

    classifier, idx_to_class = load_classifier(Path(args.ckpt), Path(args.labels), device)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    print("Classifier classes:", class_names)

    yolo = YOLO(args.yolo_model)

    # FaceLandmarker tasks model
    task_path = Path(args.face_task_path)
    ensure_face_landmarker_model(task_path)
    landmarker = make_face_landmarker(task_path)

    # Timers
    totals = {name: 0.0 for name in class_names}
    totals["phone"] = 0.0
    totals["sleeping"] = 0.0
    totals["unknown"] = 0.0

    # Preprocess for classifier
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera 1")

    prob_history = deque(maxlen=args.smooth_window)
    phone_hist = deque(maxlen=args.phone_vote_window)

    start_perf = time.perf_counter()
    last_t = start_perf
    frame_i = 0
    last_phone = False

    # Sleep state machine
    closed_time = 0.0
    open_time = 0.0
    sleeping_state = False
    last_blink = 0.0

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        frame_i += 1

        # Dark/lens-cap guard
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < args.dark_thresh:
            label, conf = "unknown", 0.0
            totals["unknown"] += dt
            prob_history.clear()
            phone_hist.clear()
            closed_time = open_time = 0.0
            sleeping_state = False
        else:
            # 1) Phone (YOLO) – run every N frames + vote
            if frame_i % args.yolo_every == 0:
                last_phone = yolo_detect_phone(yolo, frame, conf=args.yolo_conf)
            phone_hist.append(bool(last_phone))
            phone_ratio = sum(phone_hist) / len(phone_hist)

            phone_present = phone_ratio >= args.phone_vote_ratio
            if phone_present:
                totals["phone"] += dt
                label, conf = "phone", phone_ratio
                prob_history.clear()
                closed_time = open_time = 0.0
                sleeping_state = False
            else:
                # 2) Sleep detection (FaceLandmarker) – run every N frames
                if frame_i % args.sleep_every == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms = int((now - start_perf) * 1000)

                    result = landmarker.detect_for_video(mp_image, ts_ms)
                    blink = get_blink_score(result)

                    if blink is None:
                        closed_time = max(0.0, closed_time - dt)
                        open_time = max(0.0, open_time - dt)
                    else:
                        blink_avg, bl, br = blink
                        last_blink = blink_avg
                        eyes_closed = blink_avg >= args.blink_thresh

                        if eyes_closed:
                            closed_time += dt * args.sleep_every
                            open_time = 0.0
                        else:
                            open_time += dt * args.sleep_every
                            closed_time = 0.0

                        # hysteresis
                        if (not sleeping_state) and (closed_time >= args.sleep_hold_sec):
                            sleeping_state = True
                        if sleeping_state and (open_time >= args.wake_hold_sec):
                            sleeping_state = False

                if sleeping_state:
                    totals["sleeping"] += dt
                    label, conf = "sleeping", min(1.0, closed_time / max(0.01, args.sleep_hold_sec))
                    prob_history.clear()
                else:
                    # 3) Classifier fallback
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    x = preprocess(pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        probs = F.softmax(classifier(x), dim=1).squeeze(0).cpu().numpy()

                    prob_history.append(probs)
                    avg_probs = np.mean(np.stack(prob_history, axis=0), axis=0)

                    best_idx = int(np.argmax(avg_probs))
                    best_conf = float(avg_probs[best_idx])
                    predicted = class_names[best_idx]

                    if best_conf < args.min_conf:
                        totals["unknown"] += dt
                        label, conf = "unknown", best_conf
                    else:
                        totals[predicted] += dt
                        label, conf = predicted, best_conf

        # UI overlay
        overlay = frame.copy()
        cv2.putText(overlay, f"Detected: {label} (conf~{conf:.2f})", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        y = 70
        # show common timers first
        for name in ["attentive", "talking", "sleeping", "phone", "unknown"]:
            if name in totals:
                cv2.putText(overlay, f"{name:9s}: {fmt_seconds(totals[name])}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                y += 28

        # show remaining classifier classes
        for name in class_names:
            if name not in {"attentive", "talking"}:
                cv2.putText(overlay, f"{name:9s}: {fmt_seconds(totals[name])}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
                y += 24

        cv2.putText(
            overlay,
            f"blink~{last_blink:.2f} closed={closed_time:.1f}s phone_vote={sum(phone_hist)}/{len(phone_hist) if len(phone_hist) else 1}",
            (10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Phone+Sleep Timers (q to quit)", overlay)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    session_end = datetime.now()

    # Write text log
    append_text_log(
        filepath=Path(args.log_txt),
        student_name=args.student_name,
        student_id=args.student_id,
        period=args.period,
        start_dt=session_start,
        end_dt=session_end,
        totals={
            "attentive": totals.get("attentive", 0.0),
            "sleeping": totals.get("sleeping", 0.0),
            "phone": totals.get("phone", 0.0),
            "unknown": totals.get("unknown", 0.0),
        },
    )

    print(f"\n✅ Appended to {args.log_txt}")
    print("\nFinal totals:")
    for k, v in totals.items():
        print(f"  {k}: {fmt_seconds(v)}")


if __name__ == "__main__":
    main()