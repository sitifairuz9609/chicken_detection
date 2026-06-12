import cv2
import os
import logging
import numpy as np
import pandas as pd
import streamlit as st
import time
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict, Counter, deque

try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"


# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(
    page_title="Realtime Chicken DOA Detection",
    layout="wide"
)

st.title("Realtime Chicken DOA Detection Dashboard")


# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Hardcoded model paths for presentation
# Put your .pt files inside the models folder using these exact names.
MODEL1_PATH = MODEL_DIR / "wing_neck_best.pt"
MODEL2_PATH = MODEL_DIR / "cloaca_best.pt"

# Hardcoded video paths for presentation
# Put your videos inside the videos folder using these exact names.
VIDEO1_PATH = VIDEO_DIR / "video1.mp4"
VIDEO2_PATH = VIDEO_DIR / "video2.mp4"

# Output paths
OUTPUT_VIDEO = OUTPUT_DIR / "realtime_output_fast.mp4"
OUTPUT_FRAME_CSV = OUTPUT_DIR / "realtime_frame_output_fast.csv"
OUTPUT_FINAL_CSV = OUTPUT_DIR / "realtime_final_output_fast.csv"


# ======================================================
# CLASS IDS
# ======================================================
V1_NECK_NOTARCHED_ID = 0
V1_NECK_ARCHED_ID = 1
V1_WING_CLASS_ID = 2
V1_CHICKEN_CLASS_ID = 3

V2_CHICKEN_CLASS_ID = 0
V2_CLOACA_CLASS_ID = 1


# ======================================================
# SETTINGS
# ======================================================
WINDOW_SECONDS = 3
MIN_FRAMES_REQUIRED = 2
NECK_HISTORY_SIZE = 10

BODY_COMPENSATION_FACTOR = 1.0

BASELINE_HISTORY_SIZE = 50
BASELINE_WARMUP_FRAMES = 20
BASELINE_PERCENTILE = 60

WING_MARGIN = 1.0
CLOACA_MARGIN = 0.8

WING_MIN_THRESHOLD = 1.5
CLOACA_MIN_THRESHOLD = 1.0

WING_CONSEC_FRAMES = 3
CLOACA_CONSEC_FRAMES = 3

FINAL_NECK_VOTE_THRESHOLD = 0.65

MATCH_ZONE_X1_RATIO = 0.15
MATCH_ZONE_X2_RATIO = 0.35
MATCH_ZONE_Y1_RATIO = 0.00
MATCH_ZONE_Y2_RATIO = 1.00

MAX_MATCH_FRAME_GAP = 60

# SPEED SETTINGS
TARGET_PROCESS_FPS = 3             # process 3 frames per second
PROCESS_EVERY_N_FRAMES = None       # calculated automatically from video FPS
SAVE_EVERY_N_FRAMES = 300           # save CSV less often
DASHBOARD_UPDATE_EVERY_N_FRAMES = 2 # update UI less aggressively
DISPLAY_SCALE = 0.35                # smaller display = faster Streamlit
REALTIME_DELAY = False              # do not sleep
YOLO_IMGSZ = 640

RESIZE_VIDEO_TO_640 = True
VIDEO_RESIZE_SIZE = 640
# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ======================================================
# HELPER FUNCTIONS
# ======================================================
def check_file(path, name):
    path = Path(path)
    if not path.exists():
        st.error(f"{name} not found: {path}")
        st.info("Expected project structure: models/wing_neck_best.pt, models/cloaca_best.pt, videos/video1.mp4, videos/video2.mp4")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    """Cache YOLO model loading so Streamlit reruns do not reload weights repeatedly."""
    return YOLO(str(model_path))


def show_download_button(path, label, mime_type):
    path = Path(path)
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime_type
        )


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def center_inside_box(part_box, chicken_box):
    cx, cy = box_center(part_box)
    x1, y1, x2, y2 = chicken_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def center_inside_roi(box, roi):
    cx, cy = box_center(box)
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


def assign_part_to_chicken(part_box, chicken_boxes):
    if len(chicken_boxes) == 0:
        return None

    inside_candidates = []

    for chicken_id, chicken_box in chicken_boxes.items():
        if center_inside_box(part_box, chicken_box):
            pcx, pcy = box_center(part_box)
            ccx, ccy = box_center(chicken_box)
            dist = ((pcx - ccx) ** 2 + (pcy - ccy) ** 2) ** 0.5
            inside_candidates.append((chicken_id, dist))

    if inside_candidates:
        inside_candidates = sorted(inside_candidates, key=lambda x: x[1])
        return inside_candidates[0][0]

    pcx, pcy = box_center(part_box)

    nearest_id = None
    nearest_dist = float("inf")

    for chicken_id, chicken_box in chicken_boxes.items():
        ccx, ccy = box_center(chicken_box)
        dist = ((pcx - ccx) ** 2 + (pcy - ccy) ** 2) ** 0.5

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_id = chicken_id

    return nearest_id


def safe_crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def compute_motion_score(frames):
    if len(frames) < 2:
        return 0.0

    scores = []
    target_size = (128, 128)

    for i in range(1, len(frames)):
        prev = cv2.resize(frames[i - 1], target_size)
        curr = cv2.resize(frames[i], target_size)

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        scores.append(float(np.mean(diff)))

    return float(np.mean(scores))


def compute_residual(part_score, body_score):
    return max(
        0.0,
        part_score - BODY_COMPENSATION_FACTOR * body_score
    )


def adaptive_consecutive_motion(
    chicken_id,
    residual,
    residual_history,
    consecutive_counter,
    margin,
    min_threshold,
    consecutive_required
):
    hist = residual_history[chicken_id]

    if len(hist) < BASELINE_WARMUP_FRAMES:
        hist.append(residual)

        baseline = float(np.percentile(hist, BASELINE_PERCENTILE)) if len(hist) > 0 else 0.0
        adaptive_threshold = max(min_threshold, baseline + margin)

        return "Unknown", baseline, adaptive_threshold, 0, False

    baseline = float(np.percentile(hist, BASELINE_PERCENTILE))
    adaptive_threshold = max(min_threshold, baseline + margin)

    raw_active = residual > adaptive_threshold

    if raw_active:
        consecutive_counter[chicken_id] += 1
    else:
        consecutive_counter[chicken_id] = 0
        hist.append(residual)

    confirmed_event = consecutive_counter[chicken_id] >= consecutive_required

    if confirmed_event:
        state = "Alive"
    else:
        state = "Dead"

    return (
        state,
        baseline,
        adaptive_threshold,
        consecutive_counter[chicken_id],
        confirmed_event
    )


def most_common_state(history):
    if len(history) == 0:
        return "Unknown"

    return Counter(history).most_common(1)[0][0]


def vote_final_status(values, vote_threshold=0.65):
    values = [v for v in values if v in ["Alive", "Dead"]]

    if len(values) == 0:
        return "Unknown", 0, 0, 0, 0.0, 0.0

    alive_count = values.count("Alive")
    dead_count = values.count("Dead")
    total = len(values)

    alive_ratio = alive_count / total
    dead_ratio = dead_count / total

    if alive_ratio >= vote_threshold:
        return "Alive", alive_count, dead_count, total, alive_ratio, dead_ratio

    if dead_ratio >= vote_threshold:
        return "Dead", alive_count, dead_count, total, alive_ratio, dead_ratio

    return "Uncertain", alive_count, dead_count, total, alive_ratio, dead_ratio


def final_motion_event_status(
    group,
    state_col,
    active_count_col,
    residual_col,
    threshold_col,
    confirmed_col,
    consecutive_required
):
    valid = group[group[state_col].isin(["Alive", "Dead"])]

    if len(valid) == 0:
        return {
            "final": "Unknown",
            "alive_count": 0,
            "dead_count": 0,
            "valid_frames": 0,
            "max_active_count": 0,
            "confirmed_event_frames": 0,
            "mean_residual": 0.0,
            "max_residual": 0.0,
            "max_threshold": 0.0
        }

    alive_count = int((valid[state_col] == "Alive").sum())
    dead_count = int((valid[state_col] == "Dead").sum())
    valid_frames = len(valid)

    max_active_count = int(valid[active_count_col].max())
    confirmed_event_frames = int(valid[confirmed_col].sum())

    mean_residual = float(valid[residual_col].mean())
    max_residual = float(valid[residual_col].max())
    max_threshold = float(valid[threshold_col].max())

    if confirmed_event_frames > 0 or max_active_count >= consecutive_required:
        final = "Alive"
    else:
        final = "Dead"

    return {
        "final": final,
        "alive_count": alive_count,
        "dead_count": dead_count,
        "valid_frames": valid_frames,
        "max_active_count": max_active_count,
        "confirmed_event_frames": confirmed_event_frames,
        "mean_residual": round(mean_residual, 4),
        "max_residual": round(max_residual, 4),
        "max_threshold": round(max_threshold, 4)
    }


def final_overall_decision(final_wing, final_cloaca, final_neck):

    statuses = [final_wing, final_cloaca, final_neck]
    alive_votes = sum(s == "Alive" for s in statuses)

    if final_neck == "Dead":
        return "Dead", alive_votes, "neck_dead_priority"

    if final_neck == "Alive":
        return "Alive", alive_votes, "neck_alive_priority"

    if final_neck == "Unknown":
        if final_wing == "Alive" and final_cloaca == "Alive":
            return "Alive", alive_votes, "neck_unknown_wing_cloaca_alive"

        if final_wing == "Dead" and final_cloaca == "Dead":
            return "Dead", alive_votes, "neck_unknown_wing_cloaca_dead"

        return "Unknown", alive_votes, "neck_unknown_mixed_motion"

    return "Unknown", alive_votes, "unhandled_case"

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


def resize_keep_ratio(frame, target_h):
    h, w = frame.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h))

def resize_to_640(frame):
    return cv2.resize(frame, (VIDEO_RESIZE_SIZE, VIDEO_RESIZE_SIZE))

def save_outputs(output_rows):
    df = pd.DataFrame(output_rows)

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df.to_csv(str(OUTPUT_FRAME_CSV), index=False)

    final_rows = []

    for chicken_id, group in df.groupby("chicken_id"):

        wing_result = final_motion_event_status(
            group=group,
            state_col="wing",
            active_count_col="wing_active_count",
            residual_col="wing_residual",
            threshold_col="wing_adaptive_threshold",
            confirmed_col="wing_confirmed_event",
            consecutive_required=WING_CONSEC_FRAMES
        )

        cloaca_result = final_motion_event_status(
            group=group,
            state_col="cloaca",
            active_count_col="cloaca_active_count",
            residual_col="cloaca_residual",
            threshold_col="cloaca_adaptive_threshold",
            confirmed_col="cloaca_confirmed_event",
            consecutive_required=CLOACA_CONSEC_FRAMES
        )

        neck_final, neck_alive, neck_dead, neck_total, neck_alive_ratio, neck_dead_ratio = vote_final_status(
            group["neck"],
            FINAL_NECK_VOTE_THRESHOLD
        )

        final_wing = wing_result["final"]
        final_cloaca = cloaca_result["final"]
        final_neck = neck_final

        overall_status, alive_vote_count, decision_reason = final_overall_decision(
            final_wing,
            final_cloaca,
            final_neck
        )

        final_rows.append({
            "chicken_id": chicken_id,

            "final_wing": final_wing,
            "final_cloaca": final_cloaca,
            "final_neck": final_neck,

            "overall_status": overall_status,
            "alive_vote_count": alive_vote_count,
            "decision_reason": decision_reason,

            "wing_alive_count": wing_result["alive_count"],
            "wing_dead_count": wing_result["dead_count"],
            "wing_valid_frames": wing_result["valid_frames"],
            "wing_max_active_count": wing_result["max_active_count"],
            "wing_confirmed_event_frames": wing_result["confirmed_event_frames"],
            "wing_mean_residual": wing_result["mean_residual"],
            "wing_max_residual": wing_result["max_residual"],
            "wing_max_threshold": wing_result["max_threshold"],

            "cloaca_alive_count": cloaca_result["alive_count"],
            "cloaca_dead_count": cloaca_result["dead_count"],
            "cloaca_valid_frames": cloaca_result["valid_frames"],
            "cloaca_max_active_count": cloaca_result["max_active_count"],
            "cloaca_confirmed_event_frames": cloaca_result["confirmed_event_frames"],
            "cloaca_mean_residual": cloaca_result["mean_residual"],
            "cloaca_max_residual": cloaca_result["max_residual"],
            "cloaca_max_threshold": cloaca_result["max_threshold"],

            "neck_alive_count": neck_alive,
            "neck_dead_count": neck_dead,
            "neck_valid_frames": neck_total,
            "neck_alive_ratio": round(neck_alive_ratio, 4),
            "neck_dead_ratio": round(neck_dead_ratio, 4),

            "first_frame": int(group["frame"].min()),
            "last_frame": int(group["frame"].max()),
            "total_rows": len(group),

            "matched_video2": bool(group["matched_video2"].any()),
            "cloaca_detected": bool(group["cloaca_detected"].any())
        })

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(str(OUTPUT_FINAL_CSV), index=False)

    return df, final_df


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Controls")

run_button = st.sidebar.button("Start Detection")
show_frame_table = st.sidebar.checkbox("Show frame-level table", value=False)
save_video = st.sidebar.checkbox("Save annotated output video", value=False)

st.sidebar.markdown("### Presentation Files")
st.sidebar.write("Model 1:", MODEL1_PATH.name)
st.sidebar.write("Model 2:", MODEL2_PATH.name)
st.sidebar.write("Video 1:", VIDEO1_PATH.name)
st.sidebar.write("Video 2:", VIDEO2_PATH.name)
st.sidebar.caption("Place these files inside the models/ and videos/ folders before running the dashboard.")

st.sidebar.markdown("### Speed Settings")
st.sidebar.write(f"Device: {DEVICE}")
st.sidebar.write(f"Target process FPS: {TARGET_PROCESS_FPS}")
st.sidebar.write(f"Save every N frames: {SAVE_EVERY_N_FRAMES}")
st.sidebar.write(f"Display scale: {DISPLAY_SCALE}")
st.sidebar.write(f"Image size: {YOLO_IMGSZ}")


# ======================================================
# DASHBOARD PLACEHOLDERS
# ======================================================
summary_placeholder = st.empty()
video_placeholder = st.empty()
live_table_placeholder = st.empty()
frame_table_placeholder = st.empty()
status_placeholder = st.empty()


# ======================================================
# MAIN
# ======================================================
if run_button:

    check_file(MODEL1_PATH, "Model 1")
    check_file(MODEL2_PATH, "Model 2")
    check_file(VIDEO1_PATH, "Video 1")
    check_file(VIDEO2_PATH, "Video 2")

    with st.spinner("Loading models..."):
        model1 = load_yolo_model(str(MODEL1_PATH))
        model2 = load_yolo_model(str(MODEL2_PATH))

    cap1 = cv2.VideoCapture(str(VIDEO1_PATH))
    cap2 = cv2.VideoCapture(str(VIDEO2_PATH))

    if not cap1.isOpened():
        st.error("Cannot open Video 1.")
        st.stop()

    if not cap2.isOpened():
        st.error("Cannot open Video 2.")
        st.stop()

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    if fps1 <= 0:
        fps1 = 30

    if fps2 <= 0:
        fps2 = fps1

    fps = min(fps1, fps2)

    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if RESIZE_VIDEO_TO_640:
        # Both videos become 640x640 before detection and drawing.
        # This keeps displayed and saved stacked video dimensions consistent.
        target_h = VIDEO_RESIZE_SIZE
        stack_w = VIDEO_RESIZE_SIZE * 2
        stack_h = VIDEO_RESIZE_SIZE
    else:
        target_h = min(h1, h2)

        dummy1 = np.zeros((h1, w1, 3), dtype=np.uint8)
        dummy2 = np.zeros((h2, w2, 3), dtype=np.uint8)

        dummy1_r = resize_keep_ratio(dummy1, target_h)
        dummy2_r = resize_keep_ratio(dummy2, target_h)

        stack_w = dummy1_r.shape[1] + dummy2_r.shape[1]
        stack_h = target_h

    if save_video:
        # The app only writes processed frames, so use the target processing FPS.
        writer_fps = TARGET_PROCESS_FPS

        out = cv2.VideoWriter(
            str(OUTPUT_VIDEO),
            cv2.VideoWriter_fourcc(*"mp4v"),
            writer_fps,
            (stack_w, stack_h)
        )

        if not out.isOpened():
            st.warning("Video writer could not be opened. CSV outputs will still be saved.")
            out = None
    else:
        out = None

    window_size = int(WINDOW_SECONDS * TARGET_PROCESS_FPS)

    # Calculate frame skipping from original video FPS.
    # Example: 30 FPS video / 3 target FPS = process every 10th frame.
    FRAME_SKIP_INTERVAL = max(1, int(round(fps / TARGET_PROCESS_FPS)))

    match_roi = (
        int(w1 * MATCH_ZONE_X1_RATIO),
        int(h1 * MATCH_ZONE_Y1_RATIO),
        int(w1 * MATCH_ZONE_X2_RATIO),
        int(h1 * MATCH_ZONE_Y2_RATIO)
    )

    v1_track_to_chicken_id = {}
    next_chicken_id = 1

    candidate_v1_chickens = {}
    matched_v1_chickens = set()

    v2_track_to_chicken_id = {}
    v2_track_first_seen = {}

    wing_buffers = defaultdict(list)
    cloaca_buffers = defaultdict(list)
    neck_history = defaultdict(list)

    v1_body_buffers = defaultdict(list)
    v2_body_buffers = defaultdict(list)

    wing_residual_history = defaultdict(lambda: deque(maxlen=BASELINE_HISTORY_SIZE))
    cloaca_residual_history = defaultdict(lambda: deque(maxlen=BASELINE_HISTORY_SIZE))

    wing_consecutive_active = defaultdict(int)
    cloaca_consecutive_active = defaultdict(int)

    output_rows = []
    latest_status = {}

    frame_idx = 0
    processed_count = 0

    start_time = time.time()

    while True:

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Skip frames based on target processing FPS.
        # Example: if original video is 30 FPS and TARGET_PROCESS_FPS = 3,
        # this processes frame 0, 10, 20, 30...
        if frame_idx % FRAME_SKIP_INTERVAL != 0:
            frame_idx += 1
            continue

        processed_count += 1
        timestamp = round(frame_idx / fps, 2)

        # Resize before detection, tracking, matching, drawing, and motion calculation.
        # This keeps YOLO coordinates, match zone, and dashboard display consistent.
        if RESIZE_VIDEO_TO_640:
            frame1 = resize_to_640(frame1)
            frame2 = resize_to_640(frame2)

        annotated1 = frame1.copy()
        annotated2 = frame2.copy()

        # Recalculate match zone from the actual current frame size.
        # This prevents the match zone from disappearing after resizing.
        h1_current, w1_current = frame1.shape[:2]
        match_roi = (
            int(w1_current * MATCH_ZONE_X1_RATIO),
            int(h1_current * MATCH_ZONE_Y1_RATIO),
            int(w1_current * MATCH_ZONE_X2_RATIO),
            int(h1_current * MATCH_ZONE_Y2_RATIO)
        )

        # ==================================================
        # MODEL 1
        # ==================================================
        res1 = model1.track(
            frame1,
            persist=True,
            verbose=False,
            device=DEVICE,
            imgsz=YOLO_IMGSZ
        )[0]

        v1_chicken_boxes = {}
        wing_boxes = defaultdict(list)
        neck_candidates = defaultdict(list)

        if res1.boxes is not None and len(res1.boxes) > 0:

            boxes1 = res1.boxes.xyxy.cpu().numpy()
            cls1 = res1.boxes.cls.cpu().numpy()
            conf1 = res1.boxes.conf.cpu().numpy()

            if res1.boxes.id is not None:
                ids1 = res1.boxes.id.cpu().numpy().astype(int)
            else:
                ids1 = np.array([-1] * len(boxes1))

            temp_v1_chickens = []

            for box, cls_id, conf, track_id in zip(boxes1, cls1, conf1, ids1):

                if int(cls_id) == V1_CHICKEN_CLASS_ID:

                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2

                    temp_v1_chickens.append({
                        "box": box,
                        "track_id": int(track_id),
                        "cx": cx,
                        "conf": float(conf)
                    })

            temp_v1_chickens = sorted(temp_v1_chickens, key=lambda x: x["cx"])

            for order_idx, item in enumerate(temp_v1_chickens):

                raw_track_id = item["track_id"]

                if raw_track_id != -1:

                    if raw_track_id not in v1_track_to_chicken_id:
                        v1_track_to_chicken_id[raw_track_id] = next_chicken_id
                        next_chicken_id += 1

                    chicken_id = v1_track_to_chicken_id[raw_track_id]

                else:
                    chicken_id = order_idx + 1

                v1_chicken_boxes[chicken_id] = item["box"]

            for box, cls_id, conf in zip(boxes1, cls1, conf1):

                cls_id = int(cls_id)

                if cls_id == V1_WING_CLASS_ID:

                    assigned_id = assign_part_to_chicken(box, v1_chicken_boxes)

                    if assigned_id is not None:
                        wing_boxes[assigned_id].append(box)

                elif cls_id == V1_NECK_ARCHED_ID:

                    assigned_id = assign_part_to_chicken(box, v1_chicken_boxes)

                    if assigned_id is not None:
                        neck_candidates[assigned_id].append(("Alive", float(conf), box))

                elif cls_id == V1_NECK_NOTARCHED_ID:

                    assigned_id = assign_part_to_chicken(box, v1_chicken_boxes)

                    if assigned_id is not None:
                        neck_candidates[assigned_id].append(("Dead", float(conf), box))

        # Match zone registration
        for chicken_id, cbox in v1_chicken_boxes.items():

            if chicken_id in matched_v1_chickens:
                continue

            if center_inside_roi(cbox, match_roi):

                if chicken_id not in candidate_v1_chickens:

                    cx, cy = box_center(cbox)

                    candidate_v1_chickens[chicken_id] = {
                        "frame": frame_idx,
                        "cx": cx,
                        "cy": cy
                    }

        stale_candidates = []

        for chicken_id, info in candidate_v1_chickens.items():
            if chicken_id in matched_v1_chickens:
                continue

            if frame_idx - info["frame"] > MAX_MATCH_FRAME_GAP:
                stale_candidates.append(chicken_id)

        for chicken_id in stale_candidates:
            del candidate_v1_chickens[chicken_id]

        # ==================================================
        # MODEL 2
        # ==================================================
        res2 = model2.track(
            frame2,
            persist=True,
            verbose=False,
            device=DEVICE,
            imgsz=YOLO_IMGSZ
        )[0]

        v2_chicken_boxes = {}
        cloaca_boxes_by_v2_track = defaultdict(list)

        if res2.boxes is not None and len(res2.boxes) > 0:

            boxes2 = res2.boxes.xyxy.cpu().numpy()
            cls2 = res2.boxes.cls.cpu().numpy()
            conf2 = res2.boxes.conf.cpu().numpy()

            if res2.boxes.id is not None:
                ids2 = res2.boxes.id.cpu().numpy().astype(int)
            else:
                ids2 = np.array([-1] * len(boxes2))

            for box, cls_id, conf, track_id in zip(boxes2, cls2, conf2, ids2):

                if int(cls_id) == V2_CHICKEN_CLASS_ID:

                    raw_v2_track_id = int(track_id)

                    if raw_v2_track_id == -1:
                        continue

                    if raw_v2_track_id not in v2_track_first_seen:
                        v2_track_first_seen[raw_v2_track_id] = frame_idx

                    v2_chicken_boxes[raw_v2_track_id] = box

            for box, cls_id, conf in zip(boxes2, cls2, conf2):

                if int(cls_id) == V2_CLOACA_CLASS_ID:

                    assigned_v2_track = assign_part_to_chicken(
                        box,
                        v2_chicken_boxes
                    )

                    if assigned_v2_track is not None:
                        cloaca_boxes_by_v2_track[assigned_v2_track].append(box)

        # Match V2 chicken to V1 chicken
        unmatched_v2_tracks = [
            track_id for track_id in v2_chicken_boxes.keys()
            if track_id not in v2_track_to_chicken_id
        ]

        unmatched_v2_tracks = sorted(
            unmatched_v2_tracks,
            key=lambda track_id: v2_track_first_seen.get(track_id, frame_idx)
        )

        for raw_v2_track_id in unmatched_v2_tracks:

            v2_first_seen_frame = v2_track_first_seen.get(raw_v2_track_id, frame_idx)

            best_chicken_id = None
            best_frame_gap = float("inf")

            for chicken_id, info in candidate_v1_chickens.items():

                if chicken_id in matched_v1_chickens:
                    continue

                frame_gap = abs(v2_first_seen_frame - info["frame"])

                if frame_gap > MAX_MATCH_FRAME_GAP:
                    continue

                if frame_gap < best_frame_gap:
                    best_frame_gap = frame_gap
                    best_chicken_id = chicken_id

            if best_chicken_id is not None:

                v2_track_to_chicken_id[raw_v2_track_id] = best_chicken_id
                matched_v1_chickens.add(best_chicken_id)

        # ==================================================
        # PROCESS VIDEO 1
        # ==================================================
        current_wing_state = {}
        current_neck_state = {}

        current_wing_score = {}
        current_body_score_v1 = {}
        current_wing_residual = {}
        current_wing_baseline = {}
        current_wing_threshold = {}
        current_wing_active_count = {}
        current_wing_confirmed = {}

        for chicken_id, cbox in v1_chicken_boxes.items():

            draw_box(
                annotated1,
                cbox,
                f"ID {chicken_id} Chicken",
                (0, 255, 0)
            )

            body_crop_v1 = safe_crop(frame1, cbox)

            if body_crop_v1 is not None:
                v1_body_buffers[chicken_id].append(body_crop_v1)

                if len(v1_body_buffers[chicken_id]) > window_size:
                    v1_body_buffers[chicken_id].pop(0)

            wing_detected_this_frame = False

            for wbox in wing_boxes[chicken_id]:

                crop_w = safe_crop(frame1, wbox)

                if crop_w is not None:
                    wing_buffers[chicken_id].append(crop_w)
                    wing_detected_this_frame = True

                    if len(wing_buffers[chicken_id]) > window_size:
                        wing_buffers[chicken_id].pop(0)

                draw_box(
                    annotated1,
                    wbox,
                    f"ID {chicken_id} Wing",
                    (255, 255, 0)
                )

            if (
                wing_detected_this_frame and
                len(wing_buffers[chicken_id]) >= MIN_FRAMES_REQUIRED and
                len(v1_body_buffers[chicken_id]) >= MIN_FRAMES_REQUIRED
            ):
                wing_score = compute_motion_score(wing_buffers[chicken_id])
                body_score_v1 = compute_motion_score(v1_body_buffers[chicken_id])

                wing_residual = compute_residual(
                    wing_score,
                    body_score_v1
                )

                wing_state, wing_baseline, wing_adaptive_threshold, wing_active_count, wing_confirmed = adaptive_consecutive_motion(
                    chicken_id=chicken_id,
                    residual=wing_residual,
                    residual_history=wing_residual_history,
                    consecutive_counter=wing_consecutive_active,
                    margin=WING_MARGIN,
                    min_threshold=WING_MIN_THRESHOLD,
                    consecutive_required=WING_CONSEC_FRAMES
                )

            else:
                wing_score = 0.0
                body_score_v1 = 0.0
                wing_residual = 0.0
                wing_state = "Unknown"
                wing_baseline = 0.0
                wing_adaptive_threshold = 0.0
                wing_active_count = 0
                wing_confirmed = False

            current_wing_state[chicken_id] = wing_state
            current_wing_score[chicken_id] = wing_score
            current_body_score_v1[chicken_id] = body_score_v1
            current_wing_residual[chicken_id] = wing_residual
            current_wing_baseline[chicken_id] = wing_baseline
            current_wing_threshold[chicken_id] = wing_adaptive_threshold
            current_wing_active_count[chicken_id] = wing_active_count
            current_wing_confirmed[chicken_id] = wing_confirmed

            # Neck
            if chicken_id in neck_candidates and len(neck_candidates[chicken_id]) > 0:

                best_state, best_conf, best_box = max(
                    neck_candidates[chicken_id],
                    key=lambda x: x[1]
                )

                neck_history[chicken_id].append(best_state)

                if len(neck_history[chicken_id]) > NECK_HISTORY_SIZE:
                    neck_history[chicken_id].pop(0)

                smooth_state = most_common_state(neck_history[chicken_id])
                current_neck_state[chicken_id] = smooth_state

                draw_box(
                    annotated1,
                    best_box,
                    f"ID {chicken_id} Neck: {smooth_state}",
                    (0, 0, 255)
                )

            else:
                current_neck_state[chicken_id] = "Unknown"

        # ==================================================
        # PROCESS VIDEO 2
        # ==================================================
        current_cloaca_state = defaultdict(lambda: "Unknown")
        chicken_has_cloaca = defaultdict(bool)

        current_cloaca_score = {}
        current_body_score_v2 = {}
        current_cloaca_residual = {}
        current_cloaca_baseline = {}
        current_cloaca_threshold = {}
        current_cloaca_active_count = {}
        current_cloaca_confirmed = {}

        for raw_v2_track_id, cbox in v2_chicken_boxes.items():

            if raw_v2_track_id in v2_track_to_chicken_id:

                chicken_id = v2_track_to_chicken_id[raw_v2_track_id]

                draw_box(
                    annotated2,
                    cbox,
                    f"ID {chicken_id} V2 Chicken",
                    (0, 255, 0)
                )

                body_crop_v2 = safe_crop(frame2, cbox)

                if body_crop_v2 is not None:
                    v2_body_buffers[chicken_id].append(body_crop_v2)

                    if len(v2_body_buffers[chicken_id]) > window_size:
                        v2_body_buffers[chicken_id].pop(0)

                cloaca_detected_this_frame = False

                for cloaca_box in cloaca_boxes_by_v2_track[raw_v2_track_id]:

                    crop_c = safe_crop(frame2, cloaca_box)

                    if crop_c is not None:
                        cloaca_buffers[chicken_id].append(crop_c)
                        cloaca_detected_this_frame = True

                        if len(cloaca_buffers[chicken_id]) > window_size:
                            cloaca_buffers[chicken_id].pop(0)

                    chicken_has_cloaca[chicken_id] = True

                    draw_box(
                        annotated2,
                        cloaca_box,
                        f"ID {chicken_id} Cloaca",
                        (255, 0, 255)
                    )

                if (
                    cloaca_detected_this_frame and
                    len(cloaca_buffers[chicken_id]) >= MIN_FRAMES_REQUIRED and
                    len(v2_body_buffers[chicken_id]) >= MIN_FRAMES_REQUIRED
                ):
                    cloaca_score = compute_motion_score(cloaca_buffers[chicken_id])
                    body_score_v2 = compute_motion_score(v2_body_buffers[chicken_id])

                    cloaca_residual = compute_residual(
                        cloaca_score,
                        body_score_v2
                    )

                    cloaca_state, cloaca_baseline, cloaca_adaptive_threshold, cloaca_active_count, cloaca_confirmed = adaptive_consecutive_motion(
                        chicken_id=chicken_id,
                        residual=cloaca_residual,
                        residual_history=cloaca_residual_history,
                        consecutive_counter=cloaca_consecutive_active,
                        margin=CLOACA_MARGIN,
                        min_threshold=CLOACA_MIN_THRESHOLD,
                        consecutive_required=CLOACA_CONSEC_FRAMES
                    )

                else:
                    cloaca_score = 0.0
                    body_score_v2 = 0.0
                    cloaca_residual = 0.0
                    cloaca_state = "Unknown"
                    cloaca_baseline = 0.0
                    cloaca_adaptive_threshold = 0.0
                    cloaca_active_count = 0
                    cloaca_confirmed = False

                current_cloaca_state[chicken_id] = cloaca_state
                current_cloaca_score[chicken_id] = cloaca_score
                current_body_score_v2[chicken_id] = body_score_v2
                current_cloaca_residual[chicken_id] = cloaca_residual
                current_cloaca_baseline[chicken_id] = cloaca_baseline
                current_cloaca_threshold[chicken_id] = cloaca_adaptive_threshold
                current_cloaca_active_count[chicken_id] = cloaca_active_count
                current_cloaca_confirmed[chicken_id] = cloaca_confirmed

            else:
                draw_box(
                    annotated2,
                    cbox,
                    f"Unmatched V2 Chicken {raw_v2_track_id}",
                    (128, 128, 128)
                )

        # ==================================================
        # SAVE ROWS
        # ==================================================
        all_visible_chicken_ids = sorted(
            set(v1_chicken_boxes.keys()) |
            set(current_cloaca_state.keys())
        )

        for chicken_id in all_visible_chicken_ids:

            row = {
                "frame": frame_idx,
                "timestamp": timestamp,
                "chicken_id": chicken_id,

                "wing": current_wing_state.get(chicken_id, "Unknown"),
                "neck": current_neck_state.get(chicken_id, "Unknown"),
                "cloaca": current_cloaca_state.get(chicken_id, "Unknown"),

                "wing_score": round(current_wing_score.get(chicken_id, 0.0), 4),
                "body_score_v1": round(current_body_score_v1.get(chicken_id, 0.0), 4),
                "wing_residual": round(current_wing_residual.get(chicken_id, 0.0), 4),
                "wing_baseline": round(current_wing_baseline.get(chicken_id, 0.0), 4),
                "wing_adaptive_threshold": round(current_wing_threshold.get(chicken_id, 0.0), 4),
                "wing_active_count": current_wing_active_count.get(chicken_id, 0),
                "wing_confirmed_event": current_wing_confirmed.get(chicken_id, False),

                "cloaca_score": round(current_cloaca_score.get(chicken_id, 0.0), 4),
                "body_score_v2": round(current_body_score_v2.get(chicken_id, 0.0), 4),
                "cloaca_residual": round(current_cloaca_residual.get(chicken_id, 0.0), 4),
                "cloaca_baseline": round(current_cloaca_baseline.get(chicken_id, 0.0), 4),
                "cloaca_adaptive_threshold": round(current_cloaca_threshold.get(chicken_id, 0.0), 4),
                "cloaca_active_count": current_cloaca_active_count.get(chicken_id, 0),
                "cloaca_confirmed_event": current_cloaca_confirmed.get(chicken_id, False),

                "cloaca_detected": chicken_has_cloaca[chicken_id],
                "matched_video2": chicken_id in matched_v1_chickens
            }

            output_rows.append(row)

        # ==================================================
        # PERIODIC SAVE + LIVE STATUS
        # ==================================================
        if frame_idx % SAVE_EVERY_N_FRAMES == 0 and frame_idx > 0:
            _, final_df_live = save_outputs(output_rows)

            latest_status = {}

            if len(final_df_live) > 0:
                # Show only chickens matched in both videos.
                if "matched_video2" in final_df_live.columns:
                    final_df_live = final_df_live[final_df_live["matched_video2"] == True]

                for _, r in final_df_live.iterrows():
                    latest_status[int(r["chicken_id"])] = {
                        "chicken_id": int(r["chicken_id"]),
                        "wing": r["final_wing"],
                        "cloaca": r["final_cloaca"],
                        "neck": r["final_neck"],
                        "overall_status": r["overall_status"],
                        "reason": r["decision_reason"]
                    }

        # ==================================================
        # DRAW LABELS + STACK
        # ==================================================
        cv2.rectangle(
            annotated1,
            (match_roi[0], match_roi[1]),
            (match_roi[2], match_roi[3]),
            (0, 165, 255),
            2
        )

        cv2.putText(
            annotated1,
            "VIDEO 1: Wing + Neck",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated2,
            "VIDEO 2: Cloaca",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        annotated1_resized = resize_keep_ratio(annotated1, target_h)
        annotated2_resized = resize_keep_ratio(annotated2, target_h)

        stacked = cv2.hconcat([annotated1_resized, annotated2_resized])

        if out is not None:
            out.write(stacked)

        # ==================================================
        # DASHBOARD UPDATE WITHOUT DUPLICATION
        # ==================================================
        if processed_count % DASHBOARD_UPDATE_EVERY_N_FRAMES == 0:

            stacked_display = cv2.resize(
                stacked,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE
            )

            stacked_rgb = cv2.cvtColor(stacked_display, cv2.COLOR_BGR2RGB)

            video_placeholder.image(
                stacked_rgb,
                channels="RGB",
                use_container_width=True
            )

            live_df = pd.DataFrame(list(latest_status.values()))

            if len(live_df) > 0:
                total_chickens = len(live_df)
                alive_count = int((live_df["overall_status"] == "Alive").sum())
                dead_count = int((live_df["overall_status"] == "Dead").sum())
                unknown_count = int((live_df["overall_status"] == "Unknown").sum())

                with summary_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Matched Chickens", total_chickens)
                    col2.metric("Alive", alive_count)
                    col3.metric("Dead", dead_count)
                    col4.metric("Unknown", unknown_count)

                with live_table_placeholder.container():
                    st.subheader("Live Final Status Per Matched Chicken")
                    st.dataframe(live_df, use_container_width=True)

            if show_frame_table and len(output_rows) > 0:
                frame_df_live = pd.DataFrame(output_rows[-50:])

                with frame_table_placeholder.container():
                    st.subheader("Recent Frame-Level Records")
                    st.dataframe(frame_df_live, use_container_width=True)

            elapsed = time.time() - start_time
            actual_fps = processed_count / elapsed if elapsed > 0 else 0

            status_placeholder.info(
                f"Frame: {frame_idx} | Processed frames: {processed_count} | Frame skip: {FRAME_SKIP_INTERVAL} | Approx processing FPS: {actual_fps:.2f}"
            )

        frame_idx += 1

        if REALTIME_DELAY:
            time.sleep(1 / fps)

    # ==================================================
    # FINISH
    # ==================================================
    cap1.release()
    cap2.release()

    if out is not None:
        out.release()

    frame_df, final_df = save_outputs(output_rows)

    st.success("Detection completed.")

    st.subheader("Final One-Row-Per-Chicken Result")
    st.dataframe(final_df, use_container_width=True)

    st.write("Saved frame CSV:", OUTPUT_FRAME_CSV)
    st.write("Saved final CSV:", OUTPUT_FINAL_CSV)

    if save_video:
        st.write("Saved annotated video:", OUTPUT_VIDEO)

    st.markdown("### Download Results")

    show_download_button(
        OUTPUT_FRAME_CSV,
        "Download Frame-Level CSV",
        "text/csv"
    )

    show_download_button(
        OUTPUT_FINAL_CSV,
        "Download Final Result CSV",
        "text/csv"
    )

    if save_video:
        show_download_button(
            OUTPUT_VIDEO,
            "Download Annotated Video",
            "video/mp4"
        )

else:
    st.info("Click Start Detection from the sidebar.")
