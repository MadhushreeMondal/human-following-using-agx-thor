#!/usr/bin/env python3
"""
Persistent Human Target Following
---------------------------------
This version keeps TARGET-01 permanent until R is pressed and improves
re-identification by using MULTIPLE saved samples plus FACE + BODY appearance.

Controls:
    Left click  = select target
    R           = remove permanent target and select again
    F11         = fullscreen
    Q           = quit

Files:
    yolo11n.pt
    target_botsort.yaml
    target_profile_v3.npz
    target_lock.txt

IMPORTANT:
The BoT-SORT ID is temporary. TARGET-01 is the permanent application identity.
"""

import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "target_botsort.yaml"
CAMERA = "/dev/video0"

WIDTH = 1280
HEIGHT = 720
FPS = 30

CONFIDENCE = 0.35
PERSON_CLASS = 0

WINDOW_NAME = "Human Target Following"

# Target acquisition
CAPTURE_SECONDS = 8.0
CAPTURE_EVERY_FRAMES = 1
MIN_CAPTURE_SAMPLES = 8
MAX_GALLERY_SIZE = 40

# Re-ID
# Face score is much more important than clothing/background.
REID_THRESHOLD = 0.42
REID_MARGIN = 0.08
REID_CONFIRM_FRAMES = 5

# Once a target is visible again, use a slightly easier threshold.
VISIBLE_MATCH_THRESHOLD = 0.38

# Do NOT change the permanent profile while searching.
PERMANENT_PROFILE = True

PROFILE_PATH = "target_profile_insightface_v2.npz"
LOCK_PATH = "target_lock_insightface_v2.txt"

TARGET_IDENTITY = "TARGET-01"

# ============================================================
# COLORS
# ============================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)


# ============================================================
# GLOBAL STATE
# ============================================================

target_selected = False
target_track_id = None

target_gallery = []

capture_active = False
capture_start_time = 0.0
capture_samples = []
capture_frame_counter = 0

last_seen_time = 0.0
target_was_lost = False

reid_candidate_id = None
reid_candidate_frames = 0
reid_candidate_score = 0.0

target_score = 0.0

mouse_x = -1
mouse_y = -1
mouse_clicked = False

fullscreen = True


# ============================================================
# TEXT
# ============================================================

def draw_text(frame, text, position, color=WHITE, scale=0.65, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# BOX
# ============================================================

def clamp_box(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))

    return x1, y1, x2, y2


# ============================================================
# INSIGHTFACE MODEL
# ============================================================

# FaceAnalysis uses a real face detector + ArcFace recognition embedding.
# CPUExecutionProvider is selected for compatibility with Jetson setups
# where a matching ONNX Runtime GPU build may not be installed.
FACE_APP = None


def initialize_face_model():
    global FACE_APP
    print("[FACE] Loading InsightFace buffalo_l model...")
    FACE_APP = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
    det_model = FACE_APP.models.get("detection")
    if det_model is not None and hasattr(det_model, "det_thresh"):
        det_model.det_thresh = 0.35
    print("[FACE] InsightFace ready (CPUExecutionProvider, 640x640 detection).")


# ============================================================
# BODY COLOR FEATURE
# ============================================================

def body_feature(crop):
    crop = cv2.resize(crop, (96, 192), interpolation=cv2.INTER_AREA)

    # Ignore a small border to reduce background.
    crop = crop[8:184, 7:89]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    features = []

    # Global color histogram.
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [18, 12],
        [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_L2)
    features.extend(hist.flatten().astype(np.float32))

    # Body zones are weighted separately.
    h = hsv.shape[0]

    zones = [
        (0, int(h * 0.25)),
        (int(h * 0.20), int(h * 0.60)),
        (int(h * 0.55), h),
    ]

    for y1, y2 in zones:
        zone = hsv[y1:y2]

        hist = cv2.calcHist(
            [zone],
            [0, 1],
            None,
            [12, 10],
            [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_L2)
        features.extend(hist.flatten().astype(np.float32))

        lab = cv2.cvtColor(
            crop[y1:y2],
            cv2.COLOR_BGR2LAB
        )

        mean = np.mean(lab, axis=(0, 1)) / 255.0
        std = np.std(lab, axis=(0, 1)) / 255.0

        features.extend(mean.astype(np.float32))
        features.extend(std.astype(np.float32))

    # Simple edge distribution.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)

    edge_hist = cv2.calcHist(
        [edges],
        [0],
        None,
        [16],
        [0, 256]
    )
    cv2.normalize(edge_hist, edge_hist, 0, 1, cv2.NORM_L2)

    features.extend(edge_hist.flatten().astype(np.float32))

    out = np.asarray(features, dtype=np.float32)
    norm = np.linalg.norm(out)

    if norm < 1e-8:
        return None

    return out / norm


# ============================================================
# REAL FACE EMBEDDING (INSIGHTFACE / ARCFACE)
# ============================================================

def face_feature(person_crop):
    """Return a normalized ArcFace embedding from an enlarged upper-person crop."""
    if FACE_APP is None or person_crop is None or person_crop.size == 0:
        return None

    try:
        h, w = person_crop.shape[:2]
        if h < 35 or w < 35:
            return None

        # Enlarge small upper-body crops while preserving aspect ratio.
        scale = min(3.0, max(1.0, 480.0 / max(h, w)))
        if scale > 1.05:
            detect_crop = cv2.resize(
                person_crop,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC
            )
        else:
            detect_crop = person_crop

        detected_faces = FACE_APP.get(detect_crop)
        if not detected_faces:
            return None

        valid_faces = []
        for item in detected_faces:
            bbox = np.asarray(item.bbox, dtype=np.float32)
            fw = float(bbox[2] - bbox[0])
            fh = float(bbox[3] - bbox[1])
            det_score = float(getattr(item, "det_score", 0.0))
            if fw >= 24 and fh >= 24 and det_score >= 0.35:
                valid_faces.append((fw * fh, item))

        if not valid_faces:
            return None

        _, face = max(valid_faces, key=lambda pair: pair[0])
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            return None

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size != 512 or norm < 1e-8:
            return None

        return {
            "embedding": embedding / norm,
            "det_score": float(getattr(face, "det_score", 0.0))
        }

    except Exception as exc:
        print("[FACE] Detection/embedding error:", exc)
        return None


# ============================================================
# COMPLETE PERSON DESCRIPTOR
# ============================================================

def get_person_descriptor(frame, box):
    x1, y1, x2, y2 = clamp_box(frame, box)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 100 or w < 45:
        return None

    normalized_crop = cv2.resize(
        crop,
        (192, 384),
        interpolation=cv2.INTER_AREA
    )

    body = body_feature(normalized_crop)

    # Face detection is run on the upper 55% of the person crop.
    # This limits false detections from posters/background objects.
    upper = crop[:max(1, int(crop.shape[0] * 0.45))]
    face_data = face_feature(upper)

    return {
        "body": body,
        "face": face_data,
    }


# ============================================================
# FRAME-BASED DESCRIPTOR CACHE
# ============================================================

DESCRIPTOR_REFRESH_FRAMES = 4


def get_cached_person_descriptor(frame, person, frame_index, cache):
    """
    Reuse a track descriptor between refreshes.
    Returns (descriptor, refreshed_now) so enrollment and re-ID confirmation
    can distinguish a genuinely new face observation from a cached copy.
    """
    track_id = person["id"]
    cached = cache.get(track_id)

    if (
        cached is None
        or frame_index - cached["frame_index"] >= DESCRIPTOR_REFRESH_FRAMES
    ):
        descriptor = get_person_descriptor(frame, person["box"])
        cache[track_id] = {
            "frame_index": frame_index,
            "descriptor": descriptor
        }
        return descriptor, True

    return cached["descriptor"], False


# ============================================================
# COSINE
# ============================================================

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0

    if a.shape != b.shape:
        return 0.0

    denom = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denom < 1e-8:
        return 0.0

    return float(
        np.clip(
            np.dot(a, b) / denom,
            0.0,
            1.0
        )
    )


# ============================================================
# INSIGHTFACE / ARCFACE SIMILARITY
# ============================================================

def face_similarity(face_a, face_b):
    if face_a is None or face_b is None:
        return None

    emb_a = face_a.get("embedding")
    emb_b = face_b.get("embedding")

    if emb_a is None or emb_b is None:
        return None

    return cosine_similarity(
        np.asarray(emb_a, dtype=np.float32),
        np.asarray(emb_b, dtype=np.float32)
    )


# ============================================================
# DESCRIPTOR SIMILARITY
# ============================================================

def descriptor_similarity(a, b):
    if a is None or b is None:
        return 0.0

    body_score = cosine_similarity(
        a.get("body"),
        b.get("body")
    )

    face_score = face_similarity(
        a.get("face"),
        b.get("face")
    )

    # Actual ArcFace identity match is primary; clothing/body is only
    # a small supporting signal. Without embeddings, body-only scores
    # are capped so clothing similarity cannot independently confirm
    # a permanent identity.
    if face_score is not None:
        score = 0.85 * face_score + 0.15 * body_score
    else:
        score = 0.30 * body_score

    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# GALLERY SCORE
# ============================================================

def gallery_similarity(descriptor):
    if descriptor is None or not target_gallery:
        return 0.0

    scores = [
        descriptor_similarity(
            sample,
            descriptor
        )
        for sample in target_gallery
    ]

    # Top few samples are averaged instead of trusting one sample.
    scores.sort(reverse=True)

    if not scores:
        return 0.0

    if len(scores) >= 3:
        return float(
            0.60 * scores[0] +
            0.25 * scores[1] +
            0.15 * scores[2]
        )

    return float(scores[0])


# ============================================================
# ADD SAMPLE
# ============================================================

def add_target_sample(descriptor):
    global target_gallery

    if descriptor is None:
        return False

    # Do not add a near-duplicate.
    if target_gallery:
        best = max(
            descriptor_similarity(
                x,
                descriptor
            )
            for x in target_gallery
        )

        if best > 0.985:
            return False

    target_gallery.append(descriptor)

    if len(target_gallery) > MAX_GALLERY_SIZE:
        target_gallery.pop(0)

    return True


# ============================================================
# SAVE PROFILE
# ============================================================

def save_target_profile():
    if not target_gallery:
        return False

    try:
        body_gallery = []
        face_embeddings = []
        face_present = []

        embedding_size = 512

        for sample in target_gallery:
            body_gallery.append(sample["body"])
            face = sample.get("face")
            embedding = None if face is None else face.get("embedding")

            if embedding is None:
                face_present.append(False)
                face_embeddings.append(
                    np.zeros((embedding_size,), dtype=np.float32)
                )
            else:
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if embedding.size != embedding_size:
                    raise ValueError(
                        f"Unexpected InsightFace embedding size: {embedding.size}"
                    )
                face_present.append(True)
                face_embeddings.append(embedding)

        np.savez_compressed(
            PROFILE_PATH,
            target_id=TARGET_IDENTITY,
            body=np.asarray(body_gallery, dtype=np.float32),
            face_embeddings=np.asarray(face_embeddings, dtype=np.float32),
            face_present=np.asarray(face_present, dtype=np.bool_)
        )

        face_count = int(np.sum(face_present))
        print(
            f"[PROFILE] Saved {len(target_gallery)} samples "
            f"({face_count} with face embeddings) -> "
            f"{os.path.abspath(PROFILE_PATH)}"
        )
        return True

    except Exception as e:
        print("[PROFILE] Save failed:", e)
        return False


# ============================================================
# LOAD INSIGHTFACE PROFILE
# ============================================================

def load_target_profile():
    global target_gallery

    if not os.path.exists(PROFILE_PATH):
        return False

    try:
        with np.load(PROFILE_PATH, allow_pickle=False) as data:
            body = data["body"]
            face_embeddings = data["face_embeddings"]
            face_present = data["face_present"]

        target_gallery = []

        for i in range(len(body)):
            face = None
            if bool(face_present[i]):
                face = {
                    "embedding": np.asarray(
                        face_embeddings[i],
                        dtype=np.float32
                    )
                }

            target_gallery.append({
                "body": np.asarray(body[i], dtype=np.float32),
                "face": face
            })

        face_count = sum(1 for item in target_gallery if item["face"] is not None)
        print(
            f"[PROFILE] Loaded {len(target_gallery)} samples "
            f"({face_count} with InsightFace embeddings)"
        )

        if face_count == 0:
            print(
                "[PROFILE] WARNING: This profile contains no usable face "
                "embeddings. Enrol again with your face clearly visible."
            )

        if not os.path.exists(LOCK_PATH):
            write_target_lock()

        return len(target_gallery) > 0

    except Exception as e:
        print("[PROFILE] Load failed:", e)
        target_gallery = []
        return False


# ============================================================
# PERMANENT LOCK
# ============================================================

def write_target_lock():
    try:
        with open(
            LOCK_PATH,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(TARGET_IDENTITY + "\n")
            f.write(
                "Permanent target. "
                "Remove only with R.\n"
            )

        return True

    except Exception as e:
        print("[LOCK] Write failed:", e)
        return False


def remove_target_lock():
    for path in (
        PROFILE_PATH,
        LOCK_PATH
    ):
        if os.path.exists(path):
            try:
                os.remove(path)
                print(
                    f"[LOCK] Removed {path}"
                )
            except Exception as e:
                print(
                    f"[LOCK] Could not remove "
                    f"{path}: {e}"
                )


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):
    global mouse_x
    global mouse_y
    global mouse_clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y
        mouse_clicked = True


# ============================================================
# RESET
# ============================================================

def reset_target(delete_saved_profile=False):
    global target_selected
    global target_track_id
    global target_gallery
    global capture_active
    global capture_start_time
    global capture_samples
    global capture_frame_counter
    global last_seen_time
    global target_was_lost
    global reid_candidate_id
    global reid_candidate_frames
    global reid_candidate_score
    global target_score
    global mouse_x
    global mouse_y
    global mouse_clicked

    target_selected = False
    target_track_id = None
    target_gallery = []

    capture_active = False
    capture_start_time = 0.0
    capture_samples = []
    capture_frame_counter = 0

    last_seen_time = 0.0
    target_was_lost = False

    reid_candidate_id = None
    reid_candidate_frames = 0
    reid_candidate_score = 0.0

    target_score = 0.0

    mouse_x = -1
    mouse_y = -1
    mouse_clicked = False

    if delete_saved_profile:
        remove_target_lock()
        print(
            "[LOCK] TARGET-01 completely removed."
        )


# ============================================================
# START CAPTURE
# ============================================================

def start_capture(person, frame, now):
    global target_selected
    global target_track_id
    global target_gallery
    global capture_active
    global capture_start_time
    global capture_samples
    global capture_frame_counter
    global last_seen_time
    global target_was_lost

    descriptor = get_person_descriptor(
        frame,
        person["box"]
    )

    if descriptor is None:
        print(
            "[TARGET] Selection rejected: "
            "person crop too small."
        )
        return

    target_selected = True
    target_track_id = person["id"]

    # IMPORTANT:
    # Existing permanent profile is NOT replaced here
    # unless R was pressed and the profile was deleted.
    target_gallery = [descriptor]

    capture_active = True
    capture_start_time = now
    capture_samples = [descriptor]
    capture_frame_counter = 0

    last_seen_time = now
    target_was_lost = False

    print()
    print("==============================================")
    print(" TARGET SELECTED")
    print(" Permanent identity :", TARGET_IDENTITY)
    print(" Initial tracker ID :", target_track_id)
    print(
        " Capturing profile  : "
        f"{CAPTURE_SECONDS:.0f} seconds"
    )
    print("==============================================")
    print()


# ============================================================
# FINISH CAPTURE
# ============================================================

def finish_capture():
    global capture_active
    global target_gallery

    capture_active = False

    # Only real ArcFace embeddings count toward permanent face enrollment.
    face_samples = [
        descriptor for descriptor in capture_samples
        if descriptor is not None
        and descriptor.get("face") is not None
        and descriptor["face"].get("embedding") is not None
    ]

    unique = []
    for descriptor in face_samples:
        if not unique:
            unique.append(descriptor)
            continue

        best = max(
            descriptor_similarity(existing, descriptor)
            for existing in unique
        )
        if best < 0.995:
            unique.append(descriptor)

    target_gallery = unique[:MAX_GALLERY_SIZE]

    if len(target_gallery) < MIN_CAPTURE_SAMPLES:
        print()
        print("==============================================")
        print(" ENROLLMENT NOT ACCEPTED")
        print(f" Valid face samples : {len(target_gallery)}")
        print(f" Required minimum  : {MIN_CAPTURE_SAMPLES}")
        print(" Keep your face facing the webcam and move closer.")
        print(" Improve lighting, then click a person to try again.")
        print(" No new permanent profile has been saved.")
        print("==============================================")
        print()
        reset_target(delete_saved_profile=False)
        return

    if not save_target_profile():
        print("[PROFILE] ERROR: profile could not be saved.")
        reset_target(delete_saved_profile=False)
        return

    write_target_lock()

    print()
    print("==============================================")
    print(" TARGET PROFILE READY")
    print(" Permanent identity :", TARGET_IDENTITY)
    print(" Valid face samples :", len(target_gallery))
    print(" Recognition model  : InsightFace ArcFace")
    print(" Profile is immutable while searching.")
    print(" Press R only if you want a new target.")
    print("==============================================")
    print()


# ============================================================
# FULLSCREEN
# ============================================================

def set_fullscreen(value):
    global fullscreen

    fullscreen = value

    try:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
            if fullscreen
            else cv2.WINDOW_NORMAL
        )
    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

print()
print("==============================================")
print(" HUMAN TARGET FOLLOWING - INSIGHTFACE RE-ID V4.2 FIX")
print("==============================================")
print()

print("Initializing InsightFace...")
try:
    initialize_face_model()
except Exception as e:
    print("[FACE] Could not initialize InsightFace:", e)
    print("[FACE] Install a compatible insightface + onnxruntime build in this venv.")
    raise SystemExit(1)

print("Loading YOLO11...")
model = YOLO(MODEL_PATH)
print("YOLO11 loaded.")

print()
print("Loading saved target profile...")

profile_loaded = load_target_profile()

if profile_loaded:
    target_selected = True
    print(
        "[LOCK] TARGET-01 is permanently locked."
    )
    print(
        "[LOCK] Press R to deliberately remove it."
    )
else:
    print(
        "[LOCK] No target profile found."
    )
    print(
        "[LOCK] Click a person to create TARGET-01."
    )

print()
print("Opening camera...")

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)
cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)
cap.set(
    cv2.CAP_PROP_FPS,
    FPS
)
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

if not cap.isOpened():
    print(
        "ERROR: Cannot open camera:",
        CAMERA
    )
    raise SystemExit(1)

print("Camera opened successfully.")

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)

set_fullscreen(True)

fps_timer = time.time()
fps_counter = 0
display_fps = 0.0


# ============================================================
# LOOP
# ============================================================

# CPU-friendly cache: avoid running InsightFace repeatedly for the same track.
descriptor_cache = {}
frame_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame_index += 1

    frame = cv2.resize(
        frame,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    now = time.time()

    # --------------------------------------------------------
    # YOLO + BOT-SORT
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[PERSON_CLASS],
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]
    persons = []

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):
        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        if boxes.id is not None:
            ids = (
                boxes.id.cpu()
                .numpy()
                .astype(int)
            )
        else:
            ids = [-1] * len(xyxy)

        for box, conf, cls, pid in zip(
            xyxy,
            confidences,
            classes,
            ids
        ):
            if int(cls) != PERSON_CLASS:
                continue

            persons.append({
                "box": box.astype(int),
                "confidence": float(conf),
                "id": int(pid)
            })

    # ========================================================
    # NO TARGET SELECTED
    # ========================================================

    if not target_selected:

        for person in persons:
            x1, y1, x2, y2 = person["box"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BLUE,
                3
            )

            draw_text(
                frame,
                f"Person ID:{person['id']} "
                f"{person['confidence']:.2f}",
                (x1, max(30, y1 - 10)),
                WHITE,
                0.70,
                2
            )

        cv2.rectangle(
            frame,
            (20, 20),
            (760, 115),
            BLACK,
            -1
        )

        draw_text(
            frame,
            "CLICK A PERSON TO LOCK TARGET-01",
            (40, 60),
            YELLOW,
            0.82,
            2
        )

        draw_text(
            frame,
            "Target remains locked until R is pressed",
            (40, 95),
            WHITE,
            0.58,
            2
        )

        if mouse_clicked:

            selected_person = None

            for person in persons:
                x1, y1, x2, y2 = person["box"]

                if (
                    x1 <= mouse_x <= x2
                    and
                    y1 <= mouse_y <= y2
                ):
                    selected_person = person
                    break

            if selected_person is not None:
                start_capture(
                    selected_person,
                    frame,
                    now
                )

            mouse_clicked = False
            mouse_x = -1
            mouse_y = -1

    # ========================================================
    # TARGET MODE
    # ========================================================

    else:

        target_person = None
        target_score = 0.0

        # ====================================================
        # PROFILE CAPTURE
        # ====================================================

        if capture_active:

            elapsed_capture = (
                now - capture_start_time
            )

            current_target = None

            # Use temporary tracker ID ONLY during acquisition.
            for person in persons:
                if person["id"] == target_track_id:
                    current_target = person
                    break

            if current_target is not None:

                descriptor, descriptor_refreshed = get_cached_person_descriptor(
                    frame, current_target, frame_index, descriptor_cache
                )

                if descriptor is not None:

                    if descriptor_refreshed:
                        capture_frame_counter += 1
                        capture_samples.append(descriptor)

                    target_person = current_target
                    target_score = gallery_similarity(
                        descriptor
                    )

                    last_seen_time = now

            # If temporary tracker ID disappears during
            # acquisition, do not switch to another person.
            # This prevents profile contamination.

            if elapsed_capture >= CAPTURE_SECONDS:
                finish_capture()

            # Draw other people.
            for person in persons:
                x1, y1, x2, y2 = person["box"]

                if (
                    current_target is not None
                    and person["id"]
                    == current_target["id"]
                ):
                    continue

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    BLUE,
                    2
                )

            if target_person is not None:

                x1, y1, x2, y2 = (
                    target_person["box"]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    GREEN,
                    5
                )

                draw_text(
                    frame,
                    f"{TARGET_IDENTITY} CAPTURING",
                    (x1, max(35, y1 - 12)),
                    GREEN,
                    0.80,
                    2
                )

            remaining = max(
                0.0,
                CAPTURE_SECONDS - elapsed_capture
            )

            cv2.rectangle(
                frame,
                (20, 20),
                (760, 205),
                BLACK,
                -1
            )

            draw_text(
                frame,
                "TARGET DATA ACQUISITION",
                (40, 58),
                GREEN,
                0.85,
                2
            )

            draw_text(
                frame,
                f"TIME: {elapsed_capture:.1f} / "
                f"{CAPTURE_SECONDS:.1f} sec",
                (40, 95),
                WHITE,
                0.70,
                2
            )

            draw_text(
                frame,
                f"REMAINING: {remaining:.1f} sec",
                (40, 130),
                YELLOW,
                0.70,
                2
            )

            valid_face_count = sum(
                1 for item in capture_samples
                if item is not None and item.get("face") is not None
            )
            draw_text(
                frame,
                f"FACE SAMPLES: {valid_face_count}",
                (40, 165),
                CYAN,
                0.62,
                2
            )

            draw_text(
                frame,
                "KEEP TARGET IN VIEW",
                (40, 195),
                WHITE,
                0.55,
                2
            )

        # ====================================================
        # NORMAL TRACKING
        # ====================================================

        else:

            # ------------------------------------------------
            # FAST PATH:
            # current tracker ID is only accepted if its
            # appearance also matches the permanent profile.
            # ------------------------------------------------

            if not target_was_lost:

                for person in persons:

                    if (
                        person["id"]
                        != target_track_id
                    ):
                        continue

                    descriptor, descriptor_refreshed = get_cached_person_descriptor(
                        frame, person, frame_index, descriptor_cache
                    )

                    score = gallery_similarity(
                        descriptor
                    )

                    if (
                        descriptor is not None
                        and
                        score
                        >= VISIBLE_MATCH_THRESHOLD
                    ):
                        target_person = person
                        target_score = score

                    break

            # ------------------------------------------------
            # LOST:
            # Search EVERY person using permanent gallery.
            # Never change TARGET-01.
            # ------------------------------------------------

            if target_person is None:

                best_person = None
                best_score = 0.0
                second_best_score = 0.0

                candidate_scores = []

                for person in persons:

                    descriptor, descriptor_refreshed = get_cached_person_descriptor(
                        frame, person, frame_index, descriptor_cache
                    )

                    score = gallery_similarity(
                        descriptor
                    )

                    candidate_scores.append(
                        (
                            person["id"],
                            score
                        )
                    )

                    if score > best_score:
                        second_best_score = best_score
                        best_score = score
                        best_person = person

                    elif score > second_best_score:
                        second_best_score = score

                candidate_valid = (
                    best_person is not None
                    and best_score >= REID_THRESHOLD
                    and (
                        best_score
                        - second_best_score
                    ) >= REID_MARGIN
                )

                best_cache_entry = (
                    descriptor_cache.get(best_person["id"])
                    if best_person is not None else None
                )
                fresh_best_evaluation = (
                    best_cache_entry is not None
                    and best_cache_entry["frame_index"] == frame_index
                )

                # Count only genuinely refreshed observations. On the
                # intervening cached frames, KEEP the streak; do not reset it.
                if candidate_valid:
                    if fresh_best_evaluation:
                        candidate_id = best_person["id"]

                        if reid_candidate_id == candidate_id:
                            reid_candidate_frames += 1
                        else:
                            reid_candidate_id = candidate_id
                            reid_candidate_frames = 1

                        reid_candidate_score = best_score

                        if reid_candidate_frames >= REID_CONFIRM_FRAMES:
                            old_id = target_track_id
                            target_track_id = best_person["id"]
                            target_person = best_person
                            target_score = best_score
                            target_was_lost = False

                            reid_candidate_id = None
                            reid_candidate_frames = 0
                            reid_candidate_score = 0.0

                            print()
                            print("==============================================")
                            print(" TARGET RE-IDENTIFIED")
                            print(f" Permanent target : {TARGET_IDENTITY}")
                            print(f" Old tracker ID   : {old_id}")
                            print(f" New tracker ID   : {target_track_id}")
                            print(f" Match score      : {best_score:.3f}")
                            print("==============================================")
                            print()

                    # If this is a cached frame, preserve the current
                    # candidate streak until the next fresh descriptor.
                else:
                    # Reset only when a fresh evaluation disproves the
                    # candidate. Cached frames are not new evidence.
                    if fresh_best_evaluation or best_person is None:
                        reid_candidate_id = None
                        reid_candidate_frames = 0
                        reid_candidate_score = 0.0

            # =================================================
            # TARGET FOUND
            # =================================================

            if target_person is not None:

                last_seen_time = now
                target_was_lost = False

                x1, y1, x2, y2 = (
                    target_person["box"]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    GREEN,
                    5
                )

                draw_text(
                    frame,
                    f"{TARGET_IDENTITY} "
                    f"MATCH:{target_score:.2f}",
                    (
                        x1,
                        max(35, y1 - 12)
                    ),
                    GREEN,
                    0.82,
                    2
                )

                cv2.rectangle(
                    frame,
                    (20, 20),
                    (760, 225),
                    BLACK,
                    -1
                )

                draw_text(
                    frame,
                    f"TARGET LOCKED: "
                    f"{TARGET_IDENTITY}",
                    (40, 58),
                    GREEN,
                    0.85,
                    2
                )

                draw_text(
                    frame,
                    f"TARGET MATCH: "
                    f"{target_score:.2f}",
                    (40, 95),
                    WHITE,
                    0.68,
                    2
                )

                draw_text(
                    frame,
                    f"TEMP TRACK ID: "
                    f"{target_track_id}",
                    (40, 130),
                    CYAN,
                    0.62,
                    2
                )

                draw_text(
                    frame,
                    f"SAVED SAMPLES: "
                    f"{len(target_gallery)}",
                    (40, 160),
                    WHITE,
                    0.62,
                    2
                )

                draw_text(
                    frame,
                    "PROFILE: PERMANENT",
                    (40, 190),
                    CYAN,
                    0.60,
                    2
                )

                draw_text(
                    frame,
                    "R = REMOVE TARGET   "
                    "F11 = FULLSCREEN   Q = QUIT",
                    (40, 222),
                    WHITE,
                    0.52,
                    2
                )

            # =================================================
            # TARGET LOST
            # =================================================

            else:

                if last_seen_time > 0:
                    lost_seconds = (
                        now - last_seen_time
                    )
                else:
                    lost_seconds = 0.0

                target_was_lost = True

                cv2.rectangle(
                    frame,
                    (20, 20),
                    (900, 300),
                    BLACK,
                    -1
                )

                draw_text(
                    frame,
                    f"TARGET LOST: "
                    f"{TARGET_IDENTITY}",
                    (40, 62),
                    RED,
                    0.90,
                    2
                )

                draw_text(
                    frame,
                    "SEARCHING PERMANENT TARGET...",
                    (40, 105),
                    YELLOW,
                    0.72,
                    2
                )

                draw_text(
                    frame,
                    f"TARGET LOST FOR: "
                    f"{lost_seconds:.1f} sec",
                    (40, 140),
                    WHITE,
                    0.65,
                    2
                )

                draw_text(
                    frame,
                    f"SAVED SAMPLES: "
                    f"{len(target_gallery)}",
                    (40, 175),
                    CYAN,
                    0.62,
                    2
                )

                draw_text(
                    frame,
                    "PERMANENT PROFILE: NOT CHANGED",
                    (40, 210),
                    CYAN,
                    0.55,
                    2
                )

                if reid_candidate_id is not None:

                    draw_text(
                        frame,
                        f"VERIFYING TRACK "
                        f"{reid_candidate_id}: "
                        f"{reid_candidate_frames}/"
                        f"{REID_CONFIRM_FRAMES}  "
                        f"SCORE "
                        f"{reid_candidate_score:.2f}",
                        (40, 245),
                        ORANGE,
                        0.60,
                        2
                    )

                # Draw candidates only.
                for person in persons:

                    x1, y1, x2, y2 = (
                        person["box"]
                    )

                    descriptor, descriptor_refreshed = get_cached_person_descriptor(
                        frame, person, frame_index, descriptor_cache
                    )

                    score = gallery_similarity(
                        descriptor
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        YELLOW,
                        2
                    )

                    draw_text(
                        frame,
                        f"CANDIDATE "
                        f"{score:.2f}",
                        (
                            x1,
                            max(30, y1 - 8)
                        ),
                        YELLOW,
                        0.58,
                        2
                    )


    # ========================================================
    # FPS
    # ========================================================

    fps_counter += 1
    fps_elapsed = now - fps_timer

    if fps_elapsed >= 1.0:

        display_fps = (
            fps_counter / fps_elapsed
        )

        fps_counter = 0
        fps_timer = now

    draw_text(
        frame,
        f"FPS: {display_fps:.1f}",
        (WIDTH - 180, 40),
        GREEN,
        0.65,
        2
    )

    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key in (
        ord("q"),
        ord("Q")
    ):
        break

    elif key in (
        ord("r"),
        ord("R")
    ):
        reset_target(
            delete_saved_profile=True
        )

    elif key in (
        0x7A,
        0x0F
    ):
        set_fullscreen(
            not fullscreen
        )


# ============================================================
# CLEANUP
# ============================================================

print()
print("Stopping tracker...")

cap.release()
cv2.destroyAllWindows()

print("Camera released.")
print("Tracker stopped.")