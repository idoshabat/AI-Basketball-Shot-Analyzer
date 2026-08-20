import argparse
import base64
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2

from video_reader import resolve_video_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BALL_MODEL_PATH = PROJECT_ROOT / "model" / "basketball_yolo.pt"
SUPPORTED_BALL_BACKENDS = {"auto", "roboflow", "yolo", "heuristic"}
BALL_CLASS_NAME_HINTS = {"basketball", "sports ball", "ball"}


def load_project_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    file_values = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            file_values[key] = value

    for key, value in file_values.items():
        os.environ.setdefault(key, value)


load_project_env()


class YoloBallDetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.2,
        image_size: int | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_BALL_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.model = None
        self.names = {}
        self.error = None

        if not self.model_path.exists():
            self.error = f"YOLO model file not found: {self.model_path}"
            return

        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.error = f"ultralytics is not installed: {exc}"
            return

        try:
            self.model = YOLO(str(self.model_path))
            self.names = getattr(self.model, "names", {}) or {}
        except Exception as exc:
            self.error = f"Could not load YOLO model: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def is_ball_class(self, class_id: int) -> bool:
        label = str(self.names.get(class_id, class_id)).strip().lower()
        return label in BALL_CLASS_NAME_HINTS or label.endswith(" ball")

    def detect(self, frame) -> list[dict]:
        if not self.model:
            return []

        height, width = frame.shape[:2]
        try:
            results = self.model.predict(
                frame,
                verbose=False,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
            )
        except Exception as exc:
            self.error = f"YOLO inference failed: {exc}"
            return []

        candidates = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
                except Exception:
                    continue

                if confidence < self.confidence_threshold or not self.is_ball_class(class_id):
                    continue

                box_width = max(1.0, x2 - x1)
                box_height = max(1.0, y2 - y1)
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                radius = max(box_width, box_height) / (2 * max(width, height))
                aspect_ratio = box_width / box_height
                roundness = 1.0 - min(1.0, abs(1.0 - aspect_ratio))

                candidates.append(
                    {
                        "ball_x": center_x,
                        "ball_y": center_y,
                        "ball_radius": radius,
                        "ball_confidence": max(0.0, min(1.0, confidence)),
                        "candidate_source": f"yolo:{self.names.get(class_id, class_id)}",
                        "candidate_area": box_width * box_height,
                        "candidate_circularity": roundness,
                        "candidate_fill_ratio": 0.0,
                    }
                )

        return candidates


class RoboflowBallDetector:
    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        confidence_threshold: float = 0.05,
        timeout_seconds: float = 12.0,
    ) -> None:
        self.model_id = (model_id or os.getenv("ROBOFLOW_MODEL_ID") or "").strip().strip("/")
        self.api_key = (api_key or os.getenv("ROBOFLOW_API_KEY") or "").strip()
        self.api_url = (api_url or os.getenv("ROBOFLOW_API_URL") or "https://detect.roboflow.com").strip().rstrip("/")
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.error = None

        if not self.model_id:
            self.error = "ROBOFLOW_MODEL_ID is not set, for example basketball-detector-n4omu/1."
        elif not self.api_key:
            self.error = "ROBOFLOW_API_KEY is not set."

    @property
    def available(self) -> bool:
        return self.error is None

    def is_ball_class(self, label: str) -> bool:
        normalized = str(label).strip().lower()
        return normalized in BALL_CLASS_NAME_HINTS or normalized.endswith(" ball")

    def detect(self, frame) -> list[dict]:
        if not self.available:
            return []

        success, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success:
            self.error = "Could not encode frame for Roboflow inference."
            return []

        query = urlencode({"api_key": self.api_key})
        request = Request(
            f"{self.api_url}/{self.model_id}?{query}",
            data=base64.b64encode(encoded_image.tobytes()),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self.error = f"Roboflow inference failed: {exc}"
            return []

        image_info = payload.get("image") if isinstance(payload, dict) else {}
        frame_height, frame_width = frame.shape[:2]
        width = float(image_info.get("width") or frame_width)
        height = float(image_info.get("height") or frame_height)
        candidates = []

        for prediction in payload.get("predictions", []):
            label = prediction.get("class") or prediction.get("class_name") or ""
            confidence = float(prediction.get("confidence") or 0.0)
            if confidence < self.confidence_threshold or not self.is_ball_class(label):
                continue

            box_width = max(1.0, float(prediction.get("width") or 0.0))
            box_height = max(1.0, float(prediction.get("height") or 0.0))
            center_x_px = float(prediction.get("x") or 0.0)
            center_y_px = float(prediction.get("y") or 0.0)
            aspect_ratio = box_width / box_height
            roundness = 1.0 - min(1.0, abs(1.0 - aspect_ratio))

            candidates.append(
                {
                    "ball_x": center_x_px / width,
                    "ball_y": center_y_px / height,
                    "ball_radius": max(box_width, box_height) / (2 * max(width, height)),
                    "ball_confidence": max(0.0, min(1.0, confidence)),
                    "candidate_source": f"roboflow:{label}",
                    "candidate_area": box_width * box_height,
                    "candidate_circularity": roundness,
                    "candidate_fill_ratio": 0.0,
                }
            )

        return candidates


def normalize_backend(backend: str | None) -> str:
    normalized = (backend or os.getenv("BALL_DETECTION_BACKEND") or "auto").strip().lower()
    if normalized not in SUPPORTED_BALL_BACKENDS:
        raise ValueError(f"Unsupported ball detector backend: {backend}. Choose one of: auto, roboflow, yolo, heuristic.")

    return normalized


def resolve_model_path(model_path: str | Path | None = None) -> Path | None:
    raw_path = model_path or os.getenv("BALL_DETECTION_MODEL_PATH")
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    return DEFAULT_BALL_MODEL_PATH if DEFAULT_BALL_MODEL_PATH.exists() else None


def resolve_csv_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    if path.exists():
        return path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path

    raise FileNotFoundError(f"CSV file not found: {csv_path}")


def get_row_value(row: dict, key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None

    if math.isnan(value):
        return None

    return value


def wrist_point(row: dict, shooting_side: str) -> tuple[float, float] | None:
    x = get_row_value(row, f"{shooting_side}_wrist_x")
    y = get_row_value(row, f"{shooting_side}_wrist_y")
    if x is None or y is None:
        return None

    return x, y


def body_scale(row: dict) -> float:
    y_values = [
        get_row_value(row, key)
        for key in (
            "nose_y",
            "left_shoulder_y",
            "right_shoulder_y",
            "left_hip_y",
            "right_hip_y",
            "left_knee_y",
            "right_knee_y",
            "left_ankle_y",
            "right_ankle_y",
            "left_foot_index_y",
            "right_foot_index_y",
        )
    ]
    y_values = [value for value in y_values if value is not None]
    if len(y_values) < 2:
        return 0.25

    return max(0.12, max(y_values) - min(y_values))


def load_feature_rows(features_csv_path: str | Path) -> dict[int, dict]:
    rows = {}
    with resolve_csv_path(features_csv_path).open() as features_file:
        reader = csv.DictReader(features_file)
        for row in reader:
            try:
                frame = int(float(row["frame"]))
            except (KeyError, TypeError, ValueError):
                continue
            rows[frame] = row

    return rows


def find_ball_candidates(frame) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, (4, 55, 45), (28, 255, 255))
    orange_mask = cv2.medianBlur(orange_mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

    height, width = frame.shape[:2]
    min_area = max(18, int(width * height * 0.000025))
    max_area = max(140, int(width * height * 0.012))
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h else 0
        if circularity < 0.35 or aspect_ratio < 0.45 or aspect_ratio > 2.2:
            continue

        crop_mask = orange_mask[y : y + h, x : x + w]
        fill_ratio = cv2.countNonZero(crop_mask) / max(1, w * h)
        center_x = (x + w / 2) / width
        center_y = (y + h / 2) / height
        radius = max(w, h) / (2 * max(width, height))
        confidence = (
            0.46 * min(1.0, circularity)
            + 0.34 * min(1.0, fill_ratio * 2.2)
            + 0.20 * min(1.0, area / max(min_area * 3, 1))
        )
        candidates.append(
            {
                "ball_x": center_x,
                "ball_y": center_y,
                "ball_radius": radius,
                "ball_confidence": max(0.0, min(1.0, confidence)),
                "candidate_source": "color",
                "candidate_area": area,
                "candidate_circularity": circularity,
                "candidate_fill_ratio": fill_ratio,
            }
        )

    return candidates


def find_circle_candidates(frame, row: dict | None, shooting_side: str, previous: dict | None) -> list[dict]:
    height, width = frame.shape[:2]
    search_points = []
    wrist = wrist_point(row or {}, shooting_side)
    if wrist:
        search_points.append(wrist)
    if previous:
        search_points.append((previous["ball_x"], previous["ball_y"]))
    if not search_points:
        return []

    candidates = []
    seen = set()
    for center_x, center_y in search_points:
        pixel_x = int(center_x * width)
        pixel_y = int(center_y * height)
        roi_radius = max(80, int(max(width, height) * 0.14))
        x1 = max(0, pixel_x - roi_radius)
        y1 = max(0, pixel_y - roi_radius)
        x2 = min(width, pixel_x + roi_radius)
        y2 = min(height, pixel_y + roi_radius)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 1.5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=16,
            param1=80,
            param2=28,
            minRadius=max(6, int(max(width, height) * 0.009)),
            maxRadius=max(16, int(max(width, height) * 0.045)),
        )
        if circles is None:
            continue

        for circle in circles[0]:
            absolute_x = x1 + float(circle[0])
            absolute_y = y1 + float(circle[1])
            radius = float(circle[2])
            normalized_x = absolute_x / width
            normalized_y = absolute_y / height
            normalized_radius = radius / max(width, height)
            key = (round(normalized_x, 3), round(normalized_y, 3))
            if key in seen:
                continue
            seen.add(key)

            confidence = 0.62
            if wrist:
                wrist_distance = math.hypot(normalized_x - wrist[0], normalized_y - wrist[1])
                confidence += 0.30 * max(0.0, 1.0 - wrist_distance / 0.16)
                if normalized_y <= wrist[1] + 0.04:
                    confidence += 0.08
            candidates.append(
                {
                    "ball_x": normalized_x,
                    "ball_y": normalized_y,
                    "ball_radius": normalized_radius,
                    "ball_confidence": max(0.0, min(1.0, confidence)),
                    "candidate_source": "circle",
                    "candidate_area": math.pi * radius * radius,
                    "candidate_circularity": 1.0,
                    "candidate_fill_ratio": 0.0,
                }
            )

    return candidates


def choose_candidate(candidates: list[dict], row: dict | None, shooting_side: str, previous: dict | None) -> dict | None:
    if not candidates:
        return None

    wrist = wrist_point(row or {}, shooting_side)
    scale = body_scale(row or {})

    scored_candidates = []
    for candidate in candidates:
        score = candidate["ball_confidence"]
        source = str(candidate.get("candidate_source", ""))
        if source.startswith(("roboflow:", "yolo:")):
            score += 0.36
        if wrist:
            distance_to_wrist = math.hypot(candidate["ball_x"] - wrist[0], candidate["ball_y"] - wrist[1])
            candidate["distance_to_wrist"] = distance_to_wrist
            max_wrist_distance = max(0.22, scale * 1.35)
            is_plausibly_near_hand = distance_to_wrist <= max_wrist_distance
            is_not_clearly_below_hand = candidate["ball_y"] <= wrist[1] + max(0.05, scale * 0.22)
            if not (is_plausibly_near_hand and is_not_clearly_below_hand):
                if not previous:
                    continue

                travel_from_previous = math.hypot(
                    candidate["ball_x"] - previous["ball_x"],
                    candidate["ball_y"] - previous["ball_y"],
                )
                if travel_from_previous > 0.12:
                    continue

            score += 0.46 * max(0.0, 1.0 - distance_to_wrist / max(0.18, scale * 0.9))
            if candidate.get("candidate_source") == "circle" and candidate["ball_y"] <= wrist[1] + 0.04:
                score += 0.28
        else:
            candidate["distance_to_wrist"] = None

        if previous:
            travel = math.hypot(candidate["ball_x"] - previous["ball_x"], candidate["ball_y"] - previous["ball_y"])
            score += 0.20 * max(0.0, 1.0 - travel / 0.22)

        scored_candidates.append((score, candidate))

    if not scored_candidates:
        return None

    return max(scored_candidates, key=lambda item: item[0])[1]


def is_strict_fallback_candidate(candidate: dict, row: dict | None, shooting_side: str) -> bool:
    wrist = wrist_point(row or {}, shooting_side)
    if not wrist:
        return False

    scale = body_scale(row or {})
    distance_to_wrist = math.hypot(candidate["ball_x"] - wrist[0], candidate["ball_y"] - wrist[1])
    candidate["distance_to_wrist"] = distance_to_wrist
    max_distance = max(0.08, scale * 0.45)
    max_below_wrist = max(0.035, scale * 0.14)
    if distance_to_wrist > max_distance or candidate["ball_y"] > wrist[1] + max_below_wrist:
        return False

    source = candidate.get("candidate_source")
    if source == "circle":
        return candidate["ball_confidence"] >= 0.78 and candidate["ball_radius"] <= 0.055
    if source == "color":
        return candidate["ball_confidence"] >= 0.48

    return False


def build_frame_candidates(
    frame,
    frame_number: int,
    feature_row: dict | None,
    shooting_side: str,
    previous_candidate: dict | None,
    detector_backend: str,
    yolo_detector: YoloBallDetector | None,
    roboflow_detector: RoboflowBallDetector | None,
    pose_release_frame: int | None,
) -> list[dict]:
    roboflow_candidates = (
        roboflow_detector.detect(frame) if roboflow_detector and roboflow_detector.available else []
    )
    if detector_backend == "roboflow":
        return roboflow_candidates

    yolo_candidates = yolo_detector.detect(frame) if yolo_detector and yolo_detector.available else []
    if detector_backend == "yolo":
        if yolo_candidates:
            return yolo_candidates

        if pose_release_frame is None or not (pose_release_frame - 18 <= frame_number <= pose_release_frame + 24):
            return []

        fallback_candidates = [
            *find_ball_candidates(frame),
            *find_circle_candidates(frame, feature_row, shooting_side, previous_candidate),
        ]
        return [
            candidate
            for candidate in fallback_candidates
            if is_strict_fallback_candidate(candidate, feature_row, shooting_side)
        ]

    heuristic_candidates = [
        *find_ball_candidates(frame),
        *find_circle_candidates(frame, feature_row, shooting_side, previous_candidate),
    ]
    return [*roboflow_candidates, *yolo_candidates, *heuristic_candidates]


def env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

    return max(minimum, value)


def should_request_remote_frame(
    frame_number: int,
    pose_release_frame: int | None,
    stride: int,
    before_release: int,
    after_release: int,
) -> bool:
    if pose_release_frame is None:
        return frame_number == 1 or frame_number % stride == 0

    if frame_number == pose_release_frame:
        return True

    window_start = max(1, pose_release_frame - before_release)
    window_end = pose_release_frame + after_release
    if frame_number < window_start or frame_number > window_end:
        return False

    return (frame_number - window_start) % stride == 0


def prefetch_roboflow_candidates(
    video_path: Path,
    detector: RoboflowBallDetector,
    pose_release_frame: int | None,
    stride: int,
    before_release: int,
    after_release: int,
    max_workers: int,
) -> tuple[dict[int, list[dict]], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames_to_detect = []
    try:
        frame_number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            if should_request_remote_frame(frame_number, pose_release_frame, stride, before_release, after_release):
                frames_to_detect.append((frame_number, frame.copy()))
    finally:
        cap.release()

    if not frames_to_detect:
        return {}, 0

    candidates_by_frame = {}
    worker_count = min(max_workers, len(frames_to_detect))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(detector.detect, frame): frame_number
            for frame_number, frame in frames_to_detect
        }
        for future in as_completed(futures):
            frame_number = futures[future]
            try:
                candidates_by_frame[frame_number] = future.result()
            except Exception as exc:
                detector.error = f"Roboflow inference failed on frame {frame_number}: {exc}"
                candidates_by_frame[frame_number] = []

    return candidates_by_frame, len(frames_to_detect)


def infer_ball_release_frame(rows: list[dict], pose_release_frame: int | None) -> int | None:
    usable_rows = [
        row
        for row in rows
        if row["ball_detected"]
        and row.get("distance_to_wrist") is not None
        and row["ball_confidence"] >= 0.35
        and row["distance_to_wrist"] <= 0.2
    ]
    if not usable_rows:
        return None

    if pose_release_frame is None:
        return max(usable_rows, key=lambda row: row["ball_y_velocity"] or 0.0)["frame"]

    search_rows = [
        row
        for row in usable_rows
        if pose_release_frame - 14 <= row["frame"] <= pose_release_frame + 18
    ]
    if not search_rows:
        return None

    for index, row in enumerate(search_rows):
        if row["frame"] < pose_release_frame - 8:
            continue

        distance = row.get("distance_to_wrist") or 0.0
        velocity = row.get("ball_y_velocity") or 0.0
        next_rows = search_rows[index : index + 3]
        separated_count = sum(1 for next_row in next_rows if (next_row.get("distance_to_wrist") or 0.0) >= 0.075)
        if distance >= 0.075 and velocity < 0.02 and separated_count >= 2:
            return row["frame"]

    closest_row = min(search_rows, key=lambda row: abs(row["frame"] - pose_release_frame))
    return closest_row["frame"]


def summarize_ball_tracking(rows: list[dict], pose_release_frame: int | None) -> dict:
    detected_rows = [row for row in rows if row["ball_detected"]]
    high_confidence_rows = [row for row in detected_rows if row["ball_confidence"] >= 0.35]
    close_high_confidence_rows = [
        row
        for row in high_confidence_rows
        if row.get("distance_to_wrist") is not None and row["distance_to_wrist"] <= 0.2
    ]
    ball_release_frame = infer_ball_release_frame(rows, pose_release_frame)
    post_release_rows = [
        row
        for row in close_high_confidence_rows
        if ball_release_frame is not None and row["frame"] >= ball_release_frame
    ]

    avg_wrist_distance = None
    wrist_distances = [row["distance_to_wrist"] for row in high_confidence_rows if row.get("distance_to_wrist") is not None]
    if wrist_distances:
        avg_wrist_distance = sum(wrist_distances) / len(wrist_distances)

    arc_height = None
    if len(post_release_rows) >= 3:
        release_row = min(post_release_rows, key=lambda row: abs(row["frame"] - ball_release_frame))
        highest_y = min(row["ball_y"] for row in post_release_rows)
        arc_height = max(0.0, release_row["ball_y"] - highest_y)

    visibility_ratio = len(high_confidence_rows) / len(rows) if rows else 0.0
    close_visibility_ratio = len(close_high_confidence_rows) / len(rows) if rows else 0.0
    status = "not_detected"
    if close_visibility_ratio >= 0.45:
        status = "tracked"
    elif close_visibility_ratio >= 0.18 or visibility_ratio >= 0.35:
        status = "partial"

    source_counts = {}
    for row in high_confidence_rows:
        source = row.get("candidate_source") or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "status": status,
        "detected": bool(high_confidence_rows),
        "detected_frames": len(high_confidence_rows),
        "close_detected_frames": len(close_high_confidence_rows),
        "total_frames": len(rows),
        "visibility_ratio": round(visibility_ratio, 3),
        "close_visibility_ratio": round(close_visibility_ratio, 3),
        "ball_release_frame": ball_release_frame,
        "pose_release_frame": pose_release_frame,
        "release_frame_delta": None if ball_release_frame is None or pose_release_frame is None else ball_release_frame - pose_release_frame,
        "post_release_tracked_frames": len(post_release_rows),
        "arc_height": None if arc_height is None else round(arc_height, 4),
        "avg_wrist_distance": None if avg_wrist_distance is None else round(avg_wrist_distance, 4),
        "candidate_source_counts": source_counts,
        "note": "Beta: ball tracking is informational and does not affect the score yet.",
    }


def track_ball(
    video_path: str | Path,
    features_csv_path: str | Path,
    output_csv_path: str | Path,
    shooting_side: str,
    pose_release_frame: int | None = None,
    backend: str | None = None,
    model_path: str | Path | None = None,
    yolo_confidence: float | None = None,
) -> dict:
    backend = normalize_backend(backend)
    path = resolve_video_path(str(video_path))
    feature_rows = load_feature_rows(features_csv_path)
    output_path = Path(output_csv_path)
    resolved_model_path = resolve_model_path(model_path)
    yolo_detector = None
    roboflow_detector = None
    detector_backend = "heuristic"
    detector_message = "Using heuristic fallback detector."
    confidence_threshold = yolo_confidence
    if confidence_threshold is None:
        confidence_threshold = float(os.getenv("BALL_DETECTION_CONFIDENCE", "0.05"))
    image_size = int(os.getenv("BALL_DETECTION_IMAGE_SIZE", "1280"))
    roboflow_frame_stride = env_int("ROBOFLOW_FRAME_STRIDE", 2)
    roboflow_window_before = env_int("ROBOFLOW_RELEASE_WINDOW_BEFORE", 18)
    roboflow_window_after = env_int("ROBOFLOW_RELEASE_WINDOW_AFTER", 42)
    roboflow_max_workers = env_int("ROBOFLOW_MAX_WORKERS", 8)

    roboflow_model_id = os.getenv("ROBOFLOW_MODEL_ID")
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    if backend in {"auto", "roboflow"} and roboflow_model_id and roboflow_api_key:
        roboflow_detector = RoboflowBallDetector(
            roboflow_model_id,
            roboflow_api_key,
            confidence_threshold=confidence_threshold,
        )
        if roboflow_detector.available:
            detector_backend = "roboflow"
            detector_message = f"Using Roboflow model: {roboflow_detector.model_id}"
        elif backend == "roboflow":
            detector_message = f"Roboflow unavailable; using heuristic fallback. {roboflow_detector.error}"
        else:
            detector_message = f"Roboflow unavailable; checking local YOLO. {roboflow_detector.error}"
    elif backend == "roboflow":
        roboflow_detector = RoboflowBallDetector(
            roboflow_model_id,
            roboflow_api_key,
            confidence_threshold=confidence_threshold,
        )
        detector_message = f"Roboflow backend requested, but it is not configured. {roboflow_detector.error}"

    if detector_backend == "heuristic" and backend in {"auto", "yolo"} and resolved_model_path:
        yolo_detector = YoloBallDetector(resolved_model_path, confidence_threshold, image_size)
        if yolo_detector.available:
            detector_backend = "yolo"
            detector_message = f"Using YOLO model: {resolved_model_path}"
        elif backend == "yolo":
            detector_message = f"YOLO unavailable; using heuristic fallback. {yolo_detector.error}"
        else:
            detector_message = f"YOLO unavailable; using heuristic fallback. {yolo_detector.error}"
    elif backend == "yolo":
        detector_message = "YOLO backend requested, but no BALL_DETECTION_MODEL_PATH/model file was found; using heuristic fallback."

    roboflow_candidates_by_frame = {}
    roboflow_requested_frames = 0
    if detector_backend == "roboflow" and roboflow_detector and roboflow_detector.available:
        roboflow_candidates_by_frame, roboflow_requested_frames = prefetch_roboflow_candidates(
            path,
            roboflow_detector,
            pose_release_frame,
            roboflow_frame_stride,
            roboflow_window_before,
            roboflow_window_after,
            roboflow_max_workers,
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    rows = []
    previous_candidate = None

    try:
        frame_number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            feature_row = feature_rows.get(frame_number)
            if detector_backend == "roboflow":
                candidates = roboflow_candidates_by_frame.get(frame_number, [])
            else:
                candidates = build_frame_candidates(
                    frame,
                    frame_number,
                    feature_row,
                    shooting_side,
                    previous_candidate,
                    detector_backend,
                    yolo_detector,
                    roboflow_detector,
                    pose_release_frame,
                )
            candidate = choose_candidate(candidates, feature_row, shooting_side, previous_candidate)
            if candidate:
                previous_candidate = candidate
                row = {
                    "frame": frame_number,
                    "ball_detected": True,
                    "ball_x": candidate["ball_x"],
                    "ball_y": candidate["ball_y"],
                    "ball_radius": candidate["ball_radius"],
                    "ball_confidence": candidate["ball_confidence"],
                    "distance_to_wrist": candidate.get("distance_to_wrist"),
                    "candidate_count": len(candidates),
                    "candidate_source": candidate.get("candidate_source"),
                }
            else:
                row = {
                    "frame": frame_number,
                    "ball_detected": False,
                    "ball_x": None,
                    "ball_y": None,
                    "ball_radius": None,
                    "ball_confidence": 0.0,
                    "distance_to_wrist": None,
                    "candidate_count": len(candidates),
                    "candidate_source": None,
                }
            rows.append(row)
    finally:
        cap.release()

    last_x = None
    last_y = None
    for row in rows:
        if row["ball_detected"] and row["ball_x"] is not None and row["ball_y"] is not None:
            row["ball_x_velocity"] = 0.0 if last_x is None else row["ball_x"] - last_x
            row["ball_y_velocity"] = 0.0 if last_y is None else row["ball_y"] - last_y
            last_x = row["ball_x"]
            last_y = row["ball_y"]
        else:
            row["ball_x_velocity"] = None
            row["ball_y_velocity"] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        fieldnames = [
            "frame",
            "ball_detected",
            "ball_x",
            "ball_y",
            "ball_radius",
            "ball_confidence",
            "distance_to_wrist",
            "candidate_count",
            "candidate_source",
            "ball_x_velocity",
            "ball_y_velocity",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_ball_tracking(rows, pose_release_frame)
    summary["detector_backend"] = detector_backend
    summary["requested_backend"] = backend
    summary["detector_message"] = detector_message
    summary["model_path"] = None if resolved_model_path is None else str(resolved_model_path)
    summary["model_error"] = None if not yolo_detector else yolo_detector.error
    summary["roboflow_model_id"] = None if not roboflow_detector else roboflow_detector.model_id
    summary["roboflow_error"] = None if not roboflow_detector else roboflow_detector.error
    summary["roboflow_frame_stride"] = roboflow_frame_stride
    summary["roboflow_requested_frames"] = roboflow_requested_frames
    summary["roboflow_max_workers"] = roboflow_max_workers
    summary["roboflow_release_window"] = {
        "before": roboflow_window_before,
        "after": roboflow_window_after,
    }
    summary["yolo_confidence"] = confidence_threshold
    summary["yolo_image_size"] = image_size
    summary["csv_path"] = str(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Track basketball position in a shot video.")
    parser.add_argument("video_path", help="Path to a video file, for example videos/ft1.mp4")
    parser.add_argument("features_csv_path", help="Path to a features CSV")
    parser.add_argument("--output", default=str(DATA_DIR / "ball_tracking.csv"), help="Output CSV path")
    parser.add_argument("--shooting-side", choices=["right", "left"], default="right", help="Shooting hand")
    parser.add_argument("--release-frame", type=int, default=None, help="Pose-estimated release frame")
    parser.add_argument("--backend", choices=sorted(SUPPORTED_BALL_BACKENDS), default=None, help="Ball detector backend")
    parser.add_argument("--model-path", default=None, help="YOLO model path, for example model/basketball_yolo.pt")
    parser.add_argument("--yolo-confidence", type=float, default=None, help="YOLO confidence threshold")
    parser.add_argument("--roboflow-model-id", default=None, help="Roboflow model id, for example basketball-detector-n4omu/1")
    parser.add_argument("--roboflow-api-key", default=None, help="Roboflow API key")
    args = parser.parse_args()

    if args.roboflow_model_id:
        os.environ["ROBOFLOW_MODEL_ID"] = args.roboflow_model_id
    if args.roboflow_api_key:
        os.environ["ROBOFLOW_API_KEY"] = args.roboflow_api_key

    summary = track_ball(
        args.video_path,
        args.features_csv_path,
        args.output,
        args.shooting_side,
        args.release_frame,
        backend=args.backend,
        model_path=args.model_path,
        yolo_confidence=args.yolo_confidence,
    )
    print(f"Saved ball tracking CSV: {summary['csv_path']}")
    print(summary)


if __name__ == "__main__":
    main()
