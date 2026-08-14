import os
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "pose_landmarker_lite.task"
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmark,
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)


class PoseDetector:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        return self.landmarker.detect(mp_image)

    def draw_pose(self, frame, result=None):
        if result is None:
            result = self.detect(frame)
        if not result.pose_landmarks:
            return frame

        height, width = frame.shape[:2]
        landmarks = result.pose_landmarks[0]

        for connection in PoseLandmarksConnections.POSE_LANDMARKS:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        for landmark in landmarks:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

        return frame

    def extract_keypoints(self, result) -> dict:
        keypoints = {}

        if not result.pose_landmarks:
            return keypoints

        landmarks = result.pose_landmarks[0]
        for pose_landmark in PoseLandmark:
            landmark = landmarks[pose_landmark.value]
            name = pose_landmark.name.lower()
            keypoints[f"{name}_x"] = landmark.x
            keypoints[f"{name}_y"] = landmark.y
            keypoints[f"{name}_z"] = landmark.z
            keypoints[f"{name}_visibility"] = getattr(landmark, "visibility", None)
            keypoints[f"{name}_presence"] = getattr(landmark, "presence", None)

        return keypoints

    def close(self) -> None:
        self.landmarker.close()
