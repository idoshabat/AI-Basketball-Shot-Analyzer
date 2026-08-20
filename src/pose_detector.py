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
        num_poses: int = 1,
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
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        return self.landmarker.detect(mp_image)

    def pose_area(self, landmarks) -> float:
        visible_landmarks = [landmark for landmark in landmarks if getattr(landmark, "visibility", 1.0) is None or getattr(landmark, "visibility", 1.0) >= 0.35]
        if not visible_landmarks:
            visible_landmarks = landmarks

        min_x = min(landmark.x for landmark in visible_landmarks)
        max_x = max(landmark.x for landmark in visible_landmarks)
        min_y = min(landmark.y for landmark in visible_landmarks)
        max_y = max(landmark.y for landmark in visible_landmarks)
        return max(0.0, (max_x - min_x) * (max_y - min_y))

    def pose_shot_score(self, landmarks) -> float:
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[PoseLandmark.RIGHT_ELBOW.value]
        left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]

        shoulder_y = min(left_shoulder.y, right_shoulder.y)
        hip_y = (left_hip.y + right_hip.y) / 2
        highest_wrist_y = min(left_wrist.y, right_wrist.y)
        highest_elbow_y = min(left_elbow.y, right_elbow.y)
        wrist_above_shoulder = max(0.0, shoulder_y - highest_wrist_y)
        elbow_above_hip = max(0.0, hip_y - highest_elbow_y)
        area = self.pose_area(landmarks)

        return (wrist_above_shoulder * 3.0) + (elbow_above_hip * 1.2) + min(area, 0.35)

    def select_pose_landmarks(self, result):
        if not result or not result.pose_landmarks:
            return None

        return max(result.pose_landmarks, key=self.pose_shot_score)

    def draw_pose(self, frame, result=None):
        if result is None:
            result = self.detect(frame)

        landmarks = self.select_pose_landmarks(result)
        if not landmarks:
            return frame

        height, width = frame.shape[:2]

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

        if not result or not result.pose_landmarks:
            return keypoints

        landmarks = self.select_pose_landmarks(result)
        if not landmarks:
            return keypoints
        keypoints["detected_people_count"] = len(result.pose_landmarks)
        keypoints["selected_pose_area"] = self.pose_area(landmarks)
        keypoints["selected_pose_score"] = self.pose_shot_score(landmarks)
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
