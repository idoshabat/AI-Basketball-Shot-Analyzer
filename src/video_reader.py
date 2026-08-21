import argparse
import csv
from pathlib import Path

import cv2

from pose_detector import PoseDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"


def resolve_video_path(video_path: str) -> Path:
    path = Path(video_path)
    if path.exists():
        return path

    project_path = PROJECT_ROOT / video_path
    if project_path.exists():
        return project_path

    raise FileNotFoundError(f"Video file not found: {video_path}")


def build_output_path(video_path: Path) -> Path:
    return OUTPUT_DIR / f"{video_path.stem}_pose.mp4"


def build_keypoints_path(video_path: Path) -> Path:
    return DATA_DIR / f"{video_path.stem}_keypoints.csv"


def build_keypoint_headers(rows: list[dict]) -> list[str]:
    headers = ["frame"]

    for row in rows:
        for key in row:
            if key != "frame" and key not in headers:
                headers.append(key)

    return headers


def create_video_writer(cap, output_path: Path):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    codec_candidates = {
        ".webm": ("VP80", "VP90"),
        ".mp4": ("avc1", "H264", "mp4v"),
    }

    for codec in codec_candidates.get(output_path.suffix.lower(), ("avc1", "H264", "mp4v")):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Could not create video writer for: {output_path}")


def create_people_detector():
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


def count_people_in_frame(frame, detector) -> int:
    height, width = frame.shape[:2]
    if width <= 0:
        return 0

    target_width = 480
    scale = target_width / width
    resized = cv2.resize(frame, (target_width, int(height * scale))) if abs(scale - 1.0) > 0.05 else frame
    rectangles, _ = detector.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return len(rectangles)


def read_video(
    video_path: str,
    show_pose: bool = True,
    save_output: bool = False,
    save_keypoints: bool = False,
    display: bool = True,
    verbose: bool = True,
    output_path: str | Path | None = None,
    keypoints_path: str | Path | None = None,
) -> dict:
    path = resolve_video_path(video_path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_delay = int(1000 / fps) if fps > 0 else 30
    pose_detector = PoseDetector() if show_pose or save_keypoints else None
    output_path = Path(output_path) if output_path else build_output_path(path)
    keypoints_path = Path(keypoints_path) if keypoints_path else build_keypoints_path(path)
    writer = create_video_writer(cap, output_path) if save_output else None
    keypoint_rows = []
    people_detector = create_people_detector()
    people_sample_interval = max(1, frame_count // 8) if frame_count else 15
    scene_people_counts = []

    try:
        frame_number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            if frame_number == 1 or frame_number % people_sample_interval == 0:
                scene_people_counts.append(count_people_in_frame(frame, people_detector))

            pose_result = pose_detector.detect(frame) if pose_detector else None

            if save_keypoints and pose_detector:
                keypoints = pose_detector.extract_keypoints(pose_result)
                keypoint_rows.append({"frame": frame_number, **keypoints})

            if show_pose and pose_detector:
                frame = pose_detector.draw_pose(frame, pose_result)

            if writer:
                writer.write(frame)

            if display:
                cv2.imshow("Basketball Shot Video", frame)

            if display and cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                break
    finally:
        if pose_detector:
            pose_detector.close()
        if writer:
            writer.release()

    cap.release()
    if display:
        cv2.destroyAllWindows()

    if save_output and verbose:
        print(f"Saved output video: {output_path}")

    if save_keypoints:
        keypoints_path.parent.mkdir(parents=True, exist_ok=True)
        headers = build_keypoint_headers(keypoint_rows)
        with keypoints_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(keypoint_rows)
        if verbose:
            print(f"Saved keypoints CSV: {keypoints_path}")

    max_scene_people = max(scene_people_counts or [0])
    multi_person_samples = sum(1 for count in scene_people_counts if count >= 2)
    multi_person_sample_ratio = multi_person_samples / len(scene_people_counts) if scene_people_counts else 0.0

    return {
        "video_path": path,
        "output_path": output_path if save_output else None,
        "keypoints_path": keypoints_path if save_keypoints else None,
        "metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "scene_sampled_frames": len(scene_people_counts),
            "scene_max_people_count": max_scene_people,
            "scene_multi_person_frame_ratio": round(multi_person_sample_ratio, 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and display a basketball shot video.")
    parser.add_argument("video_path", help="Path to the video file, for example videos/ft10.mp4")
    parser.add_argument("--no-pose", action="store_true", help="Display the original video without pose detection")
    parser.add_argument("--save-output", action="store_true", help="Save the displayed video to the output folder")
    parser.add_argument("--save-keypoints", action="store_true", help="Save pose landmarks to the data folder")
    parser.add_argument("--no-display", action="store_true", help="Process the video without opening a display window")
    args = parser.parse_args()

    read_video(
        args.video_path,
        show_pose=not args.no_pose,
        save_output=args.save_output,
        save_keypoints=args.save_keypoints,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
