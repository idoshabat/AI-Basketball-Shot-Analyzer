import argparse
import csv
from pathlib import Path

import cv2

from pose_detector import PoseDetector
from shot_analyzer import analyze_shot
from video_reader import OUTPUT_DIR, create_video_writer, resolve_video_path


def build_annotated_video_path(video_path: Path) -> Path:
    return OUTPUT_DIR / f"{video_path.stem}_annotated.webm"


def draw_text(frame, text: str, position: tuple[int, int], scale: float = 0.6, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_banner(frame, text: str, color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 38), color, -1)
    draw_text(frame, text, (12, 26), scale=0.7)


def get_current_phase(frame_number: int, phases: dict) -> str:
    for phase_name, phase in phases.items():
        if phase["start_frame"] <= frame_number <= phase["end_frame"]:
            return phase_name

    return "unlabeled"


def load_ball_tracking_rows(ball_tracking_csv_path: str | Path | None) -> dict[int, dict]:
    if not ball_tracking_csv_path:
        return {}

    path = Path(ball_tracking_csv_path)
    if not path.exists():
        return {}

    rows = {}
    with path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                rows[int(float(row["frame"]))] = row
            except (KeyError, TypeError, ValueError):
                continue

    return rows


def draw_ball_overlay(frame, ball_row: dict | None) -> None:
    if not ball_row or ball_row.get("ball_detected") != "True":
        return

    try:
        ball_x = float(ball_row["ball_x"])
        ball_y = float(ball_row["ball_y"])
        confidence = float(ball_row["ball_confidence"])
    except (KeyError, TypeError, ValueError):
        return

    source = ball_row.get("candidate_source") or ""
    minimum_confidence = 0.05 if source.startswith(("roboflow:", "yolo:")) else 0.48
    if confidence < minimum_confidence:
        return

    height, width = frame.shape[:2]
    center = (int(ball_x * width), int(ball_y * height))
    radius = max(12, int(float(ball_row.get("ball_radius") or 0.015) * max(width, height)))
    cv2.circle(frame, center, radius + 5, (12, 14, 18), 4, cv2.LINE_AA)
    cv2.circle(frame, center, radius + 3, (87, 255, 201), 3, cv2.LINE_AA)
    label = "Ball" if source.startswith(("roboflow:", "yolo:")) else "Ball?"
    draw_text(frame, f"{label} {confidence:.2f}", (center[0] + radius + 8, max(32, center[1] - 8)), scale=0.48, color=(87, 255, 201))


def draw_overlay(frame, frame_number: int, analysis: dict) -> None:
    metrics = analysis["metrics"]
    phases = analysis["phases"]
    release_frame = metrics["release_frame"]
    follow_through_end = metrics["follow_through_end_frame"]
    follow_through_status = metrics["follow_through_status"]
    jump_start = metrics["jump_window_start"]
    jump_end = metrics["jump_window_end"]
    current_phase = get_current_phase(frame_number, phases)

    draw_text(frame, f"Frame: {frame_number}", (12, 28))
    draw_text(frame, f"Score: {analysis['score']}/100", (12, 56))
    draw_text(frame, f"View: {analysis['camera_view'].title()}", (12, 84))
    draw_text(frame, f"Release: {release_frame}", (12, 112))
    draw_text(frame, f"Method: {metrics['release_detection_method']}", (12, 140), scale=0.5)
    draw_text(
        frame,
        f"Release conf: {metrics['release_confidence']:.2f} ({metrics['release_confidence_label']})",
        (12, 168),
        scale=0.5,
    )
    draw_text(frame, f"Wrist vel: {metrics['release_wrist_y_velocity']:.3f}", (12, 196), scale=0.5)
    draw_text(frame, f"Phase: {current_phase.replace('_', ' ').title()}", (12, 224), scale=0.5)
    draw_text(frame, f"FT hold: {metrics['follow_through_frames']} frames", (12, 252), scale=0.5)
    draw_text(frame, f"Hip rise: {metrics['hip_rise']:.3f}", (12, 280), scale=0.5)
    draw_text(frame, f"Ankle lift: {metrics['ankle_lift']:.3f}", (12, 308), scale=0.5)

    has_high_quality_warning = any(warning.get("severity") == "high" for warning in analysis.get("quality_warnings", []))

    if has_high_quality_warning:
        draw_banner(frame, "INPUT QUALITY WARNING: tracked pose may not be the shooter", (40, 40, 190))
    elif frame_number == release_frame:
        draw_banner(
            frame,
            f"RELEASE: {metrics['release_detection_method']} / {metrics['release_confidence_label'].upper()}",
            (0, 120, 255),
        )
    elif follow_through_status == "held_until_video_end" and frame_number > release_frame:
        draw_banner(frame, "FOLLOW THROUGH HOLDING", (0, 150, 0))
    elif follow_through_end is not None and release_frame < frame_number < follow_through_end:
        draw_banner(frame, "FOLLOW THROUGH HOLD", (0, 150, 0))
    elif follow_through_end is not None and frame_number >= follow_through_end and frame_number <= follow_through_end + 10:
        draw_banner(frame, "FOLLOW THROUGH ENDED", (70, 70, 180))
    elif jump_start is not None and jump_end is not None and jump_start <= frame_number <= jump_end:
        draw_banner(frame, "JUMP WINDOW", (180, 80, 0))
    elif current_phase != "unlabeled":
        draw_banner(frame, current_phase.replace("_", " ").upper(), (80, 80, 80))


def annotate_video(
    video_path: str,
    features_csv_path: str,
    output_path: str | None = None,
    camera_view: str = "side",
    shooting_side: str = "auto",
    input_quality: dict | None = None,
    ball_tracking_csv_path: str | Path | None = None,
) -> Path:
    path = resolve_video_path(video_path)
    analysis = analyze_shot(features_csv_path, camera_view=camera_view, shooting_side=shooting_side, input_quality=input_quality)
    annotated_path = Path(output_path) if output_path else build_annotated_video_path(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    writer = create_video_writer(cap, annotated_path)
    pose_detector = PoseDetector()
    ball_rows = load_ball_tracking_rows(ball_tracking_csv_path)

    try:
        frame_number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            pose_result = pose_detector.detect(frame)
            frame = pose_detector.draw_pose(frame, pose_result)
            draw_ball_overlay(frame, ball_rows.get(frame_number))
            draw_overlay(frame, frame_number, analysis)
            writer.write(frame)
    finally:
        pose_detector.close()
        writer.release()
        cap.release()

    return annotated_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an annotated basketball shot video.")
    parser.add_argument("video_path", help="Path to a video file, for example videos/ft10.mp4")
    parser.add_argument("features_csv_path", help="Path to a features CSV, for example data/ft10_features.csv")
    parser.add_argument("--output", help="Optional output video path")
    parser.add_argument("--camera-view", choices=["side", "front", "back"], default="side", help="Camera angle used for view-specific overlays")
    args = parser.parse_args()

    output_path = annotate_video(args.video_path, args.features_csv_path, args.output, camera_view=args.camera_view)
    print(f"Saved annotated video: {output_path}")


if __name__ == "__main__":
    main()
