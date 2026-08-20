import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

import cv2

from ball_detector import track_ball
from feature_extractor import extract_features
from shot_analyzer import analyze_shot, build_coaching_item, print_report
from video_annotator import annotate_video
from visualize_features import plot_angles, plot_follow_through_debug
from video_reader import read_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
STORAGE_DIR = PROJECT_ROOT / "storage"
ANALYSES_DIR = STORAGE_DIR / "analyses"
ANALYSIS_VERSION = "rule-based-mvp-2026-08-15"


def safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number


def load_ball_rows(ball_tracking_path: str | Path) -> list[dict]:
    path = Path(ball_tracking_path)
    if not path.exists():
        return []

    rows = []
    with path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("ball_detected") != "True":
                continue

            parsed_row = {
                "frame": int(float(row["frame"])),
                "ball_x": safe_float(row.get("ball_x")),
                "ball_y": safe_float(row.get("ball_y")),
                "ball_confidence": safe_float(row.get("ball_confidence")) or 0.0,
                "distance_to_wrist": safe_float(row.get("distance_to_wrist")),
                "candidate_source": row.get("candidate_source"),
            }
            if parsed_row["ball_x"] is not None and parsed_row["ball_y"] is not None:
                rows.append(parsed_row)

    return rows


def detected_ball_rows_near_release(rows: list[dict], release_frame: int | None, before: int = 12, after: int = 24) -> list[dict]:
    if release_frame is None:
        return rows

    return [row for row in rows if release_frame - before <= row["frame"] <= release_frame + after]


def add_ball_based_insights(analysis: dict, ball_tracking: dict, ball_tracking_path: str | Path) -> None:
    camera_view = analysis.get("camera_view", "side")
    metrics = analysis["metrics"]
    phases = analysis["phases"]
    feedback = analysis["feedback"]
    coaching_items = analysis["coaching_items"]
    release_frame = metrics.get("release_frame")
    ball_rows = load_ball_rows(ball_tracking_path)
    release_rows = detected_ball_rows_near_release(ball_rows, release_frame)
    high_confidence_rows = [row for row in release_rows if row["ball_confidence"] >= 0.35]

    metrics["ball_release_confidence_label"] = "unavailable"
    metrics["ball_post_release_upward_frames"] = None
    metrics["ball_side_arc_quality"] = None
    metrics["ball_front_back_line_drift"] = None
    metrics["ball_release_wrist_distance"] = ball_tracking.get("avg_wrist_distance")

    if not high_confidence_rows:
        feedback.append("Ball tracking was not stable enough to add ball-flight feedback.")
        coaching_items.append(
            build_coaching_item(
                title="Improve ball visibility for ball-flight feedback",
                severity="low",
                metric="ball_visibility_ratio",
                value=ball_tracking.get("visibility_ratio"),
                target="Clear basketball detections around release",
                why_it_matters="Ball-based feedback needs the ball visible near the hand and after release.",
                drill="Record with the ball clearly visible against a simple background and avoid cropping the release path.",
                phase="release",
                start_frame=release_frame,
                end_frame=release_frame,
            )
        )
        return

    ball_release_frame = ball_tracking.get("ball_release_frame")
    if ball_release_frame is not None and release_frame is not None:
        delta = ball_release_frame - release_frame
        metrics["ball_release_confidence_label"] = "aligned" if abs(delta) <= 3 else "offset"
        if abs(delta) <= 3:
            feedback.append("Ball tracking confirms the release timing within a few frames of the pose estimate.")
        else:
            feedback.append("Ball tracking suggests the true ball release may be offset from the pose-only estimate.")
            coaching_items.append(
                build_coaching_item(
                    title="Review the actual ball release moment",
                    severity="medium",
                    metric="ball_release_frame_delta",
                    value=delta,
                    target="Within 3 frames of the pose release estimate",
                    why_it_matters="If the ball separates earlier or later than the body estimate, timing feedback can be less precise.",
                    drill="Use the annotated video and pause around release to verify when the ball leaves the fingers.",
                    phase="release",
                    start_frame=ball_release_frame,
                    end_frame=ball_release_frame,
                )
            )

    post_release_rows = [
        row
        for row in high_confidence_rows
        if ball_release_frame is not None and ball_release_frame <= row["frame"] <= ball_release_frame + 18
    ]
    if camera_view == "side":
        if len(post_release_rows) >= 3:
            release_row = min(post_release_rows, key=lambda row: abs(row["frame"] - ball_release_frame))
            upward_rows = [row for row in post_release_rows if row["ball_y"] < release_row["ball_y"] - 0.02]
            arc_height = ball_tracking.get("arc_height") or 0.0
            metrics["ball_post_release_upward_frames"] = len(upward_rows)

            if arc_height >= 0.08 and len(upward_rows) >= 3:
                metrics["ball_side_arc_quality"] = "good"
                feedback.append("Side-view ball path shows a visible upward arc after release.")
            elif arc_height >= 0.04:
                metrics["ball_side_arc_quality"] = "moderate"
                feedback.append("Side-view ball path has some lift, but the arc could be higher.")
                coaching_items.append(
                    build_coaching_item(
                        title="Create a slightly higher ball arc",
                        severity="medium",
                        metric="ball_arc_height",
                        value=arc_height,
                        target="0.08 or higher normalized side-view rise",
                        why_it_matters="From the side, a flatter ball path usually leaves less margin over the rim.",
                        drill="Shoot close form reps focusing on a relaxed upward finish and a high hand through release.",
                        phase="release",
                        start_frame=ball_release_frame,
                        end_frame=ball_release_frame + 18,
                    )
                )
            else:
                metrics["ball_side_arc_quality"] = "flat"
                feedback.append("Side-view ball path looks flat after release.")
                coaching_items.append(
                    build_coaching_item(
                        title="Add more upward lift to the ball path",
                        severity="high",
                        metric="ball_arc_height",
                        value=arc_height,
                        target="0.04 or higher normalized side-view rise",
                        why_it_matters="A very flat ball path gives the shot less vertical margin and can make distance control harder.",
                        drill="Use one-hand form shots and finish with fingers high, letting the ball rise before it travels forward.",
                        phase="release",
                        start_frame=ball_release_frame,
                        end_frame=ball_release_frame + 18,
                    )
                )
        else:
            feedback.append("Side-view arc feedback needs more ball detections after release.")
    else:
        if len(post_release_rows) >= 3:
            release_row = min(post_release_rows, key=lambda row: abs(row["frame"] - ball_release_frame))
            x_values = [row["ball_x"] for row in post_release_rows]
            line_drift = max(abs(x - release_row["ball_x"]) for x in x_values)
            metrics["ball_front_back_line_drift"] = round(line_drift, 4)
            if line_drift <= 0.035:
                feedback.append("Ball path stays close to the release line from this front/back view.")
            elif line_drift <= 0.07:
                feedback.append("Ball path drifts slightly sideways from this front/back view.")
                coaching_items.append(
                    build_coaching_item(
                        title="Keep the ball on the release line",
                        severity="medium",
                        metric="ball_front_back_line_drift",
                        value=round(line_drift, 4),
                        target="0.035 or lower normalized sideways drift",
                        why_it_matters="From front/back view, sideways ball drift can reveal a release that pushes across the body.",
                        drill="Shoot front-view form reps and freeze with wrist, elbow, and ball on the same vertical line.",
                        phase="release",
                        start_frame=ball_release_frame,
                        end_frame=ball_release_frame + 18,
                    )
                )
            else:
                feedback.append("Ball path drifts sideways from this front/back view.")
                coaching_items.append(
                    build_coaching_item(
                        title="Reduce sideways ball drift",
                        severity="high",
                        metric="ball_front_back_line_drift",
                        value=round(line_drift, 4),
                        target="0.035 or lower normalized sideways drift",
                        why_it_matters="Sideways drift after release often means the wrist or forearm is not finishing through the shot line.",
                        drill="Use close-range one-hand reps and hold the follow-through on the center line.",
                        phase="release",
                        start_frame=ball_release_frame,
                        end_frame=ball_release_frame + 18,
                    )
                )
        feedback.append("Arc height is not graded from front/back view because camera depth distorts the true shot arc.")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "video"


def build_run_id(video_path: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    video_stem = slugify(Path(video_path).stem)
    short_id = uuid4().hex[:8]
    return f"{timestamp}_{video_stem}_{short_id}"


def create_analysis_run(video_path: str, run_dir: str | Path | None = None) -> dict:
    run_path = Path(run_dir) if run_dir else ANALYSES_DIR / build_run_id(video_path)
    paths = {
        "run_dir": run_path,
        "input_dir": run_path / "input",
        "data_dir": run_path / "data",
        "output_dir": run_path / "output",
        "input_video": run_path / "input" / f"original{Path(video_path).suffix.lower()}",
        "keypoints": run_path / "data" / "keypoints.csv",
        "features": run_path / "data" / "features.csv",
        "ball_tracking": run_path / "data" / "ball_tracking.csv",
        "pose_video": run_path / "output" / "pose.mp4",
        "angles_chart": run_path / "output" / "angles.png",
        "follow_through_debug_chart": run_path / "output" / "follow_through_debug.png",
        "annotated_video": run_path / "output" / "annotated.webm",
        "report": run_path / "report.json",
    }

    for directory_key in ("input_dir", "data_dir", "output_dir"):
        paths[directory_key].mkdir(parents=True, exist_ok=True)

    return paths


def build_report_path(video_path: str) -> Path:
    path = Path(video_path)
    return OUTPUT_DIR / f"{path.stem}_report.json"


def format_path(path: Path | None) -> str | None:
    if path is None:
        return None

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def save_report(video_path: str, result: dict, report_path: str | Path | None = None) -> Path:
    report_path = Path(report_path) if report_path else build_report_path(video_path)
    files = {
        "original_video": format_path(result["input_video_path"]),
        "keypoints_csv": format_path(result["keypoints_path"]),
        "features_csv": format_path(result["features_path"]),
        "ball_tracking_csv": format_path(result["ball_tracking_path"]),
        "angles_chart": format_path(result["chart_path"]),
        "follow_through_debug_chart": format_path(result["follow_through_debug_chart_path"]),
        "pose_video": format_path(result["output_path"]),
        "annotated_video": format_path(result["annotated_video_path"]),
        "json_report": format_path(report_path),
    }
    files.update({name: format_path(path) for name, path in result.get("coaching_frame_paths", {}).items()})

    report = {
        "run_id": result["run_id"],
        "analysis_version": ANALYSIS_VERSION,
        "owner_user_id": result.get("owner_user_id", "guest"),
        "run_dir": format_path(result["run_dir"]),
        "video": format_path(Path(video_path)),
        "score": result["analysis"]["score"],
        "shooting_side": result["analysis"]["shooting_side"],
        "camera_view": result["analysis"]["camera_view"],
        "reliability": result["analysis"]["reliability"],
        "quality_warnings": result["analysis"].get("quality_warnings", []),
        "video_metadata": result["video_metadata"],
        "metrics": result["analysis"]["metrics"],
        "ball_tracking": result.get("ball_tracking", {}),
        "phases": result["analysis"]["phases"],
        "feedback": result["analysis"]["feedback"],
        "coaching_items": result["analysis"]["coaching_items"],
        "files": files,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as report_file:
        json.dump(report, report_file, indent=2)

    return report_path


def load_feature_rows(features_path: str | Path) -> dict[int, dict]:
    rows = {}
    with Path(features_path).open() as features_file:
        reader = csv.DictReader(features_file)
        for row in reader:
            try:
                rows[int(float(row["frame"]))] = row
            except (KeyError, TypeError, ValueError):
                continue

    return rows


def get_feature_row(feature_rows: dict[int, dict], frame_number: int) -> dict | None:
    if not feature_rows:
        return None

    if frame_number in feature_rows:
        return feature_rows[frame_number]

    closest_frame = min(feature_rows, key=lambda candidate: abs(candidate - frame_number))
    return feature_rows[closest_frame]


def point_from_row(row: dict, name: str, width: int, height: int) -> tuple[int, int] | None:
    try:
        x = float(row[f"{name}_x"])
        y = float(row[f"{name}_y"])
    except (KeyError, TypeError, ValueError):
        return None

    if x <= 0 and y <= 0:
        return None

    return int(x * width), int(y * height)


def draw_points_and_lines(frame, row: dict, point_names: list[str], color: tuple[int, int, int], label: str) -> None:
    height, width = frame.shape[:2]
    points = [point_from_row(row, name, width, height) for name in point_names]
    points = [point for point in points if point]
    if not points:
        return

    for first_point, second_point in zip(points, points[1:]):
        cv2.line(frame, first_point, second_point, color, 5, cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, point, 10, (12, 14, 18), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, point, 7, color, thickness=-1, lineType=cv2.LINE_AA)

    label_x = min(point[0] for point in points)
    label_y = max(76, min(point[1] for point in points) - 18)
    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 2, cv2.LINE_AA)


def draw_horizontal_reference(frame, row: dict, point_names: list[str], color: tuple[int, int, int], label: str) -> None:
    height, width = frame.shape[:2]
    points = [point_from_row(row, name, width, height) for name in point_names]
    points = [point for point in points if point]
    if not points:
        return

    y = int(sum(point[1] for point in points) / len(points))
    x1 = max(0, min(point[0] for point in points) - 42)
    x2 = min(width - 1, max(point[0] for point in points) + 42)
    cv2.line(frame, (x1, y), (x2, y), color, 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(76, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def draw_vertical_reference(frame, point: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
    height, _ = frame.shape[:2]
    x, y = point
    cv2.line(frame, (x, max(58, y - 80)), (x, min(height - 1, y + 80)), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x + 10, max(76, y - 62)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)


def draw_evidence_overlay(frame, row: dict | None, item: dict, shooting_side: str) -> None:
    if row is None:
        return

    metric = item.get("metric", "")
    side = "left" if shooting_side == "left" else "right"
    other_side = "right" if side == "left" else "left"
    highlight = (87, 255, 201)
    warning = (47, 100, 240)
    neutral = (255, 255, 255)
    height, width = frame.shape[:2]

    if metric in {"release_elbow_angle", "elbow_angle_std", "follow_through_frames", "follow_through_ratio"}:
        draw_points_and_lines(frame, row, [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"], highlight, "Shooting arm")
        wrist = point_from_row(row, f"{side}_wrist", width, height)
        if wrist:
            draw_vertical_reference(frame, wrist, warning, "finish line")
        return

    if metric in {"min_knee_angle", "hip_rise", "ankle_lift"}:
        draw_points_and_lines(frame, row, [f"{side}_hip", f"{side}_knee", f"{side}_ankle"], highlight, "Leg drive")
        draw_points_and_lines(frame, row, [f"{other_side}_hip", f"{other_side}_knee", f"{other_side}_ankle"], neutral, "Balance leg")
        draw_horizontal_reference(frame, row, [f"{side}_ankle", f"{other_side}_ankle"], warning, "ankle line")
        return

    if metric in {"left_shin_vertical_error", "right_shin_vertical_error", "shin_parallel_error"}:
        draw_points_and_lines(frame, row, ["left_knee", "left_ankle"], highlight, "left shin")
        draw_points_and_lines(frame, row, ["right_knee", "right_ankle"], warning, "right shin")
        return

    if metric in {"foot_parallel_error", "left_foot_angle_to_floor", "right_foot_angle_to_floor"}:
        draw_points_and_lines(frame, row, ["left_heel", "left_foot_index"], highlight, "left foot")
        draw_points_and_lines(frame, row, ["right_heel", "right_foot_index"], warning, "right foot")
        draw_horizontal_reference(frame, row, ["left_foot_index", "right_foot_index"], neutral, "floor reference")
        return

    if metric in {"forearm_vertical_error", "follow_through_vertical_error"}:
        draw_points_and_lines(frame, row, [f"{side}_elbow", f"{side}_wrist"], highlight, "Forearm line")
        wrist = point_from_row(row, f"{side}_wrist", width, height)
        if wrist:
            draw_vertical_reference(frame, wrist, warning, "vertical target")
        return

    if metric == "body_lean":
        draw_points_and_lines(frame, row, ["left_shoulder", "right_shoulder"], highlight, "Shoulder line")
        draw_points_and_lines(frame, row, ["left_hip", "right_hip"], warning, "Hip line")
        return

    draw_points_and_lines(frame, row, [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"], highlight, "Focus area")


def save_coaching_frame_images(
    video_path: str | Path,
    features_path: str | Path,
    coaching_items: list[dict],
    output_dir: str | Path,
    shooting_side: str = "right",
) -> dict:
    frame_paths = {}
    feature_rows = load_feature_rows(features_path)
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        return frame_paths

    try:
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for index, item in enumerate(coaching_items, start=1):
            start_frame = item.get("start_frame")
            if start_frame is None:
                continue

            frame_number = max(1, int(start_frame))
            if total_frames:
                frame_number = min(frame_number, total_frames)

            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ok, frame = video_capture.read()
            if not ok or frame is None:
                continue

            title = item.get("title", "Improvement frame")
            frame_key = f"coaching_frame_{index:02d}"
            image_path = Path(output_dir) / f"{frame_key}.jpg"
            label = f"Priority {index}: {title} | Frame {frame_number}"
            feature_row = get_feature_row(feature_rows, frame_number)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (12, 14, 18), thickness=-1)
            cv2.putText(
                frame,
                label[:110],
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            draw_evidence_overlay(frame, feature_row, item, shooting_side)
            cv2.imwrite(str(image_path), frame)

            item["evidence_frame_file"] = frame_key
            item["evidence_frame_path"] = format_path(image_path)
            item["evidence_frame_number"] = frame_number
            frame_paths[frame_key] = image_path
    finally:
        video_capture.release()

    return frame_paths


def analyze_video(
    video_path: str,
    save_output: bool = False,
    save_chart: bool = False,
    save_annotated_video: bool = False,
    save_json_report: bool = False,
    display: bool = False,
    run_dir: str | Path | None = None,
    copy_input: bool = True,
    camera_view: str = "side",
    shooting_side: str = "auto",
    owner_user_id: str = "guest",
) -> dict:
    run_paths = create_analysis_run(video_path, run_dir)
    source_video_path = Path(video_path)
    input_video_path = run_paths["input_video"]

    if copy_input:
        shutil.copy2(source_video_path, input_video_path)
    else:
        input_video_path = source_video_path

    video_result = read_video(
        str(input_video_path),
        show_pose=True,
        save_output=save_output,
        save_keypoints=True,
        display=display,
        verbose=False,
        output_path=run_paths["pose_video"],
        keypoints_path=run_paths["keypoints"],
    )

    keypoints_path = video_result["keypoints_path"]
    features_path = extract_features(str(keypoints_path), str(run_paths["features"]))
    analysis = analyze_shot(
        str(features_path),
        camera_view=camera_view,
        shooting_side=shooting_side,
        input_quality=video_result["metadata"],
    )
    ball_tracking = track_ball(
        str(input_video_path),
        str(features_path),
        str(run_paths["ball_tracking"]),
        analysis["shooting_side"],
        analysis["metrics"].get("release_frame"),
    )
    analysis["ball_tracking"] = ball_tracking
    analysis["metrics"].update(
        {
            "ball_tracking_status": ball_tracking["status"],
            "ball_detector_backend": ball_tracking["detector_backend"],
            "ball_visibility_ratio": ball_tracking["visibility_ratio"],
            "ball_close_visibility_ratio": ball_tracking["close_visibility_ratio"],
            "ball_release_frame": ball_tracking["ball_release_frame"],
            "ball_release_frame_delta": ball_tracking["release_frame_delta"],
            "ball_arc_height": ball_tracking["arc_height"],
            "ball_avg_wrist_distance": ball_tracking["avg_wrist_distance"],
        }
    )
    add_ball_based_insights(analysis, ball_tracking, run_paths["ball_tracking"])
    chart_path = (
        plot_angles(str(features_path), output_path=str(run_paths["angles_chart"]), phases=analysis["phases"])
        if save_chart
        else None
    )
    follow_through_debug_chart_path = (
        plot_follow_through_debug(
            str(features_path),
            analysis,
            output_path=str(run_paths["follow_through_debug_chart"]),
        )
        if save_chart
        else None
    )
    annotated_video_path = (
        annotate_video(
            str(input_video_path),
            str(features_path),
            output_path=str(run_paths["annotated_video"]),
            camera_view=camera_view,
            shooting_side=shooting_side,
            input_quality=video_result["metadata"],
            ball_tracking_csv_path=run_paths["ball_tracking"],
        )
        if save_annotated_video
        else None
    )
    coaching_frame_paths = save_coaching_frame_images(
        input_video_path,
        features_path,
        analysis["coaching_items"],
        run_paths["output_dir"],
        shooting_side=analysis["shooting_side"],
    )

    result = {
        "run_id": run_paths["run_dir"].name,
        "owner_user_id": owner_user_id,
        "run_dir": run_paths["run_dir"],
        "input_video_path": input_video_path,
        "keypoints_path": keypoints_path,
        "features_path": features_path,
        "ball_tracking_path": run_paths["ball_tracking"],
        "chart_path": chart_path,
        "follow_through_debug_chart_path": follow_through_debug_chart_path,
        "output_path": video_result["output_path"],
        "annotated_video_path": annotated_video_path,
        "coaching_frame_paths": coaching_frame_paths,
        "video_metadata": video_result["metadata"],
        "ball_tracking": ball_tracking,
        "analysis": analysis,
        "report_path": None,
    }
    result["report_path"] = save_report(str(input_video_path), result, run_paths["report"]) if save_json_report else None

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full basketball shot analysis pipeline.")
    parser.add_argument("video_path", help="Path to a video file, for example videos/ft1.mp4")
    parser.add_argument("--save-output", action="store_true", help="Save a video with the pose skeleton")
    parser.add_argument("--save-chart", action="store_true", help="Save a PNG chart of the extracted angles")
    parser.add_argument("--save-annotated-video", action="store_true", help="Save a video with pose skeleton and analysis overlays")
    parser.add_argument("--save-report", action="store_true", help="Save a JSON report with metrics, feedback, and file paths")
    parser.add_argument("--display", action="store_true", help="Show the processed video while analyzing")
    parser.add_argument("--camera-view", choices=["side", "front", "back"], default="side", help="Camera angle used for view-specific scoring")
    parser.add_argument("--shooting-side", choices=["auto", "right", "left"], default="auto", help="Shooting hand override")
    args = parser.parse_args()

    result = analyze_video(
        args.video_path,
        save_output=args.save_output,
        save_chart=args.save_chart,
        save_annotated_video=args.save_annotated_video,
        save_json_report=args.save_report,
        display=args.display,
        camera_view=args.camera_view,
        shooting_side=args.shooting_side,
    )

    print(f"Saved keypoints CSV: {result['keypoints_path']}")
    print(f"Saved features CSV: {result['features_path']}")
    print(f"Saved ball tracking CSV: {result['ball_tracking_path']}")
    if result["chart_path"]:
        print(f"Saved angles chart: {result['chart_path']}")
    if result["follow_through_debug_chart_path"]:
        print(f"Saved follow-through debug chart: {result['follow_through_debug_chart_path']}")
    if result["output_path"]:
        print(f"Saved output video: {result['output_path']}")
    if result["annotated_video_path"]:
        print(f"Saved annotated video: {result['annotated_video_path']}")
    if result["report_path"]:
        print(f"Saved JSON report: {result['report_path']}")
    print(f"Ball tracking: {result['ball_tracking']['status']} ({result['ball_tracking']['detected_frames']}/{result['ball_tracking']['total_frames']} frames)")
    print()
    print_report(result["analysis"])


if __name__ == "__main__":
    main()
