import argparse
import json
from pathlib import Path

from feature_extractor import extract_features
from shot_analyzer import analyze_shot, print_report
from video_annotator import annotate_video
from visualize_features import plot_angles
from video_reader import read_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


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


def save_report(video_path: str, result: dict) -> Path:
    report_path = build_report_path(video_path)
    report = {
        "video": format_path(Path(video_path)),
        "score": result["analysis"]["score"],
        "shooting_side": result["analysis"]["shooting_side"],
        "video_metadata": result["video_metadata"],
        "metrics": result["analysis"]["metrics"],
        "phases": result["analysis"]["phases"],
        "feedback": result["analysis"]["feedback"],
        "coaching_items": result["analysis"]["coaching_items"],
        "files": {
            "keypoints_csv": format_path(result["keypoints_path"]),
            "features_csv": format_path(result["features_path"]),
            "angles_chart": format_path(result["chart_path"]),
            "pose_video": format_path(result["output_path"]),
            "annotated_video": format_path(result["annotated_video_path"]),
        },
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with report_path.open("w") as report_file:
        json.dump(report, report_file, indent=2)

    return report_path


def analyze_video(
    video_path: str,
    save_output: bool = False,
    save_chart: bool = False,
    save_annotated_video: bool = False,
    save_json_report: bool = False,
    display: bool = False,
) -> dict:
    video_result = read_video(
        video_path,
        show_pose=True,
        save_output=save_output,
        save_keypoints=True,
        display=display,
        verbose=False,
    )

    keypoints_path = video_result["keypoints_path"]
    features_path = extract_features(str(keypoints_path))
    analysis = analyze_shot(str(features_path))
    chart_path = plot_angles(str(features_path), phases=analysis["phases"]) if save_chart else None
    annotated_video_path = annotate_video(video_path, str(features_path)) if save_annotated_video else None

    result = {
        "keypoints_path": keypoints_path,
        "features_path": features_path,
        "chart_path": chart_path,
        "output_path": video_result["output_path"],
        "annotated_video_path": annotated_video_path,
        "video_metadata": video_result["metadata"],
        "analysis": analysis,
        "report_path": None,
    }
    result["report_path"] = save_report(video_path, result) if save_json_report else None

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full basketball shot analysis pipeline.")
    parser.add_argument("video_path", help="Path to a video file, for example videos/ft1.mp4")
    parser.add_argument("--save-output", action="store_true", help="Save a video with the pose skeleton")
    parser.add_argument("--save-chart", action="store_true", help="Save a PNG chart of the extracted angles")
    parser.add_argument("--save-annotated-video", action="store_true", help="Save a video with pose skeleton and analysis overlays")
    parser.add_argument("--save-report", action="store_true", help="Save a JSON report with metrics, feedback, and file paths")
    parser.add_argument("--display", action="store_true", help="Show the processed video while analyzing")
    args = parser.parse_args()

    result = analyze_video(
        args.video_path,
        save_output=args.save_output,
        save_chart=args.save_chart,
        save_annotated_video=args.save_annotated_video,
        save_json_report=args.save_report,
        display=args.display,
    )

    print(f"Saved keypoints CSV: {result['keypoints_path']}")
    print(f"Saved features CSV: {result['features_path']}")
    if result["chart_path"]:
        print(f"Saved angles chart: {result['chart_path']}")
    if result["output_path"]:
        print(f"Saved output video: {result['output_path']}")
    if result["annotated_video_path"]:
        print(f"Saved annotated video: {result['annotated_video_path']}")
    if result["report_path"]:
        print(f"Saved JSON report: {result['report_path']}")
    print()
    print_report(result["analysis"])


if __name__ == "__main__":
    main()
