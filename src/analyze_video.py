import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

from feature_extractor import extract_features
from shot_analyzer import analyze_shot, print_report
from video_annotator import annotate_video
from visualize_features import plot_angles
from video_reader import read_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
STORAGE_DIR = PROJECT_ROOT / "storage"
ANALYSES_DIR = STORAGE_DIR / "analyses"


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
        "pose_video": run_path / "output" / "pose.mp4",
        "angles_chart": run_path / "output" / "angles.png",
        "annotated_video": run_path / "output" / "annotated.mp4",
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
    report = {
        "run_id": result["run_id"],
        "run_dir": format_path(result["run_dir"]),
        "video": format_path(Path(video_path)),
        "score": result["analysis"]["score"],
        "shooting_side": result["analysis"]["shooting_side"],
        "video_metadata": result["video_metadata"],
        "metrics": result["analysis"]["metrics"],
        "phases": result["analysis"]["phases"],
        "feedback": result["analysis"]["feedback"],
        "coaching_items": result["analysis"]["coaching_items"],
        "files": {
            "original_video": format_path(result["input_video_path"]),
            "keypoints_csv": format_path(result["keypoints_path"]),
            "features_csv": format_path(result["features_path"]),
            "angles_chart": format_path(result["chart_path"]),
            "pose_video": format_path(result["output_path"]),
            "annotated_video": format_path(result["annotated_video_path"]),
            "json_report": format_path(report_path),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
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
    run_dir: str | Path | None = None,
    copy_input: bool = True,
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
    analysis = analyze_shot(str(features_path))
    chart_path = (
        plot_angles(str(features_path), output_path=str(run_paths["angles_chart"]), phases=analysis["phases"])
        if save_chart
        else None
    )
    annotated_video_path = (
        annotate_video(str(input_video_path), str(features_path), output_path=str(run_paths["annotated_video"]))
        if save_annotated_video
        else None
    )

    result = {
        "run_id": run_paths["run_dir"].name,
        "run_dir": run_paths["run_dir"],
        "input_video_path": input_video_path,
        "keypoints_path": keypoints_path,
        "features_path": features_path,
        "chart_path": chart_path,
        "output_path": video_result["output_path"],
        "annotated_video_path": annotated_video_path,
        "video_metadata": video_result["metadata"],
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
