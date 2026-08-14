import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import pandas as pd


def resolve_features_path(csv_path: str) -> Path:
    path = Path(csv_path)
    if path.exists():
        return path

    project_path = PROJECT_ROOT / csv_path
    if project_path.exists():
        return project_path

    raise FileNotFoundError(f"Features CSV file not found: {csv_path}")


def build_chart_path(features_path: Path) -> Path:
    filename = features_path.name.replace("_features.csv", "_angles.png")
    if filename == features_path.name:
        filename = f"{features_path.stem}_angles.png"

    return OUTPUT_DIR / filename


def build_follow_through_debug_path(features_path: Path) -> Path:
    filename = features_path.name.replace("_features.csv", "_follow_through_debug.png")
    if filename == features_path.name:
        filename = f"{features_path.stem}_follow_through_debug.png"

    return OUTPUT_DIR / filename


PHASE_COLORS = {
    "setup": "#b0b0b0",
    "dip_load": "#f4a261",
    "upward_motion": "#2a9d8f",
    "release": "#e76f51",
    "follow_through": "#457b9d",
    "recovery": "#8d99ae",
}


def shade_phases(phases: dict | None) -> None:
    if not phases:
        return

    for phase_name, phase in phases.items():
        start_frame = phase["start_frame"]
        end_frame = phase["end_frame"]
        color = PHASE_COLORS.get(phase_name, "#cccccc")
        label = phase_name.replace("_", " ").title()

        plt.axvspan(start_frame, end_frame, color=color, alpha=0.15, label=label)

        if phase_name == "release":
            plt.axvline(start_frame, color=color, linestyle="--", linewidth=2)


def plot_angles(features_csv_path: str, output_path: str | None = None, phases: dict | None = None) -> Path:
    features_path = resolve_features_path(features_csv_path)
    chart_path = Path(output_path) if output_path else build_chart_path(features_path)

    df = pd.read_csv(features_path)
    angle_columns = [
        "right_elbow_angle",
        "left_elbow_angle",
        "right_knee_angle",
        "left_knee_angle",
    ]

    missing_columns = [column for column in angle_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {', '.join(missing_columns)}")

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    MPL_CACHE_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 7))
    shade_phases(phases)

    for column in angle_columns:
        plt.plot(df["frame"], df[column], label=column.replace("_", " ").title())

    plt.title("Basketball Shot Joint Angles")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.ylim(0, 180)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc="best")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    return chart_path


def plot_follow_through_debug(
    features_csv_path: str,
    analysis: dict,
    output_path: str | None = None,
) -> Path:
    features_path = resolve_features_path(features_csv_path)
    chart_path = Path(output_path) if output_path else build_follow_through_debug_path(features_path)
    df = pd.read_csv(features_path)
    shooting_side = analysis["shooting_side"]
    metrics = analysis["metrics"]
    phases = analysis["phases"]

    wrist_y_column = f"{shooting_side}_wrist_y"
    shoulder_y_column = f"{shooting_side}_shoulder_y"
    elbow_angle_column = f"{shooting_side}_elbow_angle"
    missing_columns = [
        column
        for column in (wrist_y_column, shoulder_y_column, elbow_angle_column)
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {', '.join(missing_columns)}")

    release_frame = metrics["release_frame"]
    release_method = metrics.get("release_detection_method", "unknown")
    release_confidence = metrics.get("release_confidence")
    release_confidence_label = metrics.get("release_confidence_label", "unknown")
    follow_through_end_frame = metrics["follow_through_end_frame"]
    smoothed_wrist_y = df[wrist_y_column].rolling(window=5, center=True, min_periods=1).mean()
    smoothed_shoulder_y = df[shoulder_y_column].rolling(window=5, center=True, min_periods=1).mean()
    smoothed_elbow_angle = df[elbow_angle_column].rolling(window=5, center=True, min_periods=1).mean()
    valid_posture = (smoothed_elbow_angle >= 155) & (smoothed_wrist_y <= smoothed_shoulder_y - 0.03)

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    MPL_CACHE_DIR.mkdir(exist_ok=True)

    figure, (height_axis, angle_axis) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    release_label = "Release"
    if release_confidence is not None:
        release_label = f"Release ({release_confidence:.2f}, {release_confidence_label})"

    follow_through_phase = phases.get("follow_through")
    if follow_through_phase:
        height_axis.axvspan(
            follow_through_phase["start_frame"],
            follow_through_phase["end_frame"],
            color="#457b9d",
            alpha=0.12,
            label="Detected Follow-Through",
        )
        angle_axis.axvspan(
            follow_through_phase["start_frame"],
            follow_through_phase["end_frame"],
            color="#457b9d",
            alpha=0.12,
        )

    height_axis.fill_between(
        df["frame"],
        smoothed_wrist_y,
        smoothed_shoulder_y,
        where=valid_posture,
        color="#2a9d8f",
        alpha=0.18,
        label="Valid Posture",
    )
    height_axis.plot(df["frame"], smoothed_wrist_y, color="#e76f51", label="Shooting Wrist Y")
    height_axis.plot(df["frame"], smoothed_shoulder_y, color="#264653", label="Shooting Shoulder Y")
    height_axis.invert_yaxis()
    height_axis.set_ylabel("Normalized Y (higher on court is lower value)")
    height_axis.grid(True, alpha=0.3)
    height_axis.legend(loc="best")

    angle_axis.plot(df["frame"], smoothed_elbow_angle, color="#f4a261", label="Shooting Elbow Angle")
    angle_axis.axhline(155, color="#888888", linestyle=":", linewidth=1.5, label="Extension Threshold")
    angle_axis.set_ylim(0, 190)
    angle_axis.set_xlabel("Frame")
    angle_axis.set_ylabel("Angle (degrees)")
    angle_axis.grid(True, alpha=0.3)
    angle_axis.legend(loc="best")

    for axis in (height_axis, angle_axis):
        axis.axvspan(release_frame - 0.5, release_frame + 0.5, color="#e76f51", alpha=0.16)
        axis.axvline(release_frame, color="#e76f51", linestyle="--", linewidth=2, label=release_label)
        if follow_through_end_frame is not None:
            axis.axvline(follow_through_end_frame, color="#6d597a", linestyle="--", linewidth=2)

    confidence_text = "Release confidence: N/A"
    if release_confidence is not None:
        confidence_text = f"Release confidence: {release_confidence:.2f} ({release_confidence_label})"
    height_axis.text(
        0.01,
        0.04,
        f"{confidence_text}\nMethod: {release_method}",
        transform=height_axis.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#e76f51", "alpha": 0.88},
        fontsize=9,
        verticalalignment="bottom",
    )

    figure.suptitle(
        f"Follow-Through Debug ({shooting_side.title()} Side) - "
        f"Release {release_frame}, {release_confidence_label.title()} Confidence - "
        f"End: {follow_through_end_frame or 'not detected'}"
    )
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)

    return chart_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create angle charts from extracted shot features.")
    parser.add_argument("features_csv_path", help="Path to a features CSV, for example data/ft1_features.csv")
    parser.add_argument("--output", help="Optional output PNG path")
    parser.add_argument("--follow-through-debug", action="store_true", help="Create a follow-through debug chart")
    args = parser.parse_args()

    if args.follow_through_debug:
        from shot_analyzer import analyze_shot

        chart_path = plot_follow_through_debug(args.features_csv_path, analyze_shot(args.features_csv_path), args.output)
    else:
        chart_path = plot_angles(args.features_csv_path, args.output)
    print(f"Saved angles chart: {chart_path}")


if __name__ == "__main__":
    main()
