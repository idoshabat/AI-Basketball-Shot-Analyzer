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

    OUTPUT_DIR.mkdir(exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create angle charts from extracted shot features.")
    parser.add_argument("features_csv_path", help="Path to a features CSV, for example data/ft1_features.csv")
    parser.add_argument("--output", help="Optional output PNG path")
    args = parser.parse_args()

    chart_path = plot_angles(args.features_csv_path, args.output)
    print(f"Saved angles chart: {chart_path}")


if __name__ == "__main__":
    main()
