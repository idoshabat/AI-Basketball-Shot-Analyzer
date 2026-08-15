import argparse
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def resolve_data_path(csv_path: str) -> Path:
    path = Path(csv_path)
    if path.exists():
        return path

    project_path = PROJECT_ROOT / csv_path
    if project_path.exists():
        return project_path

    raise FileNotFoundError(f"CSV file not found: {csv_path}")


def build_features_path(keypoints_path: Path) -> Path:
    filename = keypoints_path.name.replace("_keypoints.csv", "_features.csv")
    if filename == keypoints_path.name:
        filename = f"{keypoints_path.stem}_features.csv"

    return DATA_DIR / filename


def calculate_angle(point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float]) -> float:
    vector_ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    vector_bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])

    dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]
    magnitude_ba = math.hypot(vector_ba[0], vector_ba[1])
    magnitude_bc = math.hypot(vector_bc[0], vector_bc[1])

    if magnitude_ba == 0 or magnitude_bc == 0:
        return math.nan

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))


def get_point(row: pd.Series, landmark_name: str) -> tuple[float, float]:
    return row[f"{landmark_name}_x"], row[f"{landmark_name}_y"]


def get_coordinate(row: pd.Series, landmark_name: str, axis: str) -> float:
    return row[f"{landmark_name}_{axis}"]


def extract_frame_features(row: pd.Series) -> dict:
    return {
        "frame": int(row["frame"]),
        "right_shoulder_x": get_coordinate(row, "right_shoulder", "x"),
        "right_wrist_x": get_coordinate(row, "right_wrist", "x"),
        "right_wrist_y": get_coordinate(row, "right_wrist", "y"),
        "right_elbow_x": get_coordinate(row, "right_elbow", "x"),
        "right_elbow_y": get_coordinate(row, "right_elbow", "y"),
        "left_shoulder_x": get_coordinate(row, "left_shoulder", "x"),
        "left_wrist_x": get_coordinate(row, "left_wrist", "x"),
        "left_wrist_y": get_coordinate(row, "left_wrist", "y"),
        "left_elbow_x": get_coordinate(row, "left_elbow", "x"),
        "left_elbow_y": get_coordinate(row, "left_elbow", "y"),
        "right_shoulder_y": get_coordinate(row, "right_shoulder", "y"),
        "left_shoulder_y": get_coordinate(row, "left_shoulder", "y"),
        "right_hip_x": get_coordinate(row, "right_hip", "x"),
        "right_hip_y": get_coordinate(row, "right_hip", "y"),
        "left_hip_x": get_coordinate(row, "left_hip", "x"),
        "left_hip_y": get_coordinate(row, "left_hip", "y"),
        "right_knee_x": get_coordinate(row, "right_knee", "x"),
        "right_knee_y": get_coordinate(row, "right_knee", "y"),
        "left_knee_x": get_coordinate(row, "left_knee", "x"),
        "left_knee_y": get_coordinate(row, "left_knee", "y"),
        "right_ankle_x": get_coordinate(row, "right_ankle", "x"),
        "right_ankle_y": get_coordinate(row, "right_ankle", "y"),
        "right_heel_x": get_coordinate(row, "right_heel", "x"),
        "right_heel_y": get_coordinate(row, "right_heel", "y"),
        "right_foot_index_x": get_coordinate(row, "right_foot_index", "x"),
        "right_foot_index_y": get_coordinate(row, "right_foot_index", "y"),
        "left_ankle_x": get_coordinate(row, "left_ankle", "x"),
        "left_ankle_y": get_coordinate(row, "left_ankle", "y"),
        "left_heel_x": get_coordinate(row, "left_heel", "x"),
        "left_heel_y": get_coordinate(row, "left_heel", "y"),
        "left_foot_index_x": get_coordinate(row, "left_foot_index", "x"),
        "left_foot_index_y": get_coordinate(row, "left_foot_index", "y"),
        "right_elbow_angle": calculate_angle(
            get_point(row, "right_shoulder"),
            get_point(row, "right_elbow"),
            get_point(row, "right_wrist"),
        ),
        "left_elbow_angle": calculate_angle(
            get_point(row, "left_shoulder"),
            get_point(row, "left_elbow"),
            get_point(row, "left_wrist"),
        ),
        "right_knee_angle": calculate_angle(
            get_point(row, "right_hip"),
            get_point(row, "right_knee"),
            get_point(row, "right_ankle"),
        ),
        "left_knee_angle": calculate_angle(
            get_point(row, "left_hip"),
            get_point(row, "left_knee"),
            get_point(row, "left_ankle"),
        ),
    }


def extract_features(keypoints_csv_path: str, output_csv_path: str | None = None) -> Path:
    keypoints_path = resolve_data_path(keypoints_csv_path)
    output_path = Path(output_csv_path) if output_csv_path else build_features_path(keypoints_path)

    df = pd.read_csv(keypoints_path)
    features = [extract_frame_features(row) for _, row in df.iterrows()]
    features_df = pd.DataFrame(features)
    features_df["right_wrist_y_velocity"] = features_df["right_wrist_y"].diff().fillna(0.0)
    features_df["left_wrist_y_velocity"] = features_df["left_wrist_y"].diff().fillna(0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract shot features from pose keypoints.")
    parser.add_argument("keypoints_csv_path", help="Path to a keypoints CSV, for example data/ft1_keypoints.csv")
    parser.add_argument("--output", help="Optional output CSV path")
    args = parser.parse_args()

    output_path = extract_features(args.keypoints_csv_path, args.output)
    print(f"Saved features CSV: {output_path}")


if __name__ == "__main__":
    main()
