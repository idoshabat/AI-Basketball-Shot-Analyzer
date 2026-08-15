import argparse
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_CAMERA_VIEWS = {"side", "front", "back"}


def resolve_features_path(csv_path: str) -> Path:
    path = Path(csv_path)
    if path.exists():
        return path

    project_path = PROJECT_ROOT / csv_path
    if project_path.exists():
        return project_path

    raise FileNotFoundError(f"Features CSV file not found: {csv_path}")


def get_shooting_side(df: pd.DataFrame) -> str:
    right_max = df["right_elbow_angle"].max()
    left_max = df["left_elbow_angle"].max()

    return "right" if right_max >= left_max else "left"


def score_elbow_extension(max_elbow_angle: float) -> tuple[int, str]:
    if max_elbow_angle >= 165:
        return 25, "Good shooting arm extension."
    if max_elbow_angle >= 145:
        return 18, "Decent arm extension, but try finishing a little taller."
    return 10, "Shooting arm extension looks limited."


def score_knee_bend(min_knee_angle: float) -> tuple[int, str]:
    if 95 <= min_knee_angle <= 130:
        return 20, "Good knee bend and leg loading."
    if 130 < min_knee_angle <= 150:
        return 14, "Knee bend is a bit shallow."
    return 10, "Knee bend timing or depth needs work."


def score_consistency(elbow_angle_std: float) -> tuple[int, str]:
    if elbow_angle_std <= 25:
        return 15, "Arm motion looks fairly consistent."
    if elbow_angle_std <= 45:
        return 10, "Arm motion has some variation."
    return 6, "Arm motion looks very inconsistent across the shot."


def score_release_extension(release_elbow_angle: float) -> tuple[int, str]:
    if release_elbow_angle >= 160:
        return 10, "Release happens with a strong arm extension."
    if release_elbow_angle >= 140:
        return 7, "Release arm angle is acceptable, but could be more extended."
    return 3, "Release appears early, before the shooting arm fully extends."


def measure_follow_through(
    df: pd.DataFrame,
    shooting_side: str,
    release_frame: int,
    release_wrist_y: float | None,
    smoothing_window: int = 5,
    min_extended_elbow_angle: float = 155,
    wrist_above_shoulder_margin: float = 0.03,
    sustained_loss_frames: int = 4,
    minimum_hold_frames: int = 5,
) -> dict:
    wrist_y_column = f"{shooting_side}_wrist_y"
    elbow_angle_column = f"{shooting_side}_elbow_angle"
    shoulder_y_column = f"{shooting_side}_shoulder_y"
    if release_wrist_y is None or wrist_y_column not in df.columns:
        return {
            "follow_through_frames": 0,
            "follow_through_window": 0,
            "follow_through_ratio": 0.0,
            "follow_through_end_frame": None,
            "follow_through_wrist_drop": 0.0,
            "follow_through_drop_from_best": 0.0,
            "follow_through_status": "missing_wrist_data",
            "follow_through_end_reason": "missing_wrist_data",
        }

    follow_through_df = df[df["frame"] > release_frame]

    if follow_through_df.empty:
        return {
            "follow_through_frames": 0,
            "follow_through_window": 0,
            "follow_through_ratio": 0.0,
            "follow_through_end_frame": None,
            "follow_through_wrist_drop": 0.0,
            "follow_through_drop_from_best": 0.0,
            "follow_through_status": "no_frames_after_release",
            "follow_through_end_reason": "no_frames_after_release",
        }

    follow_through_df = follow_through_df.copy()
    follow_through_df["smoothed_wrist_y"] = (
        follow_through_df[wrist_y_column]
        .rolling(window=smoothing_window, center=True, min_periods=1)
        .mean()
    )
    follow_through_df["drop_from_best_wrist_y"] = (
        follow_through_df["smoothed_wrist_y"]
        - follow_through_df["smoothed_wrist_y"].cummin()
    )

    has_posture_data = elbow_angle_column in follow_through_df.columns and shoulder_y_column in follow_through_df.columns
    if has_posture_data:
        follow_through_df["smoothed_elbow_angle"] = (
            follow_through_df[elbow_angle_column]
            .rolling(window=smoothing_window, center=True, min_periods=1)
            .mean()
        )
        follow_through_df["smoothed_shoulder_y"] = (
            follow_through_df[shoulder_y_column]
            .rolling(window=smoothing_window, center=True, min_periods=1)
            .mean()
        )
        follow_through_df["arm_extended"] = (
            follow_through_df["smoothed_elbow_angle"] >= min_extended_elbow_angle
        )
        follow_through_df["wrist_above_shoulder"] = (
            follow_through_df["smoothed_wrist_y"]
            <= follow_through_df["smoothed_shoulder_y"] - wrist_above_shoulder_margin
        )
        follow_through_df["follow_through_posture"] = (
            follow_through_df["arm_extended"]
            & follow_through_df["wrist_above_shoulder"]
        )
        follow_through_df["posture_lost"] = ~follow_through_df["follow_through_posture"]
        follow_through_df["posture_loss_streak"] = (
            follow_through_df["posture_lost"]
            .rolling(window=sustained_loss_frames, min_periods=sustained_loss_frames)
            .sum()
        )
        end_candidates = follow_through_df[
            (follow_through_df["frame"] >= release_frame + minimum_hold_frames)
            & (follow_through_df["posture_loss_streak"] >= sustained_loss_frames)
        ]
        follow_through_end_reason = "sustained_posture_loss"
    else:
        follow_through_df["recent_wrist_y_rise"] = (
            follow_through_df["smoothed_wrist_y"]
            - follow_through_df["smoothed_wrist_y"].shift(smoothing_window - 1)
        )
        end_candidates = follow_through_df[
            (follow_through_df["drop_from_best_wrist_y"] >= 0.08)
            & (follow_through_df["recent_wrist_y_rise"] >= 0.02)
        ]
        follow_through_end_reason = "sustained_wrist_drop_fallback"

    if end_candidates.empty:
        follow_through_end_frame = None
        follow_through_frames = len(follow_through_df)
        wrist_drop_reference = follow_through_df.iloc[-1][wrist_y_column]
        drop_from_best = follow_through_df.iloc[-1]["drop_from_best_wrist_y"]
        follow_through_status = "held_until_video_end"
        follow_through_end_reason = "held_until_video_end"
    else:
        end_row = end_candidates.iloc[0]
        follow_through_end_frame = int(end_row["frame"] - sustained_loss_frames + 1)
        follow_through_frames = len(follow_through_df[follow_through_df["frame"] < follow_through_end_frame])
        end_reference_rows = follow_through_df[follow_through_df["frame"] >= follow_through_end_frame]
        end_reference_row = end_reference_rows.iloc[0] if not end_reference_rows.empty else end_row
        wrist_drop_reference = end_reference_row[wrist_y_column]
        drop_from_best = end_reference_row["drop_from_best_wrist_y"]
        follow_through_status = "ended"

    follow_through_window = len(follow_through_df)

    return {
        "follow_through_frames": follow_through_frames,
        "follow_through_window": follow_through_window,
        "follow_through_ratio": follow_through_frames / follow_through_window,
        "follow_through_end_frame": follow_through_end_frame,
        "follow_through_wrist_drop": max(0.0, wrist_drop_reference - release_wrist_y),
        "follow_through_drop_from_best": max(0.0, drop_from_best),
        "follow_through_status": follow_through_status,
        "follow_through_end_reason": follow_through_end_reason,
    }


def score_follow_through(follow_through_frames: int) -> tuple[int, str]:
    if follow_through_frames >= 10:
        return 10, "Good follow-through: shooting hand holds near max height after release."
    if follow_through_frames >= 5:
        return 6, "Follow-through is present, but the shooting hand drops a bit early."
    return 3, "Follow-through looks short; try holding your shooting hand higher after release."


def average_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    existing_columns = [column for column in columns if column in df.columns]
    if not existing_columns:
        return None

    return df[existing_columns].mean(axis=1)


def normalize_series(series: pd.Series, reverse: bool = False) -> pd.Series:
    value_range = series.max() - series.min()
    if value_range == 0:
        return pd.Series(0.0, index=series.index)

    normalized = (series - series.min()) / value_range
    return 1.0 - normalized if reverse else normalized


def measure_leg_drive(df: pd.DataFrame, release_frame: int, shooting_side: str) -> dict:
    hip_y = average_columns(df, ["left_hip_y", "right_hip_y"])
    if hip_y is None:
        return {
            "hip_load_frame": None,
            "hip_deepest_frame": None,
            "knee_load_frame": None,
            "dip_load_start_frame": None,
            "dip_load_end_frame": None,
            "hip_load_y": None,
            "release_hip_y": None,
            "hip_rise": 0.0,
            "load_score": 0.0,
            "load_detection_method": "missing_hip_data",
        }

    working_df = df.copy()
    working_df["hip_y"] = hip_y
    knee_column = f"{shooting_side}_knee_angle"
    if knee_column not in working_df.columns:
        knee_y = average_columns(working_df, ["left_knee_angle", "right_knee_angle"])
        working_df["load_knee_angle"] = knee_y
    else:
        working_df["load_knee_angle"] = working_df[knee_column]

    pre_release_df = working_df[working_df["frame"] <= release_frame]
    if pre_release_df.empty:
        pre_release_df = working_df

    pre_release_df = pre_release_df.copy()
    pre_release_df["smoothed_hip_y"] = (
        pre_release_df["hip_y"]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )
    pre_release_df["smoothed_knee_angle"] = (
        pre_release_df["load_knee_angle"]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )
    pre_release_df["hip_depth_score"] = normalize_series(pre_release_df["smoothed_hip_y"])
    pre_release_df["knee_bend_score"] = normalize_series(pre_release_df["smoothed_knee_angle"], reverse=True)
    pre_release_df["load_score"] = (
        0.45 * pre_release_df["hip_depth_score"]
        + 0.55 * pre_release_df["knee_bend_score"]
    )

    load_row = pre_release_df.loc[pre_release_df["load_score"].idxmax()]
    hip_deepest_row = pre_release_df.loc[pre_release_df["smoothed_hip_y"].idxmax()]
    knee_load_row = pre_release_df.loc[pre_release_df["smoothed_knee_angle"].idxmin()]
    release_row = working_df.loc[working_df["frame"] == release_frame].iloc[0]
    hip_rise = load_row["hip_y"] - release_row["hip_y"]
    first_frame = int(working_df["frame"].min())
    load_frame = int(load_row["frame"])
    dip_load_start_frame = max(first_frame, load_frame - 4)
    dip_load_end_frame = min(release_frame - 1, load_frame + 6)

    return {
        "hip_load_frame": load_frame,
        "hip_deepest_frame": int(hip_deepest_row["frame"]),
        "knee_load_frame": int(knee_load_row["frame"]),
        "dip_load_start_frame": dip_load_start_frame,
        "dip_load_end_frame": dip_load_end_frame,
        "hip_load_y": load_row["hip_y"],
        "release_hip_y": release_row["hip_y"],
        "hip_rise": max(0.0, hip_rise),
        "load_score": load_row["load_score"],
        "load_detection_method": "combined_hip_depth_and_knee_bend",
    }


def measure_jump(df: pd.DataFrame, release_frame: int, baseline_frames: int = 10, window_size: int = 15) -> dict:
    ankle_y = average_columns(df, ["left_ankle_y", "right_ankle_y"])
    if ankle_y is None:
        return {
            "ankle_baseline_y": None,
            "min_jump_ankle_y": None,
            "ankle_lift": 0.0,
            "jump_window_start": None,
            "jump_window_end": None,
        }

    working_df = df.copy()
    working_df["ankle_y"] = ankle_y
    baseline_df = working_df.head(baseline_frames)
    jump_window_start = max(int(working_df["frame"].min()), release_frame - 5)
    jump_window_end = min(int(working_df["frame"].max()), release_frame + window_size)
    jump_df = working_df[
        (working_df["frame"] >= jump_window_start)
        & (working_df["frame"] <= jump_window_end)
    ]

    ankle_baseline_y = baseline_df["ankle_y"].median()
    min_jump_ankle_y = jump_df["ankle_y"].min()
    ankle_lift = ankle_baseline_y - min_jump_ankle_y

    return {
        "ankle_baseline_y": ankle_baseline_y,
        "min_jump_ankle_y": min_jump_ankle_y,
        "ankle_lift": max(0.0, ankle_lift),
        "jump_window_start": jump_window_start,
        "jump_window_end": jump_window_end,
    }


def score_leg_drive(hip_rise: float) -> tuple[int, str]:
    if hip_rise >= 0.08:
        return 10, "Good upward leg drive into the release."
    if hip_rise >= 0.04:
        return 6, "Some upward leg drive is visible, but it could be stronger."
    return 3, "Leg drive looks limited; try loading and rising more into the shot."


def score_jump(ankle_lift: float) -> tuple[int, str]:
    if ankle_lift >= 0.05:
        return 10, "Clear jump lift is visible from the ankle movement."
    if ankle_lift >= 0.02:
        return 6, "Small jump lift is visible."
    return 3, "Jump lift looks limited based on ankle height change."


def normalize_camera_view(camera_view: str) -> str:
    view = (camera_view or "side").strip().lower()
    if view not in SUPPORTED_CAMERA_VIEWS:
        raise ValueError(f"Unsupported camera view: {camera_view}. Choose one of: side, front, back.")

    return view


def safe_column_value(row: pd.Series, column: str) -> float | None:
    if column not in row or pd.isna(row[column]):
        return None

    return float(row[column])


def get_body_width(row: pd.Series) -> float:
    shoulder_width = abs(row.get("right_shoulder_x", 0.0) - row.get("left_shoulder_x", 0.0))
    hip_width = abs(row.get("right_hip_x", 0.0) - row.get("left_hip_x", 0.0))

    return max(shoulder_width, hip_width, 0.001)


def line_angle_to_floor(
    start_x: float | None,
    start_y: float | None,
    end_x: float | None,
    end_y: float | None,
) -> float | None:
    if None in (start_x, start_y, end_x, end_y):
        return None

    dx = end_x - start_x
    dy = end_y - start_y
    if dx == 0 and dy == 0:
        return None

    angle = abs(math.degrees(math.atan2(dy, dx)))
    return 180 - angle if angle > 180 else angle


def vertical_error(angle_to_floor: float | None) -> float | None:
    if angle_to_floor is None:
        return None

    return abs(90 - angle_to_floor)


def horizontal_error(angle_to_floor: float | None) -> float | None:
    if angle_to_floor is None:
        return None

    return min(angle_to_floor, abs(180 - angle_to_floor))


def median_line_angle_to_floor(df: pd.DataFrame, start_prefix: str, end_prefix: str) -> float | None:
    angles = []
    for _, row in df.iterrows():
        angle = line_angle_to_floor(
            safe_column_value(row, f"{start_prefix}_x"),
            safe_column_value(row, f"{start_prefix}_y"),
            safe_column_value(row, f"{end_prefix}_x"),
            safe_column_value(row, f"{end_prefix}_y"),
        )
        if angle is not None:
            angles.append(angle)

    if not angles:
        return None

    return float(pd.Series(angles).median())


def measure_alignment(df: pd.DataFrame, shooting_side: str, release_frame: int) -> dict:
    required_columns = [
        "right_shoulder_x",
        "left_shoulder_x",
        "right_hip_x",
        "left_hip_x",
        "right_knee_x",
        "left_knee_x",
        "right_ankle_x",
        "left_ankle_x",
    ]
    if any(column not in df.columns for column in required_columns):
        return {
            "shoulder_level_delta": 0.0,
            "body_lean": 0.0,
            "arm_stack_error": None,
            "knee_to_ankle_alignment_error": 0.0,
            "leg_symmetry_error": 0.0,
            "left_shin_angle_to_floor": None,
            "right_shin_angle_to_floor": None,
            "left_shin_vertical_error": None,
            "right_shin_vertical_error": None,
            "shin_parallel_error": None,
            "left_foot_angle_to_floor": None,
            "right_foot_angle_to_floor": None,
            "left_foot_floor_error": None,
            "right_foot_floor_error": None,
            "foot_parallel_error": None,
            "forearm_angle_to_floor": None,
            "forearm_vertical_error": None,
            "follow_through_line_angle_to_floor": None,
            "follow_through_vertical_error": None,
            "alignment_status": "missing_landmark_x_features",
        }

    release_rows = df[df["frame"] == release_frame]
    release_row = release_rows.iloc[0] if not release_rows.empty else df.iloc[len(df) // 2]
    body_width = get_body_width(release_row)

    shoulder_level_delta = abs(release_row["right_shoulder_y"] - release_row["left_shoulder_y"])
    hip_center_x = (release_row["right_hip_x"] + release_row["left_hip_x"]) / 2
    shoulder_center_x = (release_row["right_shoulder_x"] + release_row["left_shoulder_x"]) / 2
    body_lean = abs(shoulder_center_x - hip_center_x) / body_width

    wrist_x = safe_column_value(release_row, f"{shooting_side}_wrist_x")
    elbow_x = safe_column_value(release_row, f"{shooting_side}_elbow_x")
    shoulder_x = safe_column_value(release_row, f"{shooting_side}_shoulder_x")
    if wrist_x is None or elbow_x is None or shoulder_x is None:
        arm_stack_error = None
    else:
        arm_stack_error = max(abs(wrist_x - elbow_x), abs(elbow_x - shoulder_x)) / body_width

    stance_df = df.head(min(15, len(df)))
    left_knee_offset = (stance_df["left_knee_x"] - stance_df["left_ankle_x"]).abs().median()
    right_knee_offset = (stance_df["right_knee_x"] - stance_df["right_ankle_x"]).abs().median()
    knee_to_ankle_alignment_error = float((left_knee_offset + right_knee_offset) / (2 * body_width))

    left_leg_width = abs(release_row["left_hip_x"] - release_row["left_ankle_x"])
    right_leg_width = abs(release_row["right_hip_x"] - release_row["right_ankle_x"])
    leg_symmetry_error = abs(left_leg_width - right_leg_width) / body_width
    left_shin_angle = line_angle_to_floor(
        safe_column_value(release_row, "left_ankle_x"),
        safe_column_value(release_row, "left_ankle_y"),
        safe_column_value(release_row, "left_knee_x"),
        safe_column_value(release_row, "left_knee_y"),
    )
    right_shin_angle = line_angle_to_floor(
        safe_column_value(release_row, "right_ankle_x"),
        safe_column_value(release_row, "right_ankle_y"),
        safe_column_value(release_row, "right_knee_x"),
        safe_column_value(release_row, "right_knee_y"),
    )
    left_shin_vertical_error = vertical_error(left_shin_angle)
    right_shin_vertical_error = vertical_error(right_shin_angle)
    shin_parallel_error = (
        None
        if left_shin_angle is None or right_shin_angle is None
        else abs(left_shin_angle - right_shin_angle)
    )
    left_foot_angle = line_angle_to_floor(
        safe_column_value(release_row, "left_heel_x"),
        safe_column_value(release_row, "left_heel_y"),
        safe_column_value(release_row, "left_foot_index_x"),
        safe_column_value(release_row, "left_foot_index_y"),
    )
    right_foot_angle = line_angle_to_floor(
        safe_column_value(release_row, "right_heel_x"),
        safe_column_value(release_row, "right_heel_y"),
        safe_column_value(release_row, "right_foot_index_x"),
        safe_column_value(release_row, "right_foot_index_y"),
    )
    foot_parallel_error = (
        None
        if left_foot_angle is None or right_foot_angle is None
        else abs(left_foot_angle - right_foot_angle)
    )
    left_foot_floor_error = horizontal_error(left_foot_angle)
    right_foot_floor_error = horizontal_error(right_foot_angle)
    forearm_angle = line_angle_to_floor(
        safe_column_value(release_row, f"{shooting_side}_elbow_x"),
        safe_column_value(release_row, f"{shooting_side}_elbow_y"),
        safe_column_value(release_row, f"{shooting_side}_wrist_x"),
        safe_column_value(release_row, f"{shooting_side}_wrist_y"),
    )
    forearm_vertical_error = vertical_error(forearm_angle)
    follow_through_window = df[
        (df["frame"] > release_frame)
        & (df["frame"] <= min(int(df["frame"].max()), release_frame + 12))
    ]
    follow_through_line_angle = median_line_angle_to_floor(
        follow_through_window,
        f"{shooting_side}_elbow",
        f"{shooting_side}_wrist",
    )
    follow_through_vertical_error = vertical_error(follow_through_line_angle)

    return {
        "shoulder_level_delta": float(shoulder_level_delta),
        "body_lean": float(body_lean),
        "arm_stack_error": None if arm_stack_error is None else float(arm_stack_error),
        "knee_to_ankle_alignment_error": knee_to_ankle_alignment_error,
        "leg_symmetry_error": float(leg_symmetry_error),
        "left_shin_angle_to_floor": left_shin_angle,
        "right_shin_angle_to_floor": right_shin_angle,
        "left_shin_vertical_error": left_shin_vertical_error,
        "right_shin_vertical_error": right_shin_vertical_error,
        "shin_parallel_error": shin_parallel_error,
        "left_foot_angle_to_floor": left_foot_angle,
        "right_foot_angle_to_floor": right_foot_angle,
        "left_foot_floor_error": left_foot_floor_error,
        "right_foot_floor_error": right_foot_floor_error,
        "foot_parallel_error": foot_parallel_error,
        "forearm_angle_to_floor": forearm_angle,
        "forearm_vertical_error": forearm_vertical_error,
        "follow_through_line_angle_to_floor": follow_through_line_angle,
        "follow_through_vertical_error": follow_through_vertical_error,
        "alignment_status": "measured",
    }


def score_alignment(metrics: dict) -> tuple[int, list[str]]:
    if metrics.get("alignment_status") == "missing_landmark_x_features":
        return 30, ["Alignment metrics need newly extracted landmark x-coordinates for this camera view."]

    score = 0
    feedback = []

    shin_error = max(
        metrics["left_shin_vertical_error"] or 90,
        metrics["right_shin_vertical_error"] or 90,
    )
    if shin_error <= 10 and (metrics["shin_parallel_error"] or 0) <= 10:
        score += 15
        feedback.append("Both lower legs stay close to vertical and parallel through the release.")
    elif shin_error <= 18 and (metrics["shin_parallel_error"] or 0) <= 18:
        score += 10
        feedback.append("Lower-leg lines are decent, but one knee/ankle line drifts slightly.")
    else:
        score += 5
        feedback.append("Lower-leg alignment is off: the knee-to-foot lines are not parallel and vertical enough.")

    foot_parallel_error = metrics.get("foot_parallel_error")
    foot_floor_error = max(
        metrics.get("left_foot_floor_error") or 90,
        metrics.get("right_foot_floor_error") or 90,
    )
    if foot_parallel_error is None:
        score += 6
        feedback.append("Foot direction could not be measured clearly from heel/toe landmarks.")
    elif foot_parallel_error <= 12 and foot_floor_error <= 22:
        score += 15
        feedback.append("Feet look mostly parallel and square from this front/back view.")
    elif foot_parallel_error <= 24:
        score += 10
        feedback.append("Feet are not perfectly parallel; check that both feet point the same direction.")
    else:
        score += 5
        feedback.append("Feet are not parallel enough, which can pull the shot line off target.")

    forearm_error = metrics.get("forearm_vertical_error")
    if forearm_error is None:
        score += 6
        feedback.append("Forearm angle could not be measured clearly at release.")
    elif forearm_error <= 12:
        score += 15
        feedback.append("Forearm is close to vertical at release.")
    elif forearm_error <= 24:
        score += 10
        feedback.append("Forearm is slightly angled at release; try keeping elbow and wrist more stacked.")
    else:
        score += 5
        feedback.append("Forearm angle is too tilted at release; wrist should finish more directly over the elbow.")

    follow_through_error = metrics.get("follow_through_vertical_error")
    if follow_through_error is None:
        score += 6
        feedback.append("Follow-through line could not be measured clearly after release.")
    elif follow_through_error <= 12:
        score += 15
        feedback.append("Follow-through line stays straight toward the basket.")
    elif follow_through_error <= 24:
        score += 10
        feedback.append("Follow-through line is close, but the hand path drifts slightly off center.")
    else:
        score += 5
        feedback.append("Follow-through line drifts sideways instead of staying straight to the basket.")

    if metrics["body_lean"] <= 0.18 and metrics["shoulder_level_delta"] <= 0.05:
        score += 15
        feedback.append("Torso and shoulders stay balanced from this angle.")
    elif metrics["body_lean"] <= 0.3:
        score += 10
        feedback.append("Torso balance is acceptable, with some lean or shoulder tilt.")
    else:
        score += 5
        feedback.append("Body lean is high; the shot may drift sideways.")

    return score, feedback


def build_coaching_item(
    title: str,
    severity: str,
    metric: str,
    value,
    target: str,
    why_it_matters: str,
    drill: str,
    phase: str,
    start_frame: int | None,
    end_frame: int | None,
) -> dict:
    return {
        "title": title,
        "severity": severity,
        "metric": metric,
        "value": value,
        "target": target,
        "why_it_matters": why_it_matters,
        "drill": drill,
        "phase": phase,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def get_phase_window(
    phases: dict,
    phase: str,
    fallback_start: int | None = None,
    fallback_end: int | None = None,
) -> tuple[str, int | None, int | None]:
    phase_data = phases.get(phase)
    if phase_data:
        return phase, phase_data["start_frame"], phase_data["end_frame"]

    return phase, fallback_start, fallback_end


def get_full_shot_window(phases: dict) -> tuple[int | None, int | None]:
    if not phases:
        return None, None

    start_frame = min(phase["start_frame"] for phase in phases.values())
    end_frame = max(phase["end_frame"] for phase in phases.values())
    return start_frame, end_frame


def build_coaching_items(metrics: dict, phases: dict, camera_view: str = "side") -> list[dict]:
    coaching_items = []
    release_phase = get_phase_window(
        phases,
        "release",
        metrics["release_frame"],
        metrics["release_frame"],
    )
    follow_through_phase = get_phase_window(
        phases,
        "follow_through",
        metrics["release_frame"] + 1,
        metrics["follow_through_end_frame"],
    )
    dip_load_phase = get_phase_window(
        phases,
        "dip_load",
        metrics["dip_load_start_frame"],
        metrics["dip_load_end_frame"],
    )
    upward_motion_phase = get_phase_window(
        phases,
        "upward_motion",
        metrics["dip_load_end_frame"],
        metrics["release_frame"],
    )
    full_shot_start, full_shot_end = get_full_shot_window(phases)
    alignment_phase = get_phase_window(
        phases,
        "release",
        metrics["release_frame"],
        metrics["release_frame"],
    )

    if camera_view in {"front", "back"}:
        shin_error = max(
            metrics.get("left_shin_vertical_error") or 0,
            metrics.get("right_shin_vertical_error") or 0,
        )
        if shin_error > 18 or (metrics.get("shin_parallel_error") or 0) > 18:
            coaching_items.append(
                build_coaching_item(
                    title="Make both knee-to-foot lines vertical",
                    severity="high",
                    metric="shin_parallel_error",
                    value=metrics.get("shin_parallel_error"),
                    target="Both shin lines within 18 degrees of vertical and parallel",
                    why_it_matters="From the front, the knees should track over the feet so the shot loads straight instead of drifting sideways.",
                    drill="Record front-view form shots and keep each knee moving directly over the same-side foot during the dip and release.",
                    phase="release",
                    start_frame=metrics["release_frame"],
                    end_frame=metrics["release_frame"],
                )
            )
        elif shin_error > 10 or (metrics.get("shin_parallel_error") or 0) > 10:
            coaching_items.append(
                build_coaching_item(
                    title="Clean up knee-to-foot alignment",
                    severity="medium",
                    metric="shin_parallel_error",
                    value=metrics.get("shin_parallel_error"),
                    target="Both shin lines within 10 degrees of vertical and parallel",
                    why_it_matters="A cleaner lower-body line keeps the base square and reduces sideways force.",
                    drill="Use slow front-view reps and pause at the load to check knees stacked over feet.",
                    phase="dip_load",
                    start_frame=metrics["dip_load_start_frame"],
                    end_frame=metrics["dip_load_end_frame"],
                )
            )

        foot_error = metrics.get("foot_parallel_error")
        if foot_error is not None and foot_error > 24:
            coaching_items.append(
                build_coaching_item(
                    title="Square the feet to the same line",
                    severity="high",
                    metric="foot_parallel_error",
                    value=foot_error,
                    target="24 degrees or lower between left and right foot direction",
                    why_it_matters="If the feet point in different directions, the hips and shoulders often rotate away from the shot line.",
                    drill="Mark a straight line on the floor and start each rep with both feet parallel to that line.",
                    phase="setup",
                    start_frame=full_shot_start,
                    end_frame=metrics["dip_load_start_frame"],
                )
            )
        elif foot_error is not None and foot_error > 12:
            coaching_items.append(
                build_coaching_item(
                    title="Make the feet more parallel",
                    severity="medium",
                    metric="foot_parallel_error",
                    value=foot_error,
                    target="12 degrees or lower between left and right foot direction",
                    why_it_matters="Parallel feet make it easier to load and release on the same line.",
                    drill="Take form shots with both toes set on the same floor line before every rep.",
                    phase="setup",
                    start_frame=full_shot_start,
                    end_frame=metrics["dip_load_start_frame"],
                )
            )

        forearm_error = metrics.get("forearm_vertical_error")
        if forearm_error is not None and forearm_error > 24:
            coaching_items.append(
                build_coaching_item(
                    title="Keep the forearm more vertical at release",
                    severity="high",
                    metric="forearm_vertical_error",
                    value=forearm_error,
                    target="24 degrees or less away from vertical",
                    why_it_matters="From the front, a tilted forearm usually means the wrist is not stacked over the elbow, which can create side spin.",
                    drill="Shoot one-hand form shots facing the camera and freeze with the wrist directly above the elbow.",
                    phase=alignment_phase[0],
                    start_frame=alignment_phase[1],
                    end_frame=alignment_phase[2],
                )
            )
        elif forearm_error is not None and forearm_error > 12:
            coaching_items.append(
                build_coaching_item(
                    title="Straighten the forearm line",
                    severity="medium",
                    metric="forearm_vertical_error",
                    value=forearm_error,
                    target="12 degrees or less away from vertical",
                    why_it_matters="A straighter forearm line helps the ball leave on a cleaner path.",
                    drill="Pause at release on close shots and check wrist-over-elbow alignment.",
                    phase=alignment_phase[0],
                    start_frame=alignment_phase[1],
                    end_frame=alignment_phase[2],
                )
            )

        follow_through_error = metrics.get("follow_through_vertical_error")
        if follow_through_error is not None and follow_through_error > 24:
            coaching_items.append(
                build_coaching_item(
                    title="Finish the follow-through on the shot line",
                    severity="high",
                    metric="follow_through_vertical_error",
                    value=follow_through_error,
                    target="24 degrees or less away from vertical",
                    why_it_matters="A sideways follow-through often means the hand is pushing across the ball instead of through the target.",
                    drill="Hold the finish and point the fingers straight through the center line after release.",
                    phase=follow_through_phase[0],
                    start_frame=follow_through_phase[1],
                    end_frame=follow_through_phase[2],
                )
            )
        elif follow_through_error is not None and follow_through_error > 12:
            coaching_items.append(
                build_coaching_item(
                    title="Reduce follow-through drift",
                    severity="medium",
                    metric="follow_through_vertical_error",
                    value=follow_through_error,
                    target="12 degrees or less away from vertical",
                    why_it_matters="A cleaner follow-through line makes the release direction more repeatable.",
                    drill="Shoot form reps and freeze the hand on the center line for one second.",
                    phase=follow_through_phase[0],
                    start_frame=follow_through_phase[1],
                    end_frame=follow_through_phase[2],
                )
            )

        if metrics.get("body_lean", 0) > 0.3:
            coaching_items.append(
                build_coaching_item(
                    title="Reduce side lean",
                    severity="medium",
                    metric="body_lean",
                    value=metrics["body_lean"],
                    target="0.30 or lower normalized torso lean",
                    why_it_matters="Side lean can push the release line off target.",
                    drill="Shoot balanced form reps and land with your head and hips stacked over the same base.",
                    phase="full_shot",
                    start_frame=full_shot_start,
                    end_frame=full_shot_end,
                )
            )

    if metrics["release_elbow_angle"] < 140:
        coaching_items.append(
            build_coaching_item(
                title="Release after fuller arm extension",
                severity="high",
                metric="release_elbow_angle",
                value=metrics["release_elbow_angle"],
                target="140 degrees or higher at release",
                why_it_matters="A more extended shooting arm usually gives the ball a cleaner path and a higher release point.",
                drill="Start close to the rim and pause with your elbow high and arm extended after every make.",
                phase=release_phase[0],
                start_frame=release_phase[1],
                end_frame=release_phase[2],
            )
        )
    elif metrics["release_elbow_angle"] < 160:
        coaching_items.append(
            build_coaching_item(
                title="Finish a little taller through release",
                severity="medium",
                metric="release_elbow_angle",
                value=metrics["release_elbow_angle"],
                target="160 degrees or higher at release",
                why_it_matters="A taller finish can make the shot less flat and harder to contest.",
                drill="Shoot one-hand form shots and hold the finish until the ball reaches the rim.",
                phase=release_phase[0],
                start_frame=release_phase[1],
                end_frame=release_phase[2],
            )
        )

    if metrics["follow_through_frames"] < 5:
        coaching_items.append(
            build_coaching_item(
                title="Hold the follow-through longer",
                severity="high",
                metric="follow_through_frames",
                value=metrics["follow_through_frames"],
                target="At least 5 frames after release",
                why_it_matters="Dropping the wrist early can disturb touch and make the release less repeatable.",
                drill="Use a hold-your-finish drill: release, freeze the wrist, then count one full second before relaxing.",
                phase=follow_through_phase[0],
                start_frame=follow_through_phase[1],
                end_frame=follow_through_phase[2],
            )
        )
    elif metrics["follow_through_frames"] < 10:
        coaching_items.append(
            build_coaching_item(
                title="Make the follow-through more stable",
                severity="medium",
                metric="follow_through_frames",
                value=metrics["follow_through_frames"],
                target="10 frames or more after release",
                why_it_matters="A stable finish helps repeat the same release under fatigue and pressure.",
                drill="Shoot five makes from each spot while keeping your shooting hand above eye level after release.",
                phase=follow_through_phase[0],
                start_frame=follow_through_phase[1],
                end_frame=follow_through_phase[2],
            )
        )

    if camera_view == "side" and metrics["hip_rise"] < 0.04:
        coaching_items.append(
            build_coaching_item(
                title="Use more leg drive into the shot",
                severity="high",
                metric="hip_rise",
                value=metrics["hip_rise"],
                target="0.04 or higher normalized hip rise",
                why_it_matters="Leg drive helps transfer power upward so the arm does not have to force the shot.",
                drill="Practice dip-and-rise form shots: dip, rise smoothly, and release on the way up.",
                phase=upward_motion_phase[0],
                start_frame=upward_motion_phase[1],
                end_frame=upward_motion_phase[2],
            )
        )
    elif camera_view == "side" and metrics["hip_rise"] < 0.08:
        coaching_items.append(
            build_coaching_item(
                title="Strengthen the upward push",
                severity="medium",
                metric="hip_rise",
                value=metrics["hip_rise"],
                target="0.08 or higher normalized hip rise",
                why_it_matters="A stronger upward push can improve range without changing the release.",
                drill="Take rhythm shots where the ball lift and leg rise start together.",
                phase=upward_motion_phase[0],
                start_frame=upward_motion_phase[1],
                end_frame=upward_motion_phase[2],
            )
        )

    if camera_view == "side" and metrics["ankle_lift"] < 0.02:
        coaching_items.append(
            build_coaching_item(
                title="Add a cleaner jump lift",
                severity="medium",
                metric="ankle_lift",
                value=metrics["ankle_lift"],
                target="0.02 or higher normalized ankle lift",
                why_it_matters="A visible lift can help timing and power, especially outside the paint.",
                drill="Shoot short jumpers focusing on landing in the same spot with balanced feet.",
                phase="jump_window",
                start_frame=metrics["jump_window_start"],
                end_frame=metrics["jump_window_end"],
            )
        )
    elif camera_view == "side" and metrics["ankle_lift"] < 0.05:
        coaching_items.append(
            build_coaching_item(
                title="Increase jump lift",
                severity="medium",
                metric="ankle_lift",
                value=metrics["ankle_lift"],
                target="0.05 or higher normalized ankle lift",
                why_it_matters="A little more lift can improve rhythm and make the shot easier to repeat from distance.",
                drill="Shoot short rhythm jumpers and focus on leaving the floor smoothly without drifting.",
                phase="jump_window",
                start_frame=metrics["jump_window_start"],
                end_frame=metrics["jump_window_end"],
            )
        )

    if camera_view == "side" and metrics["min_knee_angle"] > 150:
        coaching_items.append(
            build_coaching_item(
                title="Load the knees more before rising",
                severity="medium",
                metric="min_knee_angle",
                value=metrics["min_knee_angle"],
                target="130 degrees or lower at the deepest load",
                why_it_matters="A shallow knee load can make the shot rely too much on the upper body.",
                drill="Use chair-height rhythm reps: bend, rise, and shoot without pausing at the bottom.",
                phase=dip_load_phase[0],
                start_frame=dip_load_phase[1],
                end_frame=dip_load_phase[2],
            )
        )
    elif camera_view == "side" and metrics["min_knee_angle"] > 130:
        coaching_items.append(
            build_coaching_item(
                title="Get slightly deeper in the load",
                severity="medium",
                metric="min_knee_angle",
                value=metrics["min_knee_angle"],
                target="130 degrees or lower at the deepest load",
                why_it_matters="A slightly deeper knee load can create power earlier, so the release does not have to do all the work.",
                drill="Use rhythm reps from close range: dip smoothly, keep balance, and rise without pausing.",
                phase=dip_load_phase[0],
                start_frame=dip_load_phase[1],
                end_frame=dip_load_phase[2],
            )
        )

    if metrics["elbow_angle_std"] > 45:
        coaching_items.append(
            build_coaching_item(
                title="Make the arm path more repeatable",
                severity="medium",
                metric="elbow_angle_std",
                value=metrics["elbow_angle_std"],
                target="45 degrees or lower",
                why_it_matters="Large variation in the shooting arm makes the release timing harder to repeat.",
                drill="Shoot slow form reps from close range and keep the ball path centered every time.",
                phase="full_shot",
                start_frame=full_shot_start,
                end_frame=full_shot_end,
            )
        )
    elif metrics["elbow_angle_std"] > 25:
        coaching_items.append(
            build_coaching_item(
                title="Tighten arm motion consistency",
                severity="medium",
                metric="elbow_angle_std",
                value=metrics["elbow_angle_std"],
                target="25 degrees or lower",
                why_it_matters="Less variation in the shooting arm helps the release happen the same way each time.",
                drill="Shoot slow form reps and check that the ball path and elbow finish match on every shot.",
                phase="full_shot",
                start_frame=full_shot_start,
                end_frame=full_shot_end,
            )
        )

    if not coaching_items:
        coaching_items.append(
            build_coaching_item(
                title="Keep building repeatability",
                severity="low",
                metric="overall",
                value=None,
                target="Same mechanics across multiple shots",
                why_it_matters="The current shot has no major rule-based issues, so consistency becomes the priority.",
                drill="Record 10 shots from the same angle and compare release frame, follow-through, and leg drive.",
                phase="full_shot",
                start_frame=full_shot_start,
                end_frame=full_shot_end,
            )
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(coaching_items, key=lambda item: severity_rank.get(item["severity"], 3))


def add_phase(phases: dict, name: str, start_frame: int, end_frame: int | None) -> None:
    if end_frame is None or start_frame > end_frame:
        return

    phases[name] = {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
    }


def detect_phases(df: pd.DataFrame, release_frame: int, leg_drive: dict, follow_through: dict) -> dict:
    first_frame = int(df["frame"].min())
    last_frame = int(df["frame"].max())
    load_start = leg_drive["dip_load_start_frame"] or first_frame
    load_end = leg_drive["dip_load_end_frame"] or load_start
    follow_through_end = follow_through["follow_through_end_frame"] or last_frame

    phases = {}
    add_phase(phases, "setup", first_frame, load_start - 1)
    add_phase(phases, "dip_load", load_start, load_end)
    add_phase(phases, "upward_motion", load_end + 1, release_frame - 1)
    add_phase(phases, "release", release_frame, release_frame)
    add_phase(phases, "follow_through", release_frame + 1, follow_through_end - 1)

    if follow_through["follow_through_status"] == "ended":
        add_phase(phases, "recovery", follow_through_end, last_frame)

    return phases


def find_release_row(df: pd.DataFrame, shooting_side: str) -> tuple[pd.Series, str]:
    wrist_y_column = f"{shooting_side}_wrist_y"
    wrist_velocity_column = f"{shooting_side}_wrist_y_velocity"
    elbow_column = f"{shooting_side}_elbow_angle"

    if wrist_y_column not in df.columns:
        return df.loc[df[elbow_column].idxmax()], "max_elbow_fallback"

    highest_wrist_y = df[wrist_y_column].min()
    wrist_height_tolerance = 0.04
    extended_elbow_threshold = max(160, df[elbow_column].max() - 15)
    candidate_df = df[
        (df[elbow_column] >= extended_elbow_threshold)
        & (df[wrist_y_column] <= highest_wrist_y + wrist_height_tolerance)
    ].copy()

    if candidate_df.empty:
        return df.loc[df[wrist_y_column].idxmin()], "highest_wrist_fallback"

    if wrist_velocity_column not in candidate_df.columns:
        return candidate_df.iloc[0], "height_and_extension"

    candidate_df["previous_wrist_y_velocity"] = df[wrist_velocity_column].shift(1)
    release_candidates = candidate_df[
        (candidate_df["previous_wrist_y_velocity"] < 0)
        & (candidate_df[wrist_velocity_column] >= -0.005)
    ]

    if not release_candidates.empty:
        return release_candidates.iloc[0], "velocity_transition"

    stable_candidates = candidate_df[candidate_df[wrist_velocity_column].abs() <= 0.01]
    if not stable_candidates.empty:
        return stable_candidates.iloc[0], "stable_high_wrist"

    return candidate_df.iloc[0], "height_and_extension"


def label_confidence(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"

    return "low"


def calculate_release_confidence(
    df: pd.DataFrame,
    shooting_side: str,
    release_row: pd.Series,
    release_detection_method: str,
    phases: dict,
) -> tuple[float, str]:
    wrist_y_column = f"{shooting_side}_wrist_y"
    wrist_velocity_column = f"{shooting_side}_wrist_y_velocity"
    elbow_column = f"{shooting_side}_elbow_angle"
    release_frame = int(release_row["frame"])

    confidence = 0.0

    max_elbow_angle = df[elbow_column].max()
    release_elbow_angle = release_row[elbow_column]
    if max_elbow_angle > 0:
        extension_score = max(0.0, min(1.0, release_elbow_angle / max_elbow_angle))
        confidence += 0.30 * extension_score

    if wrist_y_column in df.columns:
        highest_wrist_y = df[wrist_y_column].min()
        wrist_range = max(0.001, df[wrist_y_column].max() - highest_wrist_y)
        wrist_height_score = 1.0 - min(1.0, max(0.0, (release_row[wrist_y_column] - highest_wrist_y) / (wrist_range * 0.35)))
        confidence += 0.25 * wrist_height_score

    method_scores = {
        "velocity_transition": 1.0,
        "stable_high_wrist": 0.8,
        "height_and_extension": 0.7,
        "highest_wrist_fallback": 0.45,
        "max_elbow_fallback": 0.35,
    }
    confidence += 0.25 * method_scores.get(release_detection_method, 0.4)

    if wrist_velocity_column in df.columns:
        release_velocity = abs(release_row[wrist_velocity_column])
        velocity_score = 1.0 - min(1.0, release_velocity / 0.03)
        confidence += 0.10 * velocity_score

    upward_motion = phases.get("upward_motion")
    release_phase = phases.get("release")
    if release_phase and release_phase["start_frame"] == release_frame:
        phase_score = 1.0
    elif upward_motion and upward_motion["start_frame"] <= release_frame <= upward_motion["end_frame"] + 3:
        phase_score = 0.8
    else:
        phase_score = 0.35
    confidence += 0.10 * phase_score

    confidence = round(max(0.0, min(1.0, confidence)), 2)
    return confidence, label_confidence(confidence)


def build_reliability_check(name: str, status: str, detail: str) -> dict:
    return {
        "name": name,
        "status": status,
        "detail": detail,
    }


def calculate_analysis_reliability(df: pd.DataFrame, camera_view: str, metrics: dict) -> dict:
    score = 100
    checks = []

    release_confidence = metrics.get("release_confidence", 0.0)
    if release_confidence >= 0.75:
        checks.append(build_reliability_check("Release detection", "pass", "Release timing confidence is high."))
    elif release_confidence >= 0.5:
        score -= 15
        checks.append(build_reliability_check("Release detection", "warning", "Release timing confidence is medium."))
    else:
        score -= 30
        checks.append(build_reliability_check("Release detection", "fail", "Release timing confidence is low."))

    if len(df) >= 60:
        checks.append(build_reliability_check("Video length", "pass", f"{len(df)} usable pose frames were analyzed."))
    elif len(df) >= 35:
        score -= 10
        checks.append(build_reliability_check("Video length", "warning", f"Only {len(df)} usable pose frames were analyzed."))
    else:
        score -= 25
        checks.append(build_reliability_check("Video length", "fail", f"Only {len(df)} usable pose frames were analyzed."))

    follow_window = metrics.get("follow_through_window", 0)
    if follow_window >= 20:
        checks.append(build_reliability_check("Post-release frames", "pass", "Enough frames exist after release."))
    elif follow_window >= 8:
        score -= 10
        checks.append(build_reliability_check("Post-release frames", "warning", "Post-release window is short."))
    else:
        score -= 20
        checks.append(build_reliability_check("Post-release frames", "fail", "Too few frames after release for strong follow-through judgment."))

    full_body_columns = [
        "left_shoulder_y",
        "right_shoulder_y",
        "left_hip_y",
        "right_hip_y",
        "left_ankle_y",
        "right_ankle_y",
    ]
    full_body_available = all(column in df.columns for column in full_body_columns)
    if full_body_available:
        checks.append(build_reliability_check("Full-body landmarks", "pass", "Upper body, hips, and ankles are available."))
    else:
        score -= 25
        checks.append(build_reliability_check("Full-body landmarks", "fail", "Some required full-body landmarks are missing."))

    if camera_view in {"front", "back"}:
        if metrics.get("alignment_status") == "measured":
            checks.append(build_reliability_check("Front/back line metrics", "pass", "Feet, shin, forearm, and follow-through line metrics were measured."))
        else:
            score -= 35
            checks.append(build_reliability_check("Front/back line metrics", "fail", "Front/back line metrics were not available. Re-analyze with the updated backend."))

        foot_metrics_available = metrics.get("foot_parallel_error") is not None
        if foot_metrics_available:
            checks.append(build_reliability_check("Foot landmarks", "pass", "Heel and toe landmarks were detected for foot direction."))
        else:
            score -= 15
            checks.append(build_reliability_check("Foot landmarks", "warning", "Heel/toe landmarks were not clear enough for foot direction."))
    else:
        checks.append(build_reliability_check("Camera-specific metrics", "pass", "Side-view biomechanics were used for this analysis."))

    score = max(0, min(100, round(score)))
    if score >= 80:
        label = "high"
    elif score >= 55:
        label = "medium"
    else:
        label = "low"

    return {
        "score": score,
        "label": label,
        "checks": checks,
    }


def analyze_shot(features_csv_path: str, camera_view: str = "side") -> dict:
    camera_view = normalize_camera_view(camera_view)
    path = resolve_features_path(features_csv_path)
    df = pd.read_csv(path).dropna()

    if df.empty:
        raise ValueError(f"No valid feature rows found in: {path}")

    shooting_side = get_shooting_side(df)
    elbow_column = f"{shooting_side}_elbow_angle"
    knee_column = f"{shooting_side}_knee_angle"

    max_elbow_angle = df[elbow_column].max()
    min_knee_angle = df[knee_column].min()
    elbow_angle_std = df[elbow_column].std()
    release_row, release_detection_method = find_release_row(df, shooting_side)
    release_frame = int(release_row["frame"])
    release_elbow_angle = release_row[elbow_column]
    release_knee_angle = release_row[knee_column]
    release_wrist_y_column = f"{shooting_side}_wrist_y"
    release_wrist_y = release_row.get(release_wrist_y_column)
    release_wrist_velocity = release_row.get(f"{shooting_side}_wrist_y_velocity")
    follow_through = measure_follow_through(df, shooting_side, release_frame, release_wrist_y)
    leg_drive = measure_leg_drive(df, release_frame, shooting_side)
    jump = measure_jump(df, release_frame)
    phases = detect_phases(df, release_frame, leg_drive, follow_through)
    alignment = measure_alignment(df, shooting_side, release_frame) if camera_view in {"front", "back"} else {}
    release_confidence, release_confidence_label = calculate_release_confidence(
        df,
        shooting_side,
        release_row,
        release_detection_method,
        phases,
    )

    elbow_score, elbow_feedback = score_elbow_extension(max_elbow_angle)
    knee_score, knee_feedback = score_knee_bend(min_knee_angle)
    consistency_score, consistency_feedback = score_consistency(elbow_angle_std)
    release_score, release_feedback = score_release_extension(release_elbow_angle)
    follow_through_score, follow_through_feedback = score_follow_through(follow_through["follow_through_frames"])
    leg_drive_score, leg_drive_feedback = score_leg_drive(leg_drive["hip_rise"])
    jump_score, jump_feedback = score_jump(jump["ankle_lift"])
    alignment_score, alignment_feedback = score_alignment(alignment) if alignment else (0, [])

    if camera_view == "side":
        score = (
            elbow_score
            + knee_score
            + consistency_score
            + release_score
            + follow_through_score
            + leg_drive_score
            + jump_score
        )
        feedback = [
            elbow_feedback,
            knee_feedback,
            consistency_feedback,
            release_feedback,
            follow_through_feedback,
            leg_drive_feedback,
            jump_feedback,
        ]
    else:
        score = (
            (alignment_score / 75) * 60
            + release_score
            + follow_through_score
            + min(10, consistency_score)
            + min(10, elbow_score)
        )
        feedback = [
            f"{camera_view.title()} view selected: grading emphasizes alignment metrics visible from this camera angle.",
            *alignment_feedback,
            release_feedback,
            follow_through_feedback,
            consistency_feedback,
        ]
    score = max(0, min(100, round(score)))

    metrics = {
        "max_elbow_angle": max_elbow_angle,
        "min_knee_angle": min_knee_angle,
        "elbow_angle_std": elbow_angle_std,
        "release_frame": release_frame,
        "release_detection_method": release_detection_method,
        "release_confidence": release_confidence,
        "release_confidence_label": release_confidence_label,
        "release_elbow_angle": release_elbow_angle,
        "release_knee_angle": release_knee_angle,
        "release_wrist_y": release_wrist_y,
        "release_wrist_y_velocity": release_wrist_velocity,
        **follow_through,
        **leg_drive,
        **jump,
        **alignment,
    }
    reliability = calculate_analysis_reliability(df, camera_view, metrics)

    return {
        "score": score,
        "shooting_side": shooting_side,
        "camera_view": camera_view,
        "reliability": reliability,
        "metrics": metrics,
        "phases": phases,
        "feedback": feedback,
        "coaching_items": build_coaching_items(metrics, phases, camera_view),
    }


def print_report(analysis: dict) -> None:
    metrics = analysis["metrics"]

    print("Shot Analysis")
    print(f"Score: {analysis['score']}/100")
    print(f"Shooting side: {analysis['shooting_side']}")
    print(f"Camera view: {analysis['camera_view']}")
    print()
    print("Metrics:")
    print(f"- Max elbow angle: {metrics['max_elbow_angle']:.1f}")
    print(f"- Min knee angle: {metrics['min_knee_angle']:.1f}")
    print(f"- Elbow angle standard deviation: {metrics['elbow_angle_std']:.1f}")
    print(f"- Release frame: {metrics['release_frame']}")
    print(f"- Release detection method: {metrics['release_detection_method']}")
    print(f"- Release confidence: {metrics['release_confidence']:.2f} ({metrics['release_confidence_label']})")
    print(f"- Release elbow angle: {metrics['release_elbow_angle']:.1f}")
    print(f"- Release knee angle: {metrics['release_knee_angle']:.1f}")
    if metrics["release_wrist_y"] is not None:
        print(f"- Release wrist y: {metrics['release_wrist_y']:.3f}")
    if metrics["release_wrist_y_velocity"] is not None:
        print(f"- Release wrist y velocity: {metrics['release_wrist_y_velocity']:.3f}")
    print(f"- Follow-through frames: {metrics['follow_through_frames']}/{metrics['follow_through_window']}")
    print(f"- Follow-through ratio: {metrics['follow_through_ratio']:.2f}")
    if metrics["follow_through_end_frame"] is None:
        print("- Follow-through end frame: not detected before video ended")
    else:
        print(f"- Follow-through end frame: {metrics['follow_through_end_frame']}")
    print(f"- Follow-through wrist drop: {metrics['follow_through_wrist_drop']:.3f}")
    print(f"- Follow-through drop from best height: {metrics['follow_through_drop_from_best']:.3f}")
    print(f"- Follow-through status: {metrics['follow_through_status']}")
    if metrics["hip_load_y"] is not None:
        print(f"- Hip load frame: {metrics['hip_load_frame']}")
        print(f"- Hip deepest frame: {metrics['hip_deepest_frame']}")
        print(f"- Knee load frame: {metrics['knee_load_frame']}")
        print(f"- Dip/load window: {metrics['dip_load_start_frame']}-{metrics['dip_load_end_frame']}")
        print(f"- Load detection method: {metrics['load_detection_method']}")
        print(f"- Hip rise into release: {metrics['hip_rise']:.3f}")
    if metrics["ankle_baseline_y"] is not None:
        print(f"- Ankle lift estimate: {metrics['ankle_lift']:.3f}")
        print(f"- Jump window: {metrics['jump_window_start']}-{metrics['jump_window_end']}")
    if "arm_stack_error" in metrics:
        print(f"- Left shin vertical error: {metrics['left_shin_vertical_error']:.1f} deg" if metrics["left_shin_vertical_error"] is not None else "- Left shin vertical error: N/A")
        print(f"- Right shin vertical error: {metrics['right_shin_vertical_error']:.1f} deg" if metrics["right_shin_vertical_error"] is not None else "- Right shin vertical error: N/A")
        print(f"- Shin parallel error: {metrics['shin_parallel_error']:.1f} deg" if metrics["shin_parallel_error"] is not None else "- Shin parallel error: N/A")
        print(f"- Foot parallel error: {metrics['foot_parallel_error']:.1f} deg" if metrics["foot_parallel_error"] is not None else "- Foot parallel error: N/A")
        print(f"- Forearm vertical error: {metrics['forearm_vertical_error']:.1f} deg" if metrics["forearm_vertical_error"] is not None else "- Forearm vertical error: N/A")
        print(f"- Follow-through line vertical error: {metrics['follow_through_vertical_error']:.1f} deg" if metrics["follow_through_vertical_error"] is not None else "- Follow-through line vertical error: N/A")
        print(f"- Arm stack error: {metrics['arm_stack_error']:.3f}" if metrics["arm_stack_error"] is not None else "- Arm stack error: N/A")
        print(f"- Knee-to-ankle alignment error: {metrics['knee_to_ankle_alignment_error']:.3f}")
        print(f"- Leg symmetry error: {metrics['leg_symmetry_error']:.3f}")
        print(f"- Shoulder level delta: {metrics['shoulder_level_delta']:.3f}")
        print(f"- Body lean: {metrics['body_lean']:.3f}")
    print()
    print("Phases:")
    for phase_name, phase in analysis["phases"].items():
        print(f"- {phase_name}: {phase['start_frame']}-{phase['end_frame']}")
    print()
    print("Feedback:")
    for item in analysis["feedback"]:
        print(f"- {item}")
    print()
    print("Coaching items:")
    for item in analysis["coaching_items"]:
        print(f"- [{item['severity']}] {item['title']}: {item['drill']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a basketball shot from extracted features.")
    parser.add_argument("features_csv_path", help="Path to a features CSV, for example data/ft1_features.csv")
    parser.add_argument("--camera-view", choices=["side", "front", "back"], default="side", help="Camera angle used for view-specific scoring")
    args = parser.parse_args()

    analysis = analyze_shot(args.features_csv_path, camera_view=args.camera_view)
    print_report(analysis)


if __name__ == "__main__":
    main()
