import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from analyze_video import analyze_video  # noqa: E402


@dataclass(frozen=True)
class QACase:
    video: str
    camera_view: str
    label: str
    expected_kind: str
    expected_shooting_side: str | None = None
    min_score: int = 60
    max_score: int | None = None
    min_reliability: int = 75


QA_CASES = [
    QACase("videos/ft10.mp4", "front", "AI front view - FT10", "good", expected_shooting_side="right"),
    QACase("videos/ft3.mp4", "side", "AI side view - FT3", "good"),
    QACase("videos/ft6.mp4", "back", "AI back view - FT6", "good"),
]


def severity_count(items: list[dict], severity: str) -> int:
    return sum(1 for item in items if item.get("severity") == severity)


def warning_titles(warnings: list[dict]) -> str:
    titles = [warning.get("title", "Warning") for warning in warnings]
    return ", ".join(titles) if titles else "-"


def coaching_titles(items: list[dict], limit: int = 3) -> str:
    titles = [item.get("title", "Improvement") for item in items[:limit]]
    return " | ".join(titles) if titles else "-"


def format_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value * 100:.1f}%"


def evaluate_case(case: QACase, analysis: dict, ball_tracking: dict) -> tuple[bool, list[str]]:
    failures = []
    score = analysis.get("score") or 0
    reliability_score = (analysis.get("reliability") or {}).get("score") or 0
    quality_warnings = analysis.get("quality_warnings") or []
    high_warning_count = severity_count(quality_warnings, "high")
    camera_view = analysis.get("camera_view")
    shooting_side = analysis.get("shooting_side")

    if camera_view != case.camera_view:
        failures.append(f"camera view {camera_view!r} != expected {case.camera_view!r}")

    if case.expected_shooting_side and shooting_side != case.expected_shooting_side:
        failures.append(f"shooting side {shooting_side!r} != expected {case.expected_shooting_side!r}")

    if case.expected_kind == "good":
        if score < case.min_score:
            failures.append(f"score {score} < minimum {case.min_score}")
        if reliability_score < case.min_reliability:
            failures.append(f"reliability {reliability_score} < minimum {case.min_reliability}")
        if high_warning_count:
            failures.append(f"{high_warning_count} high quality warning(s) on a good-input video")
    else:
        if case.max_score is not None and score > case.max_score:
            failures.append(f"bad input score {score} > maximum {case.max_score}")
        if reliability_score >= case.min_reliability and not quality_warnings:
            failures.append("bad input was not flagged by reliability or quality warnings")

    if ball_tracking.get("status") == "error":
        failures.append("ball tracking errored")

    return not failures, failures


def summarize_case(case: QACase, result: dict, passed: bool, failures: list[str]) -> dict:
    analysis = result["analysis"]
    metrics = analysis.get("metrics") or {}
    ball_tracking = result.get("ball_tracking") or {}
    quality_warnings = analysis.get("quality_warnings") or []
    coaching_items = analysis.get("coaching_items") or []

    return {
        "case": case.label,
        "video": case.video,
        "expected_kind": case.expected_kind,
        "expected_camera_view": case.camera_view,
        "score": analysis.get("score"),
        "camera_view": analysis.get("camera_view"),
        "shooting_side": analysis.get("shooting_side"),
        "shooting_side_confidence": metrics.get("shooting_side_confidence"),
        "reliability_score": (analysis.get("reliability") or {}).get("score"),
        "quality_warning_count": len(quality_warnings),
        "high_warning_count": severity_count(quality_warnings, "high"),
        "quality_warnings": warning_titles(quality_warnings),
        "top_improvements": coaching_titles(coaching_items),
        "ball_backend": ball_tracking.get("detector_backend"),
        "ball_visibility": ball_tracking.get("visibility_ratio"),
        "ball_close_visibility": ball_tracking.get("close_visibility_ratio"),
        "ball_release_frame": ball_tracking.get("ball_release_frame"),
        "passed": passed,
        "failures": failures,
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "Status",
        "Case",
        "Score",
        "Rel",
        "View",
        "Hand",
        "Warnings",
        "Ball",
        "Top improvements",
    ]
    table_rows = []
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        table_rows.append(
            [
                status,
                row["case"],
                str(row["score"]),
                str(row["reliability_score"]),
                row["camera_view"],
                row["shooting_side"],
                row["quality_warnings"],
                format_percent(row["ball_close_visibility"]),
                row["top_improvements"],
            ]
        )

    widths = [
        max(len(str(value)) for value in [header, *[row[index] for row in table_rows]])
        for index, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def run_case(case: QACase, owner_user_id: str) -> dict:
    video_path = PROJECT_ROOT / case.video
    if not video_path.exists():
        return {
            "case": case.label,
            "video": case.video,
            "passed": False,
            "failures": [f"missing video file: {case.video}"],
        }

    with tempfile.TemporaryDirectory(prefix="basketball_qa_") as tmp_dir:
        result = analyze_video(
            str(video_path),
            save_output=False,
            save_chart=False,
            save_annotated_video=False,
            save_json_report=False,
            display=False,
            run_dir=Path(tmp_dir) / Path(case.video).stem,
            copy_input=False,
            camera_view=case.camera_view,
            shooting_side="auto",
            owner_user_id=owner_user_id,
        )

    passed, failures = evaluate_case(case, result["analysis"], result.get("ball_tracking") or {})
    return summarize_case(case, result, passed, failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model QA checks against known basketball shot videos.")
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.video for case in QA_CASES],
        help="Run only a specific case. Can be provided more than once.",
    )
    parser.add_argument(
        "--ball-backend",
        choices=["env", "auto", "heuristic", "yolo", "roboflow"],
        default="heuristic",
        help="Ball detector backend for QA. Default is heuristic for fast, offline checks.",
    )
    parser.add_argument("--json-output", help="Optional path to save full QA results as JSON.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit with code 0, even when QA cases fail.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ball_backend != "env":
        os.environ["BALL_DETECTION_BACKEND"] = args.ball_backend

    selected_cases = QA_CASES
    if args.case:
        selected = set(args.case)
        selected_cases = [case for case in QA_CASES if case.video in selected]

    print(f"Running {len(selected_cases)} QA case(s)", flush=True)
    print(f"Ball backend: {os.getenv('BALL_DETECTION_BACKEND', 'env/default')}", flush=True)
    print(flush=True)

    rows = []
    for case in selected_cases:
        print(f"Analyzing {case.label} ({case.video}, {case.camera_view})...", flush=True)
        try:
            rows.append(run_case(case, owner_user_id="model-qa"))
        except Exception as exc:
            rows.append(
                {
                    "case": case.label,
                    "video": case.video,
                    "passed": False,
                    "failures": [str(exc)],
                }
            )

    print()
    print_table(rows)

    failures = [row for row in rows if not row.get("passed")]
    if failures:
        print()
        print("Failures:")
        for row in failures:
            print(f"- {row['case']}: {'; '.join(row.get('failures') or ['unknown failure'])}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print()
        print(f"Saved JSON QA report: {output_path}")

    print()
    print(f"Result: {len(rows) - len(failures)} passed, {len(failures)} failed")
    return 0 if args.no_fail or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
