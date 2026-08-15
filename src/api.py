import os
import base64
import hashlib
import hmac
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
UPLOAD_DIR = PROJECT_ROOT / "videos" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"
STORAGE_DIR = PROJECT_ROOT / "storage"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://ai-basketball-shot-analyzer.vercel.app",
    "https://ai-basketball-shot-analyzer-edb2.vercel.app",
]
DEFAULT_CORS_ORIGIN_REGEX = r"https://.*\.vercel\.app|http://127\.0\.0\.1:5173|http://localhost:5173"
DEFAULT_ANALYSIS_RETENTION_DAYS = 7
GUEST_USER_ID = "guest"

sys.path.insert(0, str(SRC_DIR))

from analyze_video import ANALYSIS_VERSION, analyze_video, create_analysis_run, format_path
from shot_analyzer import normalize_camera_view
import supabase_store


def get_cors_origins() -> list[str]:
    env_origins = os.getenv("CORS_ORIGINS")
    if not env_origins:
        return DEFAULT_CORS_ORIGINS

    return [origin.strip().rstrip("/") for origin in env_origins.split(",") if origin.strip()]


def get_cors_origin_regex() -> str | None:
    return os.getenv("CORS_ORIGIN_REGEX", DEFAULT_CORS_ORIGIN_REGEX)


def get_analysis_retention_days() -> int:
    raw_value = os.getenv("ANALYSIS_RETENTION_DAYS")
    if raw_value is None:
        return DEFAULT_ANALYSIS_RETENTION_DAYS

    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_ANALYSIS_RETENTION_DAYS


def decode_base64url(value: str) -> bytes:
    padded_value = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded_value.encode())


def verify_supabase_jwt(token: str) -> dict:
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")

    try:
        header_part, payload_part, signature_part = token.split(".")
        header = json.loads(decode_base64url(header_part))
        payload = json.loads(decode_base64url(payload_part))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid auth token.") from exc

    if header.get("alg") != "HS256":
        return verify_supabase_token_with_auth_server(token)

    if not jwt_secret:
        return verify_supabase_token_with_auth_server(token)

    signed_content = f"{header_part}.{payload_part}".encode()
    expected_signature = hmac.new(jwt_secret.encode(), signed_content, hashlib.sha256).digest()
    actual_signature = decode_base64url(signature_part)
    if not hmac.compare_digest(expected_signature, actual_signature):
        raise HTTPException(status_code=401, detail="Invalid auth token signature.")

    expires_at = payload.get("exp")
    if isinstance(expires_at, (int, float)) and expires_at < time.time():
        raise HTTPException(status_code=401, detail="Auth token expired.")

    if not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Auth token is missing a user id.")

    return payload


def verify_supabase_token_with_auth_server(token: str) -> dict:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_anon_key:
        raise HTTPException(
            status_code=401,
            detail="Supabase asymmetric JWT verification requires SUPABASE_URL and SUPABASE_ANON_KEY on the backend.",
        )

    request = Request(
        f"{supabase_url.rstrip('/')}/auth/v1/user",
        headers={
            "apikey": supabase_anon_key,
            "Authorization": f"Bearer {token}",
        },
    )

    try:
        with urlopen(request, timeout=8) as response:
            user = json.loads(response.read().decode())
    except HTTPError as exc:
        raise HTTPException(status_code=401, detail="Invalid Supabase auth token.") from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Could not verify Supabase auth token.") from exc

    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Supabase auth token is missing a user id.")

    return {"sub": user_id, "email": user.get("email")}


def get_request_user_id(authorization: str | None = None) -> str:
    if not authorization:
        return GUEST_USER_ID

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header.")

    return verify_supabase_jwt(token)["sub"]


def report_belongs_to_user(report: dict, user_id: str) -> bool:
    return report.get("owner_user_id", GUEST_USER_ID) == user_id


OUTPUT_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
app = FastAPI(title="AI Basketball Shot Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_origin_regex=get_cors_origin_regex(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")


def cleanup_old_analyses(retention_days: int | None = None) -> int:
    retention_days = get_analysis_retention_days() if retention_days is None else retention_days
    analyses_dir = STORAGE_DIR / "analyses"
    if retention_days <= 0 or not analyses_dir.exists():
        return 0

    cutoff_timestamp = time.time() - (retention_days * 24 * 60 * 60)
    deleted_count = 0

    for run_dir in analyses_dir.iterdir():
        if not run_dir.is_dir():
            continue

        report_path = run_dir / "report.json"
        reference_path = report_path if report_path.exists() else run_dir
        if reference_path.stat().st_mtime >= cutoff_timestamp:
            continue

        try:
            run_dir.resolve().relative_to(analyses_dir.resolve())
        except ValueError:
            continue

        shutil.rmtree(run_dir)
        deleted_count += 1

    return deleted_count


@app.on_event("startup")
def cleanup_analyses_on_startup() -> None:
    deleted_count = cleanup_old_analyses()
    retention_days = get_analysis_retention_days()
    if retention_days > 0:
        print(f"Analysis cleanup removed {deleted_count} run(s). Retention: {retention_days} day(s).")


def validate_video_file(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    return suffix


def build_file_response(result: dict) -> dict:
    files = {
        "original_video": format_path(result["input_video_path"]),
        "keypoints_csv": format_path(result["keypoints_path"]),
        "features_csv": format_path(result["features_path"]),
        "angles_chart": format_path(result["chart_path"]),
        "follow_through_debug_chart": format_path(result["follow_through_debug_chart_path"]),
        "pose_video": format_path(result["output_path"]),
        "annotated_video": format_path(result["annotated_video_path"]),
        "json_report": format_path(result["report_path"]),
    }

    output_urls = {}
    for name, path in files.items():
        if path and (path.startswith("output/") or path.startswith("storage/")):
            output_urls[name] = f"/{path}"

    return {
        "files": files,
        "output_urls": output_urls,
    }


def build_urls_from_files(files: dict) -> dict:
    output_urls = {}
    for name, path in files.items():
        if path and (path.startswith("output/") or path.startswith("storage/")):
            output_urls[name] = f"/{path}"

    return output_urls


def build_output_urls(report: dict, files: dict) -> dict:
    output_urls = build_urls_from_files(files)
    output_urls.update(report.get("output_urls", {}))
    return output_urls


def get_report_path(run_id: str) -> Path:
    report_path = STORAGE_DIR / "analyses" / run_id / "report.json"
    try:
        report_path.resolve().relative_to((STORAGE_DIR / "analyses").resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run id.") from exc

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Analysis run not found.")

    return report_path


def get_run_dir(run_id: str) -> Path:
    run_dir = STORAGE_DIR / "analyses" / run_id
    try:
        run_dir.resolve().relative_to((STORAGE_DIR / "analyses").resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run id.") from exc

    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Analysis run not found.")

    return run_dir


def load_local_report(run_id: str) -> dict:
    report_path = get_report_path(run_id)
    with report_path.open() as report_file:
        return json.load(report_file)


def load_saved_reports() -> list[dict]:
    analyses_dir = STORAGE_DIR / "analyses"
    if not analyses_dir.exists():
        return []

    reports = []
    for report_path in analyses_dir.glob("*/report.json"):
        with report_path.open() as report_file:
            reports.append(json.load(report_file))

    return reports


def hydrate_supabase_report(report: dict) -> dict:
    if report.get("storage_files"):
        return supabase_store.add_signed_output_urls(report)

    return report


def load_reports_for_user(user_id: str, limit: int | None = None) -> list[dict]:
    if supabase_store.is_configured():
        reports = supabase_store.list_reports(user_id, limit or 20)
        return [hydrate_supabase_report(report) for report in reports]

    reports = [report for report in load_saved_reports() if report_belongs_to_user(report, user_id)]
    return sorted(reports, key=lambda report: report.get("run_id", ""), reverse=True)


def load_report_for_user(run_id: str, user_id: str) -> dict:
    if supabase_store.is_configured():
        report = supabase_store.load_report(run_id, user_id)
        if not report:
            raise HTTPException(status_code=404, detail="Analysis run not found.")

        return hydrate_supabase_report(report)

    report = load_local_report(run_id)
    if not report_belongs_to_user(report, user_id):
        raise HTTPException(status_code=404, detail="Analysis run not found.")

    return report


def build_saved_analysis_response(report: dict) -> dict:
    files = report.get("files", {})
    if report.get("run_id") and "json_report" not in files:
        files["json_report"] = f"storage/analyses/{report['run_id']}/report.json"
    if report.get("run_id") and "follow_through_debug_chart" not in files:
        debug_path = STORAGE_DIR / "analyses" / report["run_id"] / "output" / "follow_through_debug.png"
        if debug_path.exists():
            files["follow_through_debug_chart"] = format_path(debug_path)

    return make_json_safe(
        {
            "run_id": report.get("run_id"),
            "analysis_version": report.get("analysis_version"),
            "score": report.get("score"),
            "shooting_side": report.get("shooting_side"),
            "camera_view": report.get("camera_view", "side"),
            "reliability": report.get("reliability"),
            "video_metadata": report.get("video_metadata"),
            "metrics": report.get("metrics", {}),
            "phases": report.get("phases", {}),
            "feedback": report.get("feedback", []),
            "coaching_items": report.get("coaching_items", []),
            "files": files,
            "output_urls": build_output_urls(report, files),
        }
    )


def parse_run_created_at(run_id: str) -> str:
    parts = run_id.split("_")
    if len(parts) < 2:
        return ""

    time_part = parts[1]
    if len(time_part) == 6:
        time_part = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"

    return f"{parts[0]} {time_part}"


def summarize_report(report: dict) -> dict:
    files = report.get("files", {})
    if report.get("run_id") and "json_report" not in files:
        files["json_report"] = f"storage/analyses/{report['run_id']}/report.json"
    if report.get("run_id") and "follow_through_debug_chart" not in files:
        debug_path = STORAGE_DIR / "analyses" / report["run_id"] / "output" / "follow_through_debug.png"
        if debug_path.exists():
            files["follow_through_debug_chart"] = format_path(debug_path)
    output_urls = build_output_urls(report, files)
    run_id = report.get("run_id")

    return make_json_safe(
        {
            "run_id": run_id,
            "created_at": parse_run_created_at(run_id or ""),
            "score": report.get("score"),
            "shooting_side": report.get("shooting_side"),
            "camera_view": report.get("camera_view", "side"),
            "video": Path(files.get("original_video", "original.mp4")).name,
            "report_url": output_urls.get("json_report"),
            "chart_url": output_urls.get("angles_chart"),
            "follow_through_debug_chart_url": output_urls.get("follow_through_debug_chart"),
            "annotated_video_url": output_urls.get("annotated_video"),
        }
    )


def metric_comparison(
    first_metrics: dict,
    second_metrics: dict,
    key: str,
    label: str,
    digits: int = 3,
) -> dict:
    first_value = first_metrics.get(key)
    second_value = second_metrics.get(key)
    delta = None

    if isinstance(first_value, (int, float)) and isinstance(second_value, (int, float)):
        delta = round(second_value - first_value, digits)

    return {
        "key": key,
        "label": label,
        "first": first_value,
        "second": second_value,
        "delta": delta,
    }


def compare_reports(first_report: dict, second_report: dict) -> dict:
    first = build_saved_analysis_response(first_report)
    second = build_saved_analysis_response(second_report)
    first_metrics = first.get("metrics", {})
    second_metrics = second.get("metrics", {})

    metric_rows = [
        metric_comparison(first_metrics, second_metrics, "release_confidence", "Release Confidence", digits=2),
        metric_comparison(first_metrics, second_metrics, "release_elbow_angle", "Release Elbow Angle", digits=2),
        metric_comparison(first_metrics, second_metrics, "follow_through_frames", "Follow-Through Frames", digits=0),
        metric_comparison(first_metrics, second_metrics, "follow_through_ratio", "Follow-Through Ratio", digits=2),
        metric_comparison(first_metrics, second_metrics, "hip_rise", "Hip Rise", digits=3),
        metric_comparison(first_metrics, second_metrics, "ankle_lift", "Ankle Lift", digits=3),
        metric_comparison(first_metrics, second_metrics, "min_knee_angle", "Deepest Knee Angle", digits=2),
        metric_comparison(first_metrics, second_metrics, "elbow_angle_std", "Arm Motion Variation", digits=2),
    ]

    return make_json_safe(
        {
            "first": first,
            "second": second,
            "score_delta": (second.get("score") or 0) - (first.get("score") or 0),
            "metrics": metric_rows,
            "coaching_items": {
                "first": first.get("coaching_items", []),
                "second": second.get("coaching_items", []),
            },
        }
    )


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)

    return value


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "analysis_version": ANALYSIS_VERSION,
        "auth_configured": bool(
            os.getenv("SUPABASE_JWT_SECRET")
            or (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"))
        ),
        "persistence": "supabase" if supabase_store.is_configured() else "local",
    }


@app.get("/analyses")
def list_analyses(limit: int = 20, authorization: str | None = Header(default=None)) -> list[dict]:
    user_id = get_request_user_id(authorization)
    reports = load_reports_for_user(user_id, limit)
    summaries = []

    for report in reports[: max(1, min(limit, 100))]:
        summaries.append(summarize_report(report))

    return summaries


@app.get("/analyses/compare")
def compare_analyses(run_a: str, run_b: str, authorization: str | None = Header(default=None)) -> dict:
    if run_a == run_b:
        raise HTTPException(status_code=400, detail="Choose two different analysis runs.")

    user_id = get_request_user_id(authorization)
    first_report = load_report_for_user(run_a, user_id)
    second_report = load_report_for_user(run_b, user_id)

    return compare_reports(first_report, second_report)


@app.get("/analyses/{run_id}/compare-best")
def compare_analysis_to_best(run_id: str, authorization: str | None = Header(default=None)) -> dict:
    user_id = get_request_user_id(authorization)
    current_report = load_report_for_user(run_id, user_id)

    candidates = [
        report
        for report in load_reports_for_user(user_id, 100)
        if report.get("run_id") != run_id
    ]

    if not candidates:
        raise HTTPException(status_code=404, detail="No other saved analyses are available for comparison.")

    best_report = max(candidates, key=lambda report: report.get("score") or 0)
    comparison = compare_reports(best_report, current_report)
    comparison["mode"] = "best_baseline"
    comparison["baseline_label"] = "Best Saved Shot"
    comparison["current_label"] = "Current Shot"

    return comparison


@app.get("/analyses/{run_id}")
def get_analysis(run_id: str, authorization: str | None = Header(default=None)) -> dict:
    user_id = get_request_user_id(authorization)
    report = load_report_for_user(run_id, user_id)

    return build_saved_analysis_response(report)


@app.delete("/analyses/{run_id}")
def delete_analysis(run_id: str, authorization: str | None = Header(default=None)) -> dict:
    user_id = get_request_user_id(authorization)
    if supabase_store.is_configured():
        if not supabase_store.delete_report(run_id, user_id):
            raise HTTPException(status_code=404, detail="Analysis run not found.")

        return {"deleted": True, "run_id": run_id}

    report = load_local_report(run_id)
    if not report_belongs_to_user(report, user_id):
        raise HTTPException(status_code=404, detail="Analysis run not found.")

    run_dir = get_run_dir(run_id)
    shutil.rmtree(run_dir)

    return {"deleted": True, "run_id": run_id}


@app.post("/analyze-shot")
async def analyze_shot_endpoint(
    file: UploadFile = File(...),
    save_chart: bool = True,
    save_annotated_video: bool = True,
    save_report: bool = True,
    camera_view: str = "side",
    camera_view_form: str | None = Form(default=None, alias="camera_view"),
    x_camera_view: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict:
    user_id = get_request_user_id(authorization)
    validate_video_file(file)
    try:
        camera_view = x_camera_view or camera_view_form or camera_view
        camera_view = normalize_camera_view(camera_view)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_paths = create_analysis_run(file.filename or "uploaded-video.mp4")
    upload_path = run_paths["input_video"]

    try:
        with upload_path.open("wb") as output_file:
            while chunk := await file.read(1024 * 1024):
                output_file.write(chunk)

        result = analyze_video(
            str(upload_path),
            run_dir=run_paths["run_dir"],
            copy_input=False,
            save_chart=save_chart,
            save_annotated_video=save_annotated_video,
            save_json_report=save_report,
            display=False,
            camera_view=camera_view,
            owner_user_id=user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = {
        "run_id": result["run_id"],
        "analysis_version": ANALYSIS_VERSION,
        "score": result["analysis"]["score"],
        "shooting_side": result["analysis"]["shooting_side"],
        "camera_view": result["analysis"]["camera_view"],
        "reliability": result["analysis"]["reliability"],
        "video_metadata": result["video_metadata"],
        "metrics": result["analysis"]["metrics"],
        "phases": result["analysis"]["phases"],
        "feedback": result["analysis"]["feedback"],
        "coaching_items": result["analysis"]["coaching_items"],
        "persistence": "local",
    }
    response.update(build_file_response(result))

    if save_report and supabase_store.is_configured():
        try:
            with Path(result["report_path"]).open() as report_file:
                report = json.load(report_file)
            persisted_report = supabase_store.persist_report(report)
            response["persistence"] = "supabase"
            response["files"] = persisted_report.get("files", response["files"])
            response["output_urls"] = build_output_urls(persisted_report, response["files"])
        except supabase_store.SupabaseStoreError as exc:
            response["persistence_warning"] = str(exc)

    return make_json_safe(response)
