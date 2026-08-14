from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
UPLOAD_DIR = PROJECT_ROOT / "videos" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

sys.path.insert(0, str(SRC_DIR))

from analyze_video import analyze_video, format_path


OUTPUT_DIR.mkdir(exist_ok=True)
app = FastAPI(title="AI Basketball Shot Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "https://ai-basketball-shot-analyzer-edb2.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


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
        "keypoints_csv": format_path(result["keypoints_path"]),
        "features_csv": format_path(result["features_path"]),
        "angles_chart": format_path(result["chart_path"]),
        "pose_video": format_path(result["output_path"]),
        "annotated_video": format_path(result["annotated_video_path"]),
        "json_report": format_path(result["report_path"]),
    }

    output_urls = {}
    for name, path in files.items():
        if path and path.startswith("output/"):
            output_urls[name] = f"/{path}"

    return {
        "files": files,
        "output_urls": output_urls,
    }


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
    return {"status": "ok"}


@app.post("/analyze-shot")
async def analyze_shot_endpoint(
    file: UploadFile = File(...),
    save_chart: bool = True,
    save_annotated_video: bool = True,
    save_report: bool = True,
) -> dict:
    suffix = validate_video_file(file)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"

    try:
        with upload_path.open("wb") as output_file:
            while chunk := await file.read(1024 * 1024):
                output_file.write(chunk)

        result = analyze_video(
            str(upload_path),
            save_chart=save_chart,
            save_annotated_video=save_annotated_video,
            save_json_report=save_report,
            display=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = {
        "score": result["analysis"]["score"],
        "shooting_side": result["analysis"]["shooting_side"],
        "video_metadata": result["video_metadata"],
        "metrics": result["analysis"]["metrics"],
        "phases": result["analysis"]["phases"],
        "feedback": result["analysis"]["feedback"],
        "coaching_items": result["analysis"]["coaching_items"],
    }
    response.update(build_file_response(result))

    return make_json_safe(response)
