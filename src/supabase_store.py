import json
import mimetypes
import os
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLE = "analyses"
DEFAULT_BUCKET = "shot-analyses"
SIGNED_URL_EXPIRES_SECONDS = 60 * 60 * 24


class SupabaseStoreError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def get_table_name() -> str:
    return os.getenv("SUPABASE_ANALYSES_TABLE", DEFAULT_TABLE)


def get_bucket_name() -> str:
    return os.getenv("SUPABASE_STORAGE_BUCKET", DEFAULT_BUCKET)


def request_json(method: str, path: str, payload=None, extra_headers: dict | None = None):
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise SupabaseStoreError("Supabase persistence requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

    data = None
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
    }
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)

    request = Request(f"{supabase_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read().decode()
    except HTTPError as exc:
        detail = exc.read().decode() or exc.reason
        raise SupabaseStoreError(f"Supabase request failed: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise SupabaseStoreError("Could not reach Supabase persistence.") from exc

    if not response_body:
        return None

    return json.loads(response_body)


def upload_file(object_path: str, local_path: str) -> None:
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise SupabaseStoreError("Supabase persistence requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

    path = Path(local_path)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    quoted_path = quote(object_path, safe="/")
    request = Request(
        f"{supabase_url.rstrip('/')}/storage/v1/object/{get_bucket_name()}/{quoted_path}",
        data=path.read_bytes(),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response.read()
    except HTTPError as exc:
        detail = exc.read().decode() or exc.reason
        raise SupabaseStoreError(f"Supabase storage upload failed: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise SupabaseStoreError("Could not upload file to Supabase Storage.") from exc


def signed_url(object_path: str) -> str:
    quoted_path = quote(object_path, safe="/")
    data = request_json(
        "POST",
        f"/storage/v1/object/sign/{get_bucket_name()}/{quoted_path}",
        {"expiresIn": SIGNED_URL_EXPIRES_SECONDS},
    )
    url = data.get("signedURL") if isinstance(data, dict) else None
    if not url:
        raise SupabaseStoreError("Supabase did not return a signed storage URL.")

    if url.startswith("http://") or url.startswith("https://"):
        return url

    return f"{os.getenv('SUPABASE_URL').rstrip('/')}{url}"


def local_file_path(relative_path: str | None) -> Path | None:
    if not relative_path or relative_path.startswith("http://") or relative_path.startswith("https://"):
        return None

    path = PROJECT_ROOT / relative_path
    return path if path.exists() and path.is_file() else None


def build_storage_files(report: dict) -> dict:
    run_id = report["run_id"]
    owner_user_id = report.get("owner_user_id", "guest")
    storage_files = {}

    for name, relative_path in report.get("files", {}).items():
        local_path = local_file_path(relative_path)
        if not local_path:
            continue

        object_path = f"analyses/{owner_user_id}/{run_id}/{name}{local_path.suffix.lower()}"
        upload_file(object_path, str(local_path))
        storage_files[name] = object_path

    return storage_files


def add_signed_output_urls(report: dict) -> dict:
    hydrated_report = deepcopy(report)
    output_urls = {}

    for name, object_path in hydrated_report.get("storage_files", {}).items():
        if name in {"angles_chart", "follow_through_debug_chart", "annotated_video", "original_video", "json_report"}:
            output_urls[name] = signed_url(object_path)

    hydrated_report["output_urls"] = output_urls
    return hydrated_report


def persist_report(report: dict) -> dict:
    report_to_save = deepcopy(report)
    report_to_save["storage_bucket"] = get_bucket_name()
    report_to_save["storage_files"] = build_storage_files(report_to_save)

    body = {
        "run_id": report_to_save["run_id"],
        "owner_user_id": report_to_save.get("owner_user_id", "guest"),
        "analysis_version": report_to_save.get("analysis_version"),
        "score": report_to_save.get("score"),
        "shooting_side": report_to_save.get("shooting_side"),
        "camera_view": report_to_save.get("camera_view", "side"),
        "report": report_to_save,
    }
    table = quote(get_table_name(), safe="")
    request_json(
        "POST",
        f"/rest/v1/{table}?on_conflict=run_id",
        body,
        {"Prefer": "resolution=merge-duplicates,return=minimal"},
    )

    return add_signed_output_urls(report_to_save)


def list_reports(owner_user_id: str, limit: int = 20) -> list[dict]:
    table = quote(get_table_name(), safe="")
    owner_filter = quote(owner_user_id, safe="")
    path = (
        f"/rest/v1/{table}?select=report"
        f"&owner_user_id=eq.{owner_filter}"
        f"&order=created_at.desc"
        f"&limit={max(1, min(limit, 100))}"
    )
    rows = request_json("GET", path) or []
    return [row["report"] for row in rows if row.get("report")]


def load_report(run_id: str, owner_user_id: str) -> dict | None:
    table = quote(get_table_name(), safe="")
    run_filter = quote(run_id, safe="")
    owner_filter = quote(owner_user_id, safe="")
    path = f"/rest/v1/{table}?select=report&run_id=eq.{run_filter}&owner_user_id=eq.{owner_filter}&limit=1"
    rows = request_json("GET", path) or []
    return rows[0]["report"] if rows else None


def delete_report(run_id: str, owner_user_id: str) -> bool:
    report = load_report(run_id, owner_user_id)
    if not report:
        return False

    storage_paths = list(report.get("storage_files", {}).values())
    if storage_paths:
        try:
            request_json("DELETE", f"/storage/v1/object/{get_bucket_name()}", {"prefixes": storage_paths})
        except SupabaseStoreError:
            pass

    table = quote(get_table_name(), safe="")
    run_filter = quote(run_id, safe="")
    owner_filter = quote(owner_user_id, safe="")
    request_json("DELETE", f"/rest/v1/{table}?run_id=eq.{run_filter}&owner_user_id=eq.{owner_filter}")
    return True
