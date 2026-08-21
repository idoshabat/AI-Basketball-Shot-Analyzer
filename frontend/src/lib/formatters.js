import { API_BASE_URL } from "./appConfig";

export function formatMetric(value, digits = 2) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(digits);
  }

  return String(value);
}

export function titleCase(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatCoachingValue(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return typeof value === "number" ? formatMetric(value) : String(value);
}

export function formatFrameRange(item) {
  if (item.start_frame === null || item.start_frame === undefined) {
    return "N/A";
  }

  if (item.end_frame === null || item.end_frame === undefined || item.end_frame === item.start_frame) {
    return `Frame ${item.start_frame}`;
  }

  return `Frames ${item.start_frame}-${item.end_frame}`;
}

export function formatRunDate(value) {
  if (!value) {
    return "Unknown time";
  }

  return value;
}

export function formatDelta(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  if (typeof value !== "number") {
    return String(value);
  }

  const formatted = Number.isInteger(value) ? value.toString() : value.toFixed(2);
  return value > 0 ? `+${formatted}` : formatted;
}

export function resolveOutputUrl(path) {
  if (!path) {
    return null;
  }

  if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("/samples/")) {
    return path;
  }

  if (path.startsWith("/storage/v1/") && import.meta.env.VITE_SUPABASE_URL) {
    return `${import.meta.env.VITE_SUPABASE_URL.replace(/\/$/, "")}${path}`;
  }

  if (path.startsWith("/object/sign/") && import.meta.env.VITE_SUPABASE_URL) {
    return `${import.meta.env.VITE_SUPABASE_URL.replace(/\/$/, "")}/storage/v1${path}`;
  }

  return `${API_BASE_URL}${path}`;
}

export function getRequestErrorMessage(error) {
  if (error instanceof TypeError && error.message === "Failed to fetch") {
    return `Could not reach the backend at ${API_BASE_URL}. Check Render status, VITE_API_BASE_URL, and CORS_ORIGINS/CORS_ORIGIN_REGEX.`;
  }

  return error.message;
}

export function getUserName(session) {
  return session?.user?.user_metadata?.full_name || session?.user?.email || "Signed-in player";
}

export function formatElapsedTime(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
