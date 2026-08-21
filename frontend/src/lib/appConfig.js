export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD ? "https://ai-basketball-shot-analyzer.onrender.com" : "http://127.0.0.1:8000");

export const ACCESS_MODE_STORAGE_KEY = "shotAnalyzerAccessMode";
export const REPORT_CACHE_STORAGE_KEY = "shotAnalyzerReportCache";

export function readAccessMode() {
  try {
    return globalThis.window?.localStorage?.getItem(ACCESS_MODE_STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

export function writeAccessMode(value) {
  try {
    globalThis.window?.localStorage?.setItem(ACCESS_MODE_STORAGE_KEY, value);
  } catch {
    // Some browser privacy modes block storage. The in-memory state still keeps the current page usable.
  }
}

export function clearAccessMode() {
  try {
    globalThis.window?.localStorage?.removeItem(ACCESS_MODE_STORAGE_KEY);
  } catch {
    // Ignore blocked storage.
  }
}

export const CAMERA_VIEW_GUIDANCE = {
  side: {
    title: "Side view",
    description: "Best for release timing, elbow extension, knee bend, leg drive, jump lift, and follow-through hold.",
  },
  front: {
    title: "Front view",
    description: "Best for feet parallelism, knee-to-foot lines, forearm verticality, follow-through direction, and body lean.",
  },
  back: {
    title: "Back view",
    description: "Best for shoulder/hip alignment, arm path, follow-through direction, stance symmetry, and body lean.",
  },
};

export const SAMPLE_RESULTS = [
  {
    id: "front-ft10",
    label: "AI Front View - FT10",
    description: "High-reliability AI-generated front-view shot with strong alignment signals.",
    path: "/samples/front-ft10/analysis.json",
  },
  {
    id: "side-ft3",
    label: "AI Side View - FT3",
    description: "AI-generated side-view shot with release and follow-through analysis.",
    path: "/samples/side-ft3/analysis.json",
  },
  {
    id: "back-ft6",
    label: "AI Back View - FT6",
    description: "AI-generated back-view shot with clean reliability.",
    path: "/samples/back-ft6/analysis.json",
  },
];
