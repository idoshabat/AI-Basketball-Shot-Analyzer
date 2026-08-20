import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { isSupabaseConfigured, supabase } from "./supabaseClient";
import "./styles.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD ? "https://ai-basketball-shot-analyzer.onrender.com" : "http://127.0.0.1:8000");

const CAMERA_VIEW_GUIDANCE = {
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

const ACCESS_MODE_STORAGE_KEY = "shotAnalyzerAccessMode";

const SAMPLE_RESULTS = [
  {
    id: "side-ft2",
    label: "Good Side View - FT2",
    description: "Clean side-view form shot.",
    path: "/samples/side-ft2/analysis.json",
  },
  {
    id: "side-ft3",
    label: "Good Side View - FT3",
    description: "Clean side-view shot with strong score.",
    path: "/samples/side-ft3/analysis.json",
  },
  {
    id: "front-ft4",
    label: "Good Front View - FT4",
    description: "Front-view alignment analysis.",
    path: "/samples/front-ft4/analysis.json",
  },
  {
    id: "front-ft5",
    label: "Good Front View - FT5",
    description: "Clean single-player front-view shot.",
    path: "/samples/front-ft5/analysis.json",
  },
  {
    id: "back-ft6",
    label: "Good Back View - FT6",
    description: "Back-view shot with clean reliability.",
    path: "/samples/back-ft6/analysis.json",
  },
  {
    id: "bad-ft1",
    label: "Bad Input - FT1",
    description: "Far, crowded video that should trigger quality warnings.",
    path: "/samples/bad-ft1/analysis.json",
  },
];

function formatMetric(value, digits = 2) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(digits);
  }

  return String(value);
}

function titleCase(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatCoachingValue(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return typeof value === "number" ? formatMetric(value) : String(value);
}

function formatFrameRange(item) {
  if (item.start_frame === null || item.start_frame === undefined) {
    return "N/A";
  }

  if (item.end_frame === null || item.end_frame === undefined || item.end_frame === item.start_frame) {
    return `Frame ${item.start_frame}`;
  }

  return `Frames ${item.start_frame}-${item.end_frame}`;
}

function formatRunDate(value) {
  if (!value) {
    return "Unknown time";
  }

  return value;
}

function formatDelta(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  if (typeof value !== "number") {
    return String(value);
  }

  const formatted = Number.isInteger(value) ? value.toString() : value.toFixed(2);
  return value > 0 ? `+${formatted}` : formatted;
}

function resolveOutputUrl(path) {
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

function getRequestErrorMessage(error) {
  if (error instanceof TypeError && error.message === "Failed to fetch") {
    return `Could not reach the backend at ${API_BASE_URL}. Check Render status, VITE_API_BASE_URL, and CORS_ORIGINS/CORS_ORIGIN_REGEX.`;
  }

  return error.message;
}

function getUserName(session) {
  return session?.user?.user_metadata?.full_name || session?.user?.email || "Signed-in player";
}

function formatElapsedTime(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function App() {
  const [file, setFile] = useState(null);
  const [cameraView, setCameraView] = useState("side");
  const [shootingSide, setShootingSide] = useState("auto");
  const [selectedSampleId, setSelectedSampleId] = useState(SAMPLE_RESULTS[0].id);
  const [includeAnnotatedVideo, setIncludeAnnotatedVideo] = useState(true);
  const [result, setResult] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [selectedComparisonRuns, setSelectedComparisonRuns] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisElapsedSeconds, setAnalysisElapsedSeconds] = useState(0);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const [openingRunId, setOpeningRunId] = useState(null);
  const [deletingRunId, setDeletingRunId] = useState(null);
  const [isDeletingAllHistory, setIsDeletingAllHistory] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [isComparingBest, setIsComparingBest] = useState(false);
  const [session, setSession] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(!isSupabaseConfigured);
  const [isSigningIn, setIsSigningIn] = useState(false);
  const [workspaceView, setWorkspaceView] = useState("capture");
  const [hasEnteredApp, setHasEnteredApp] = useState(() => {
    return window.localStorage.getItem(ACCESS_MODE_STORAGE_KEY) === "guest";
  });
  const [isGuestMode, setIsGuestMode] = useState(() => {
    return window.localStorage.getItem(ACCESS_MODE_STORAGE_KEY) === "guest";
  });
  const [selectedEvidenceItem, setSelectedEvidenceItem] = useState(null);
  const [deletePrompt, setDeletePrompt] = useState(null);
  const [actionToast, setActionToast] = useState(null);
  const annotatedVideoRef = useRef(null);
  const annotatedVideoSectionRef = useRef(null);
  const resultsSectionRef = useRef(null);

  const appView = isAnalyzing ? "analyzing" : result ? "report" : workspaceView;
  const chartUrl = resolveOutputUrl(result?.output_urls?.angles_chart);
  const followThroughDebugChartUrl = resolveOutputUrl(result?.output_urls?.follow_through_debug_chart);
  const annotatedVideoUrl = resolveOutputUrl(result?.output_urls?.annotated_video);
  const authHeaders = session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {};
  const selectedSample = SAMPLE_RESULTS.find((sample) => sample.id === selectedSampleId) || SAMPLE_RESULTS[0];
  const ballTracking = result?.ball_tracking?.status || result?.metrics?.ball_tracking_status
    ? result?.ball_tracking?.status
      ? result.ball_tracking
      : {
        status: result.metrics.ball_tracking_status,
        detector_backend: result.metrics.ball_detector_backend,
        visibility_ratio: result.metrics.ball_visibility_ratio,
        close_visibility_ratio: result.metrics.ball_close_visibility_ratio,
        ball_release_frame: result.metrics.ball_release_frame,
        release_frame_delta: result.metrics.ball_release_frame_delta,
        arc_height: result.metrics.ball_arc_height,
        avg_wrist_distance: result.metrics.ball_avg_wrist_distance,
        note: "Beta: ball tracking is informational and does not affect the score yet.",
      }
    : null;

  function showActionToast(title, detail = "", tone = "info") {
    setActionToast({
      id: Date.now(),
      title,
      detail,
      tone,
    });
  }

  const coreMetrics = useMemo(() => {
    if (!result) {
      return [];
    }

    const metrics = [
      ["Release Frame", result.metrics.release_frame],
      ["Release Confidence", result.metrics.release_confidence],
      ["Release Confidence Label", result.metrics.release_confidence_label],
      ["Release Elbow Angle", result.metrics.release_elbow_angle],
      ["Follow-Through End", result.metrics.follow_through_end_frame],
      ["Follow-Through Ratio", result.metrics.follow_through_ratio],
      ["Hip Rise", result.metrics.hip_rise],
      ["Ankle Lift", result.metrics.ankle_lift],
      ["Scene People Max", result.metrics.scene_max_people_count],
      ["Scene Multi-Person Frames", result.metrics.scene_multi_person_frame_ratio],
      ["Pose People Max", result.metrics.max_detected_people_count],
      ["Player Size", result.metrics.median_selected_pose_area],
      ["Body Height", result.metrics.median_body_height],
      ["Ball Close Visibility", result.metrics.ball_close_visibility_ratio],
      ["Ball Release Frame", result.metrics.ball_release_frame],
      ["Ball Release Alignment", result.metrics.ball_release_confidence_label],
    ];

    if (result.camera_view === "side") {
      metrics.push(
        ["Ball Arc Quality", result.metrics.ball_side_arc_quality],
        ["Ball Upward Frames", result.metrics.ball_post_release_upward_frames],
      );
    } else {
      metrics.push(
        ["Ball Line Drift", result.metrics.ball_front_back_line_drift],
        ["Left Shin Vertical Error", result.metrics.left_shin_vertical_error],
        ["Right Shin Vertical Error", result.metrics.right_shin_vertical_error],
        ["Shin Parallel Error", result.metrics.shin_parallel_error],
        ["Foot Parallel Error", result.metrics.foot_parallel_error],
        ["Left Foot Square Error", result.metrics.left_foot_floor_error],
        ["Right Foot Square Error", result.metrics.right_foot_floor_error],
        ["Load Foot Parallel Error", result.metrics.load_foot_parallel_error],
        ["Load Left Foot Square Error", result.metrics.load_left_foot_floor_error],
        ["Load Right Foot Square Error", result.metrics.load_right_foot_floor_error],
        ["Load Knee Alignment", result.metrics.load_knee_to_ankle_alignment_error],
        ["Load Shin Parallel Error", result.metrics.load_shin_parallel_error],
        ["Foot Stagger Error", result.metrics.foot_stagger_error],
        ["Load Base Score", result.metrics.load_base_score],
        ["Load Base Penalty", result.metrics.load_base_penalty],
        ["Forearm Vertical Error", result.metrics.forearm_vertical_error],
        ["Follow-Through Line Error", result.metrics.follow_through_vertical_error],
        ["Body Lean", result.metrics.body_lean],
      );
    }

    return metrics;
  }, [result]);

  const improvementItems = useMemo(() => {
    if (!result) {
      return [];
    }

    return result.coaching_items || [];
  }, [result]);

  function watchCoachingMoment(item) {
    const fps = result?.video_metadata?.fps;
    const video = annotatedVideoRef.current;
    const videoSection = annotatedVideoSectionRef.current;

    if (!video || !fps || !item.start_frame) {
      showActionToast("Annotated video unavailable", "Generate an annotated video to watch this moment.", "danger");
      return;
    }

    showActionToast("Jumping to moment", `${titleCase(item.phase)} - ${formatFrameRange(item)}`);
    video.currentTime = Math.max(0, (item.start_frame - 1) / fps);
    videoSection?.scrollIntoView({ behavior: "smooth", block: "start" });
    video.play().catch(() => {});
  }

  function openEvidenceFrame(item) {
    setSelectedEvidenceItem(item);
    showActionToast("Evidence frame opened", item.title);
  }

  function getEvidenceFrameUrl(item) {
    if (!item) {
      return null;
    }

    if (item.evidence_frame_url) {
      return resolveOutputUrl(item.evidence_frame_url);
    }

    if (item.evidence_frame_file && result?.output_urls?.[item.evidence_frame_file]) {
      return resolveOutputUrl(result.output_urls[item.evidence_frame_file]);
    }

    return resolveOutputUrl(item.evidence_frame_path);
  }

  async function loadAnalysisHistory() {
    if (!hasEnteredApp) {
      setAnalysisHistory([]);
      return;
    }

    setIsLoadingHistory(true);
    showActionToast("Refreshing library", "Checking your saved shot reports.");
    try {
      const response = await fetch(`${API_BASE_URL}/analyses?limit=10`, {
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not load recent analyses.");
      }

      setAnalysisHistory(data);
      showActionToast("Library updated", `${data.length} saved ${data.length === 1 ? "shot" : "shots"} ready.`, "success");
    } catch (historyError) {
      setError(getRequestErrorMessage(historyError));
      showActionToast("Refresh failed", "Could not load saved analyses.", "danger");
    } finally {
      setIsLoadingHistory(false);
    }
  }

  async function openSavedAnalysis(runId) {
    setError("");
    setOpeningRunId(runId);
    showActionToast("Opening report", "Loading the saved analysis.");
    try {
      const response = await fetch(`${API_BASE_URL}/analyses/${runId}`, {
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not open saved analysis.");
      }

      setResult(data);
      setSelectedEvidenceItem(null);
      showActionToast("Report opened", "Jumping to the analysis details.", "success");
      window.setTimeout(() => {
        resultsSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    } catch (savedAnalysisError) {
      setError(getRequestErrorMessage(savedAnalysisError));
      showActionToast("Open failed", "Could not load that saved report.", "danger");
    } finally {
      setOpeningRunId(null);
    }
  }

  function toggleComparisonRun(runId) {
    setSelectedComparisonRuns((currentRuns) => {
      if (currentRuns.includes(runId)) {
        return currentRuns.filter((selectedRunId) => selectedRunId !== runId);
      }

      return [...currentRuns, runId].slice(-2);
    });
  }

  async function compareSelectedRuns() {
    if (selectedComparisonRuns.length !== 2) {
      setError("Choose exactly two analyses to compare.");
      return;
    }

    setIsComparing(true);
    setError("");
    showActionToast("Comparing shots", "Building the side-by-side report.");
    try {
      const params = new URLSearchParams({
        run_a: selectedComparisonRuns[0],
        run_b: selectedComparisonRuns[1],
      });
      const response = await fetch(`${API_BASE_URL}/analyses/compare?${params}`, {
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not compare analyses.");
      }

      setComparison(data);
      showActionToast("Comparison ready", "The comparison panel is updated.", "success");
    } catch (compareError) {
      setError(getRequestErrorMessage(compareError));
      showActionToast("Compare failed", "Could not compare those shots.", "danger");
    } finally {
      setIsComparing(false);
    }
  }

  async function compareToBestShot() {
    if (!result?.run_id) {
      setError("Analyze or open a saved shot before comparing to your best shot.");
      return;
    }

    setIsComparingBest(true);
    setError("");
    showActionToast("Finding your best shot", "Comparing against your saved baseline.");
    try {
      const response = await fetch(`${API_BASE_URL}/analyses/${result.run_id}/compare-best`, {
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not compare to your best shot.");
      }

      setComparison(data);
      showActionToast("Best-shot comparison ready", "Baseline comparison is updated.", "success");
    } catch (bestComparisonError) {
      setError(getRequestErrorMessage(bestComparisonError));
      showActionToast("Compare failed", "Could not compare to your best shot.", "danger");
    } finally {
      setIsComparingBest(false);
    }
  }

  async function performDeleteSavedAnalysis(analysis) {
    setError("");
    setDeletingRunId(analysis.run_id);
    setDeletePrompt(null);
    showActionToast("Deleting analysis", "Removing the selected report.");
    try {
      const response = await fetch(`${API_BASE_URL}/analyses/${analysis.run_id}`, {
        method: "DELETE",
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not delete saved analysis.");
      }

      if (result?.run_id === analysis.run_id) {
        setResult(null);
        setSelectedEvidenceItem(null);
      }
      setSelectedComparisonRuns((currentRuns) => currentRuns.filter((runId) => runId !== analysis.run_id));
      if (comparison?.first?.run_id === analysis.run_id || comparison?.second?.run_id === analysis.run_id) {
        setComparison(null);
      }
      showActionToast("Analysis deleted", "The report was removed from your library.", "success");
      loadAnalysisHistory();
    } catch (deleteError) {
      setError(getRequestErrorMessage(deleteError));
      showActionToast("Delete failed", "Could not delete that report.", "danger");
    } finally {
      setDeletingRunId(null);
    }
  }

  function requestDeleteSavedAnalysis(analysis) {
    showActionToast("Confirm delete", "Review the confirmation card.");
    setDeletePrompt({
      type: "single",
      analysis,
      title: "Delete this analysis?",
      detail: `This will permanently remove ${analysis.video || analysis.run_id} from your recent analyses.`,
      confirmLabel: "Delete analysis",
    });
  }

  async function performDeleteAllSavedAnalyses() {
    if (analysisHistory.length === 0) {
      return;
    }

    setError("");
    setIsDeletingAllHistory(true);
    setDeletePrompt(null);
    showActionToast("Deleting library", "Clearing saved analyses from this account.");
    try {
      const response = await fetch(`${API_BASE_URL}/analyses`, {
        method: "DELETE",
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not delete saved analyses.");
      }

      setAnalysisHistory([]);
      setSelectedComparisonRuns([]);
      setComparison(null);
      setResult(null);
      setSelectedEvidenceItem(null);
      showActionToast("Library cleared", "All recent analyses were deleted.", "success");
    } catch (deleteError) {
      setError(getRequestErrorMessage(deleteError));
      showActionToast("Delete failed", "Could not clear the library.", "danger");
    } finally {
      setIsDeletingAllHistory(false);
    }
  }

  function requestDeleteAllSavedAnalyses() {
    if (analysisHistory.length === 0) {
      return;
    }

    showActionToast("Confirm delete all", "Review the confirmation card.");
    setDeletePrompt({
      type: "all",
      title: "Delete all analyses?",
      detail: `This will permanently remove ${analysisHistory.length} saved ${analysisHistory.length === 1 ? "analysis" : "analyses"} from this library.`,
      confirmLabel: "Delete all",
    });
  }

  function confirmDeletePrompt() {
    if (!deletePrompt || isDeletingAllHistory || deletingRunId) {
      return;
    }

    if (deletePrompt.type === "single") {
      performDeleteSavedAnalysis(deletePrompt.analysis);
      return;
    }

    performDeleteAllSavedAnalyses();
  }

  useEffect(() => {
    if (!supabase) {
      return;
    }

    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      if (data.session) {
        setHasEnteredApp(true);
        setIsGuestMode(false);
        window.localStorage.setItem(ACCESS_MODE_STORAGE_KEY, "signed_in");
      }
      setIsAuthReady(true);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      if (nextSession) {
        setHasEnteredApp(true);
        setIsGuestMode(false);
        window.localStorage.setItem(ACCESS_MODE_STORAGE_KEY, "signed_in");
      }
      setSelectedComparisonRuns([]);
      setComparison(null);
      setResult(null);
      setSelectedEvidenceItem(null);
    });

    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (isAuthReady && hasEnteredApp) {
      loadAnalysisHistory();
    }
  }, [isAuthReady, hasEnteredApp, session?.access_token]);

  useEffect(() => {
    if (!actionToast) {
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setActionToast(null);
    }, 2600);

    return () => window.clearTimeout(timerId);
  }, [actionToast]);

  useEffect(() => {
    if (!isAnalyzing) {
      return undefined;
    }

    const timerId = window.setInterval(() => {
      setAnalysisElapsedSeconds((currentSeconds) => currentSeconds + 1);
    }, 1000);

    return () => window.clearInterval(timerId);
  }, [isAnalyzing]);

  async function signInWithGoogle() {
    if (!supabase) {
      setError("Supabase is not configured yet.");
      showActionToast("Sign in unavailable", "Supabase is not configured yet.", "danger");
      return;
    }

    setIsSigningIn(true);
    setError("");
    showActionToast("Opening Google", "Redirecting to secure sign-in.");
    const { error: signInError } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: window.location.origin,
      },
    });

    if (signInError) {
      setError(signInError.message);
      showActionToast("Sign in failed", signInError.message, "danger");
      setIsSigningIn(false);
    }
  }

  async function signOut() {
    if (!supabase) {
      return;
    }

    setError("");
    await supabase.auth.signOut();
    showActionToast("Signed out", "Your local workspace is back to the welcome screen.", "success");
    setHasEnteredApp(false);
    setIsGuestMode(false);
    window.localStorage.removeItem(ACCESS_MODE_STORAGE_KEY);
  }

  function continueAsGuest() {
    setError("");
    setHasEnteredApp(true);
    setIsGuestMode(true);
    window.localStorage.setItem(ACCESS_MODE_STORAGE_KEY, "guest");
    showActionToast("Guest mode started", "You can analyze a shot without signing in.", "success");
  }

  function startNewAnalysis() {
    setResult(null);
    setComparison(null);
    setSelectedEvidenceItem(null);
    setError("");
    setWorkspaceView("capture");
    showActionToast("New analysis", "Upload area is ready.");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function openHistoryView() {
    setError("");
    setWorkspaceView("history");
    showActionToast("Shot library", "Opening your recent analyses.");
    loadAnalysisHistory();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function openCaptureView() {
    setError("");
    setWorkspaceView("capture");
    showActionToast("Capture page", "Ready for another video.");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function analyzeShot(event) {
    event.preventDefault();
    if (!hasEnteredApp) {
      setError("Choose sign in or continue as guest before analyzing.");
      showActionToast("Choose access mode", "Sign in or continue as guest first.", "danger");
      return;
    }

    if (!file) {
      setError("Choose a video before analyzing.");
      showActionToast("Choose a video", "Pick an mp4 or mov before analyzing.", "danger");
      return;
    }

    const selectedCameraView = event.currentTarget.elements.camera_view?.value || cameraView;
    const selectedShootingSide = event.currentTarget.elements.shooting_side?.value || shootingSide;
    setAnalysisElapsedSeconds(0);
    setIsAnalyzing(true);
    setError("");
    setResult(null);
    setSelectedEvidenceItem(null);
    showActionToast("Analysis started", "Locking upload and reading the shot.");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("camera_view", selectedCameraView);
    formData.append("shooting_side", selectedShootingSide);

    const params = new URLSearchParams({
      save_chart: "true",
      save_annotated_video: includeAnnotatedVideo ? "true" : "false",
      save_report: "true",
      camera_view: selectedCameraView,
      shooting_side: selectedShootingSide,
    });

    try {
      const response = await fetch(`${API_BASE_URL}/analyze-shot?${params}`, {
        method: "POST",
        headers: {
          ...authHeaders,
          "X-Camera-View": selectedCameraView,
          "X-Shooting-Side": selectedShootingSide,
        },
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed.");
      }

      if (!data.camera_view) {
        throw new Error("The backend did not return a camera view. Restart the backend and analyze again.");
      }

      if (data.camera_view !== selectedCameraView) {
        throw new Error(`Camera view mismatch: selected ${selectedCameraView}, backend analyzed ${data.camera_view}.`);
      }

      if (selectedCameraView !== "side" && data.metrics?.alignment_status !== "measured") {
        throw new Error("Front/back alignment metrics were not generated. Restart the backend and analyze the video again.");
      }

      if (selectedShootingSide !== "auto" && data.shooting_side !== selectedShootingSide) {
        throw new Error(`Shooting hand mismatch: selected ${selectedShootingSide}, backend analyzed ${data.shooting_side}.`);
      }

      setResult(data);
      setSelectedEvidenceItem(null);
      showActionToast("Analysis complete", `Shot score ${data.score}. Report is ready.`, "success");
      setAnalysisHistory((currentHistory) => {
        const optimisticSummary = {
          run_id: data.run_id,
          created_at: "Saving...",
          score: data.score,
          shooting_side: data.shooting_side,
          camera_view: data.camera_view,
          video: file.name,
        };
        return [optimisticSummary, ...currentHistory.filter((analysis) => analysis.run_id !== data.run_id)].slice(0, 10);
      });
      if (data.persistence === "supabase_pending") {
        window.setTimeout(loadAnalysisHistory, 2500);
      } else {
        loadAnalysisHistory();
      }
    } catch (requestError) {
      setError(getRequestErrorMessage(requestError));
      showActionToast("Analysis failed", getRequestErrorMessage(requestError), "danger");
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function loadSampleResult() {
    setError("");
    setComparison(null);
    setIsLoadingSample(true);
    showActionToast("Loading demo", selectedSample.label);
    try {
      const response = await fetch(selectedSample.path);
      const data = await response.json();
      if (!response.ok) {
        throw new Error("Could not load sample analysis.");
      }

      setResult(data);
      setSelectedEvidenceItem(null);
      showActionToast("Demo loaded", "Sample report is ready.", "success");
      window.setTimeout(() => {
        resultsSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    } catch (sampleError) {
      setError(sampleError.message);
      showActionToast("Demo failed", sampleError.message, "danger");
    } finally {
      setIsLoadingSample(false);
    }
  }

  const selectedEvidenceFrameUrl = getEvidenceFrameUrl(selectedEvidenceItem);

  return (
    <main className="app-shell">
      {actionToast && (
        <div className={`action-toast ${actionToast.tone}`} role="status" aria-live="polite">
          <span aria-hidden="true" />
          <div>
            <strong>{actionToast.title}</strong>
            {actionToast.detail && <p>{actionToast.detail}</p>}
          </div>
        </div>
      )}
      <section className="workspace">
        {isAuthReady && !hasEnteredApp && !session ? (
          <section className="welcome-panel">
            <div className="welcome-copy">
              <p className="eyebrow">AI Basketball Shot Analyzer</p>
              <h1>Train your jumper with motion data.</h1>
              <p>
                Upload a shot, choose the camera angle, and get a clean report with score, priorities, charts, and an
                annotated video.
              </p>
              <div className="welcome-highlights">
                <span>Pose tracking</span>
                <span>Camera-aware feedback</span>
                <span>Saved comparisons</span>
              </div>
            </div>

            <div className="welcome-actions">
              <div>
                <strong>Start analyzing</strong>
                <span>Sign in to keep your shot history, or try the app as a guest.</span>
              </div>
              <button type="button" onClick={signInWithGoogle} disabled={!isSupabaseConfigured || isSigningIn}>
                {isSigningIn && <span className="button-spinner small" aria-hidden="true" />}
                <span>{isSigningIn ? "Opening Google..." : "Sign in with Google"}</span>
              </button>
              <button className="sample-button" type="button" onClick={continueAsGuest}>
                Continue as guest
              </button>
              {!isSupabaseConfigured && (
                <p className="processing-note">Google sign-in needs Supabase env vars. Guest mode is available now.</p>
              )}
              {error && <p className="error-text">{error}</p>}
            </div>
          </section>
        ) : (
          <div className={`app-stage ${appView}`}>
        {appView !== "analyzing" && (
          <nav className="top-nav" aria-label="App navigation">
            <button className="brand-button" type="button" onClick={startNewAnalysis}>
              AI Basketball Shot Analyzer
            </button>
            <div className="nav-actions">
              {appView === "history" ? (
                <button className="secondary-button" type="button" onClick={openCaptureView}>
                  New analysis
                </button>
              ) : (
                <button className="secondary-button" type="button" onClick={openHistoryView}>
                  Recent analyses
                </button>
              )}
              {isSupabaseConfigured ? (
                session ? (
                  <>
                    <div className="account-chip">
                      <strong>{getUserName(session)}</strong>
                      <span>Signed in</span>
                    </div>
                    <button className="secondary-button" type="button" onClick={signOut}>
                      Sign Out
                    </button>
                  </>
                ) : isGuestMode ? (
                  <>
                    <div className="account-chip">
                      <strong>Guest session</strong>
                      <span>Not saved to your account</span>
                    </div>
                    <button className="secondary-button" type="button" onClick={signInWithGoogle} disabled={isSigningIn}>
                      {isSigningIn && <span className="button-spinner small" aria-hidden="true" />}
                      <span>{isSigningIn ? "Opening Google..." : "Sign in"}</span>
                    </button>
                  </>
                ) : (
                  <button className="secondary-button" type="button" onClick={signInWithGoogle} disabled={isSigningIn}>
                    {isSigningIn && <span className="button-spinner small" aria-hidden="true" />}
                    <span>{isSigningIn ? "Opening Google..." : "Sign in with Google"}</span>
                  </button>
                )
              ) : (
                <div className="account-chip">
                  <strong>Guest mode</strong>
                  <span>Supabase auth not configured</span>
                </div>
              )}
            </div>
          </nav>
        )}
        {appView === "capture" && (
        <div className="upload-panel stage-card">
          <div className="capture-copy">
            <p className="eyebrow">AI Basketball Shot Analyzer</p>
            <h1>Upload a shot and get a motion report.</h1>
            <div className="capture-hints" aria-label="Recording tips">
              <span>One shooter</span>
              <span>Full body visible</span>
              <span>Steady camera</span>
            </div>
            <div className="sample-picker">
              <label>
                <span>Demo sample</span>
                <select value={selectedSampleId} onChange={(event) => setSelectedSampleId(event.target.value)}>
                  {SAMPLE_RESULTS.map((sample) => (
                    <option value={sample.id} key={sample.id}>
                      {sample.label}
                    </option>
                  ))}
                </select>
              </label>
              <p>{selectedSample.description}</p>
              <button className="sample-button" type="button" onClick={loadSampleResult} disabled={isLoadingSample}>
                {isLoadingSample && <span className="button-spinner small" aria-hidden="true" />}
                <span>{isLoadingSample ? "Loading demo..." : "Load Demo"}</span>
              </button>
            </div>
          </div>

          <form onSubmit={analyzeShot} className="upload-form">
            <div className="form-section-title">
              <strong>New analysis</strong>
              <span>Choose the video and how it was filmed.</span>
            </div>
            <label className="file-input">
              <span>{file ? file.name : "Choose video file"}</span>
              <input
                type="file"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
            </label>

            <label className="toggle-row">
              <span>Camera view</span>
              <select name="camera_view" value={cameraView} onChange={(event) => setCameraView(event.target.value)}>
                <option value="side">Side</option>
                <option value="front">Front</option>
                <option value="back">Back</option>
              </select>
            </label>

            <div className="camera-guidance">
              <strong>{CAMERA_VIEW_GUIDANCE[cameraView].title}</strong>
              <p>{CAMERA_VIEW_GUIDANCE[cameraView].description}</p>
              <span>Compare shots only when they were filmed from the same camera angle.</span>
            </div>

            <label className="toggle-row">
              <span>Shooting hand</span>
              <select name="shooting_side" value={shootingSide} onChange={(event) => setShootingSide(event.target.value)}>
                <option value="auto">Auto detect</option>
                <option value="right">Right</option>
                <option value="left">Left</option>
              </select>
            </label>

            <div className="capture-guidance">
              <strong>For the best result</strong>
              <p>
                Film one player only, keep the full body visible, use good lighting and video quality, and keep the camera
                steady from the selected angle.
              </p>
            </div>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={includeAnnotatedVideo}
                onChange={(event) => setIncludeAnnotatedVideo(event.target.checked)}
              />
              <span>Generate annotated video</span>
            </label>

            <button type="submit" disabled={isAnalyzing}>
              {isAnalyzing && <span className="button-spinner" aria-hidden="true" />}
              <span>{isAnalyzing ? "Analyzing shot..." : "Analyze Shot"}</span>
            </button>
            <p className="processing-note">
              Analysis can take a few minutes on the hosted backend, especially when annotated video is enabled.
            </p>
          </form>

          {error && <p className="error-text">{error}</p>}
        </div>
        )}

        {appView === "analyzing" && (
          <section className="analysis-loading-screen" aria-live="polite">
            <div className="loading-orbit" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div className="loading-copy">
              <p className="eyebrow">Shot analysis in progress</p>
              <h1>Reading body motion frame by frame.</h1>
              <p>
                Pose tracking, phase detection, scoring, coaching frames, charts, and optional annotated video are being
                generated. Hosted analysis can take a few minutes on Render's low-CPU instance.
              </p>
            </div>
            <div className="loading-dashboard">
              <div className="elapsed-timer large" aria-live="polite">
                <span>Running time</span>
                <strong>{formatElapsedTime(analysisElapsedSeconds)}</strong>
              </div>
              <div className="loading-steps" aria-hidden="true">
                <span>Upload locked</span>
                <span>Pose map</span>
                <span>Release scan</span>
                <span>Report render</span>
              </div>
              <div className="loading-bar" aria-hidden="true">
                <span />
              </div>
              <p>Camera view: {titleCase(cameraView)}. Shooting hand: {titleCase(shootingSide)}.</p>
            </div>
          </section>
        )}

        {appView === "report" && (
          <section className="report-hero">
            <div>
              <p className="eyebrow">Analysis complete</p>
              <h1>Your shot report is ready.</h1>
              <p>
                Review the score, reliability, priorities, evidence frames, charts, and annotated video from this analysis.
              </p>
            </div>
            <button type="button" onClick={startNewAnalysis}>
              Analyze another shot
            </button>
          </section>
        )}

        {appView === "history" && (
        <section className="history-panel history-screen stage-card">
          <div className="history-hero">
            <div>
              <p className="eyebrow dark">Shot library</p>
              <h2>Recent Analyses</h2>
              <p className="subtle">Open saved reports, compare progress, or clear old test runs from this account.</p>
            </div>
            <div className="history-summary">
              <span>Saved shots</span>
              <strong>{analysisHistory.length}</strong>
            </div>
          </div>

          <div className="history-toolbar">
            <div>
              <span>{selectedComparisonRuns.length}/2 selected for comparison</span>
              <strong>{isGuestMode ? "Guest library" : "Personal library"}</strong>
            </div>
            <div className="header-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={compareSelectedRuns}
                disabled={isComparing || selectedComparisonRuns.length !== 2 || isDeletingAllHistory}
              >
                {isComparing && <span className="button-spinner small" aria-hidden="true" />}
                <span>{isComparing ? "Comparing..." : "Compare"}</span>
              </button>
              <button
                className="secondary-button"
                type="button"
                onClick={loadAnalysisHistory}
                disabled={isLoadingHistory || isDeletingAllHistory}
              >
                {isLoadingHistory && <span className="button-spinner small" aria-hidden="true" />}
                <span>{isLoadingHistory ? "Loading..." : "Refresh"}</span>
              </button>
              <button
                className="danger-button ghost-danger"
                type="button"
                onClick={requestDeleteAllSavedAnalyses}
                disabled={analysisHistory.length === 0 || isDeletingAllHistory || isLoadingHistory}
              >
                {isDeletingAllHistory && <span className="button-spinner small" aria-hidden="true" />}
                <span>{isDeletingAllHistory ? "Deleting all..." : "Delete all"}</span>
              </button>
            </div>
          </div>

          {isLoadingHistory && analysisHistory.length === 0 ? (
            <div className="history-loading" aria-live="polite">
              <div className="history-loading-header">
                <span className="button-spinner" aria-hidden="true" />
                <strong>Loading previous analyses</strong>
              </div>
              <div className="history-skeleton-list" aria-hidden="true">
                {[1, 2, 3].map((item) => (
                  <div className="history-skeleton" key={item}>
                    <span />
                    <div />
                    <strong />
                  </div>
                ))}
              </div>
            </div>
          ) : analysisHistory.length > 0 ? (
            <div className="history-list">
              {isLoadingHistory && (
                <div className="history-refreshing" aria-live="polite">
                  <span className="button-spinner small" aria-hidden="true" />
                  Refreshing saved shots...
                </div>
              )}
              {analysisHistory.map((analysis) => {
                const isOpening = openingRunId === analysis.run_id;
                const isDeleting = deletingRunId === analysis.run_id;
                const isBusy = isOpening || isDeleting || isDeletingAllHistory;
                return (
                <article className={`history-item ${isBusy ? "is-busy" : ""}`} key={analysis.run_id}>
                  <label className="compare-check">
                    <input
                      type="checkbox"
                      checked={selectedComparisonRuns.includes(analysis.run_id)}
                      disabled={isBusy}
                      onChange={() => toggleComparisonRun(analysis.run_id)}
                    />
                    <span>Compare</span>
                  </label>
                  <div className="history-main">
                    <strong>{analysis.video}</strong>
                    <span>{formatRunDate(analysis.created_at)}</span>
                  </div>
                  <div className="history-meta">
                    <span>Score {analysis.score}</span>
                    <span>{titleCase(analysis.camera_view || "side")} · {analysis.shooting_side}</span>
                  </div>
                  <div className="history-actions">
                    <button
                      className="secondary-button history-action-button"
                      type="button"
                      onClick={() => openSavedAnalysis(analysis.run_id)}
                      disabled={isBusy}
                    >
                      {isOpening && <span className="button-spinner small" aria-hidden="true" />}
                      <span>{isOpening ? "Opening..." : "Open"}</span>
                    </button>
                    <button
                      className="danger-button history-action-button"
                      type="button"
                      onClick={() => requestDeleteSavedAnalysis(analysis)}
                      disabled={isBusy}
                    >
                      {isDeleting && <span className="button-spinner small" aria-hidden="true" />}
                      <span>{isDeleting ? "Deleting..." : "Delete"}</span>
                    </button>
                  </div>
                </article>
                );
              })}
            </div>
          ) : (
            <p className="empty-state">
              {isSupabaseConfigured && !session ? "Sign in to view your saved analyses." : "No saved analyses yet."}
            </p>
          )}
        </section>
        )}

        {comparison && (
          <section className="comparison-panel">
            <div className="section-header">
              <div>
                <h2>{comparison.mode === "best_baseline" ? "Compared to Best Shot" : "Shot Comparison"}</h2>
                <p className="subtle">
                  {comparison.first.run_id} vs {comparison.second.run_id}
                </p>
              </div>
              <div className={`comparison-score ${comparison.score_delta >= 0 ? "positive" : "negative"}`}>
                {formatDelta(comparison.score_delta)}
              </div>
            </div>

            <div className="comparison-summary">
              <div>
                <span>{comparison.baseline_label || "First Shot"}</span>
                <strong>{comparison.first.score}</strong>
              </div>
              <div>
                <span>{comparison.current_label || "Second Shot"}</span>
                <strong>{comparison.second.score}</strong>
              </div>
            </div>

            <div className="comparison-table">
              {comparison.metrics.map((metric) => (
                <div className="comparison-row" key={metric.key}>
                  <span>{metric.label}</span>
                  <strong>{formatMetric(metric.first)}</strong>
                  <strong>{formatMetric(metric.second)}</strong>
                  <strong className={metric.delta === null || metric.delta === undefined ? "" : metric.delta >= 0 ? "positive-text" : "negative-text"}>
                    {formatDelta(metric.delta)}
                  </strong>
                </div>
              ))}
            </div>

            <div className="comparison-coaching">
              <div>
                <h3>First Shot Priorities</h3>
                {comparison.coaching_items.first.map((item) => (
                  <p key={`${comparison.first.run_id}-${item.metric}-${item.title}`}>{item.title}</p>
                ))}
              </div>
              <div>
                <h3>Second Shot Priorities</h3>
                {comparison.coaching_items.second.map((item) => (
                  <p key={`${comparison.second.run_id}-${item.metric}-${item.title}`}>{item.title}</p>
                ))}
              </div>
            </div>
          </section>
        )}

        {result && (
          <div className="results-grid stage-card" ref={resultsSectionRef}>
            <section className="score-panel">
              <div>
                <p className="eyebrow">Shot Score</p>
                <div className="score">{result.score}</div>
                <p className="subtle">Shooting side: {result.shooting_side}</p>
                {result.metrics?.shooting_side_source && (
                  <p className="subtle">
                    Hand source: {result.metrics.shooting_side_source}
                    {result.metrics.shooting_side_source === "auto" && result.metrics.shooting_side_confidence !== undefined
                      ? ` (${formatMetric(result.metrics.shooting_side_confidence, 2)} confidence)`
                      : ""}
                  </p>
                )}
                <p className="subtle">Camera view: {titleCase(result.camera_view || "side")}</p>
                <button
                  className="best-shot-button"
                  type="button"
                  onClick={compareToBestShot}
                  disabled={isComparingBest}
                >
                  {isComparingBest && <span className="button-spinner small" aria-hidden="true" />}
                  <span>{isComparingBest ? "Comparing..." : "Compare to Best"}</span>
                </button>
              </div>
            </section>

            <section className="feedback-panel">
              <h2>Feedback</h2>
              <ul>
                {result.feedback.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </section>

            {result.quality_warnings?.length > 0 && (
              <section className="quality-panel wide">
                <div className="section-header">
                  <div>
                    <h2>Quality Warnings</h2>
                    <p className="subtle">These items can affect how much you should trust the score.</p>
                  </div>
                  <strong className={`quality-badge ${result.reliability?.label || "medium"}`}>
                    {result.quality_warnings.length}
                  </strong>
                </div>
                <div className="quality-list">
                  {result.quality_warnings.map((warning) => (
                    <article className={`quality-warning ${warning.severity}`} key={`${warning.title}-${warning.detail}`}>
                      <div>
                        <span>{warning.severity}</span>
                        <strong>{warning.title}</strong>
                      </div>
                      <p>{warning.detail}</p>
                      <p>{warning.suggestion}</p>
                    </article>
                  ))}
                </div>
              </section>
            )}

            {result.reliability && (
              <section className="reliability-panel wide">
                <div className="section-header">
                  <div>
                    <h2>Analysis Reliability</h2>
                    <p className="subtle">Camera view: {titleCase(result.camera_view || "side")}</p>
                  </div>
                  <div className={`reliability-score ${result.reliability.label}`}>
                    <strong>{result.reliability.score}</strong>
                    <span>{result.reliability.label}</span>
                  </div>
                </div>
                <div className="reliability-list">
                  {result.reliability.checks.map((check) => (
                    <div className={`reliability-check ${check.status}`} key={check.name}>
                      <strong>{check.name}</strong>
                      <span>{check.detail}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {ballTracking && (
              <section className="ball-panel wide">
                <div className="section-header">
                  <div>
                    <p className="eyebrow dark">Beta</p>
                    <h2>Ball Tracking</h2>
                    <p className="subtle">
                      {ballTracking.note || "Ball tracking is informational and does not affect the score yet."}
                    </p>
                    {ballTracking.detector_message && <p className="subtle">{ballTracking.detector_message}</p>}
                  </div>
                  <strong className={`ball-status ${ballTracking.status || "not_detected"}`}>
                    {titleCase(ballTracking.status || "not_detected")}
                  </strong>
                </div>
                <div className="ball-metrics">
                  <div>
                    <span>Detector</span>
                    <strong>{titleCase(ballTracking.detector_backend || "unknown")}</strong>
                  </div>
                  {ballTracking.roboflow_model_id && (
                    <div>
                      <span>Roboflow model</span>
                      <strong>{ballTracking.roboflow_model_id}</strong>
                    </div>
                  )}
                  <div>
                    <span>YOLO image size</span>
                    <strong>{formatMetric(ballTracking.yolo_image_size)}</strong>
                  </div>
                  <div>
                    <span>Detector confidence</span>
                    <strong>{formatMetric(ballTracking.yolo_confidence)}</strong>
                  </div>
                  <div>
                    <span>Close visibility</span>
                    <strong>{formatMetric((ballTracking.close_visibility_ratio || 0) * 100, 1)}%</strong>
                  </div>
                  <div>
                    <span>Raw visibility</span>
                    <strong>{formatMetric((ballTracking.visibility_ratio || 0) * 100, 1)}%</strong>
                  </div>
                  <div>
                    <span>Ball release frame</span>
                    <strong>{formatMetric(ballTracking.ball_release_frame)}</strong>
                  </div>
                  <div>
                    <span>Pose vs ball release</span>
                    <strong>{formatDelta(ballTracking.release_frame_delta)}</strong>
                  </div>
                  <div>
                    <span>Arc height</span>
                    <strong>{formatMetric(ballTracking.arc_height)}</strong>
                  </div>
                  <div>
                    <span>Avg wrist distance</span>
                    <strong>{formatMetric(ballTracking.avg_wrist_distance)}</strong>
                  </div>
                  <div>
                    <span>Tracked frames</span>
                    <strong>
                      {formatMetric(ballTracking.close_detected_frames || ballTracking.detected_frames)}/{formatMetric(ballTracking.total_frames)}
                    </strong>
                  </div>
                  {ballTracking.roboflow_requested_frames !== undefined && (
                    <div>
                      <span>Remote frames</span>
                      <strong>
                        {formatMetric(ballTracking.roboflow_requested_frames)}
                        {ballTracking.roboflow_max_workers ? ` / ${formatMetric(ballTracking.roboflow_max_workers)} workers` : ""}
                      </strong>
                    </div>
                  )}
                  <div>
                    <span>Sources</span>
                    <strong>
                      {ballTracking.candidate_source_counts
                        ? Object.entries(ballTracking.candidate_source_counts)
                            .map(([source, count]) => `${source}: ${count}`)
                            .join(", ")
                        : "N/A"}
                    </strong>
                  </div>
                </div>
              </section>
            )}

            <section className="improvement-panel wide">
              <h2>What do I need to improve?</h2>
              <div className="improvement-list">
                {improvementItems.map((item, index) => (
                  <article className="improvement-item" key={`${item.metric}-${item.title}`}>
                    <div className="improvement-header">
                      <span>Priority {index + 1}</span>
                      <strong className={`severity ${item.severity}`}>{item.severity}</strong>
                    </div>
                    <h3>{item.title}</h3>
                    <dl>
                      <div>
                        <dt>Metric</dt>
                        <dd>{titleCase(item.metric)}: {formatCoachingValue(item.value)}</dd>
                      </div>
                      <div>
                        <dt>Target</dt>
                        <dd>{item.target}</dd>
                      </div>
                      <div>
                        <dt>Where it happens</dt>
                        <dd>{titleCase(item.phase)} - {formatFrameRange(item)}</dd>
                      </div>
                    </dl>
                    <p>{item.why_it_matters}</p>
                    <div className="drill">
                      <span>Drill</span>
                      <p>{item.drill}</p>
                    </div>
                    <div className="improvement-actions">
                      <button
                        className="watch-button"
                        type="button"
                        disabled={!annotatedVideoUrl}
                        onClick={() => watchCoachingMoment(item)}
                      >
                        Watch this moment
                      </button>
                      <button
                        className="evidence-button"
                        type="button"
                        disabled={!getEvidenceFrameUrl(item)}
                        onClick={() => openEvidenceFrame(item)}
                      >
                        View frame
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="metrics-panel">
              <h2>Key Metrics</h2>
              <div className="metric-grid">
                {coreMetrics.map(([label, value]) => (
                  <div className="metric" key={label}>
                    <span>{label}</span>
                    <strong>{formatMetric(value)}</strong>
                  </div>
                ))}
              </div>
            </section>

            <section className="phases-panel">
              <h2>Phases</h2>
              <div className="phase-list">
                {Object.entries(result.phases).map(([name, phase]) => (
                  <div className="phase" key={name}>
                    <span>{titleCase(name)}</span>
                    <strong>
                      {phase.start_frame}-{phase.end_frame}
                    </strong>
                  </div>
                ))}
              </div>
            </section>

            {chartUrl && (
              <section className="media-panel wide">
                <h2>Angle Chart</h2>
                <img src={chartUrl} alt="Phase shaded joint angle chart" />
              </section>
            )}

            {followThroughDebugChartUrl && (
              <section className="media-panel wide">
                <h2>Follow-Through Debug</h2>
                <img src={followThroughDebugChartUrl} alt="Follow-through wrist, shoulder, and elbow debug chart" />
              </section>
            )}

            {annotatedVideoUrl && (
              <section className="media-panel wide" ref={annotatedVideoSectionRef}>
                <h2>Annotated Video</h2>
                <video
                  key={annotatedVideoUrl}
                  ref={annotatedVideoRef}
                  src={annotatedVideoUrl}
                  controls
                  preload="metadata"
                />
                <a className="media-link" href={annotatedVideoUrl} target="_blank" rel="noreferrer">
                  Open annotated video
                </a>
              </section>
            )}
          </div>
        )}
          </div>
        )}
        {deletePrompt && (
          <div className="modal-backdrop" role="presentation" onClick={() => setDeletePrompt(null)}>
            <section
              className="delete-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby="delete-modal-title"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="delete-modal-mark" aria-hidden="true">!</div>
              <div>
                <p className="eyebrow">Confirm delete</p>
                <h2 id="delete-modal-title">{deletePrompt.title}</h2>
                <p>{deletePrompt.detail}</p>
              </div>
              <div className="delete-modal-actions">
                <button
                  className="secondary-button cancel-button"
                  type="button"
                  onClick={() => setDeletePrompt(null)}
                  disabled={isDeletingAllHistory || Boolean(deletingRunId)}
                >
                  Cancel
                </button>
                <button
                  className="danger-button"
                  type="button"
                  onClick={confirmDeletePrompt}
                  disabled={isDeletingAllHistory || Boolean(deletingRunId)}
                >
                  {(isDeletingAllHistory || Boolean(deletingRunId)) && (
                    <span className="button-spinner small" aria-hidden="true" />
                  )}
                  <span>
                    {isDeletingAllHistory || Boolean(deletingRunId) ? "Deleting..." : deletePrompt.confirmLabel}
                  </span>
                </button>
              </div>
            </section>
          </div>
        )}
        {selectedEvidenceItem && selectedEvidenceFrameUrl && (
          <div className="modal-backdrop" role="presentation" onClick={() => setSelectedEvidenceItem(null)}>
            <section
              className="evidence-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby="evidence-modal-title"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="section-header">
                <div>
                  <p className="eyebrow">Evidence Frame</p>
                  <h2 id="evidence-modal-title">{selectedEvidenceItem.title}</h2>
                  <p className="subtle">
                    {titleCase(selectedEvidenceItem.phase)} - {formatFrameRange(selectedEvidenceItem)}
                  </p>
                </div>
                <button className="secondary-button" type="button" onClick={() => setSelectedEvidenceItem(null)}>
                  Close
                </button>
              </div>
              <img src={selectedEvidenceFrameUrl} alt={`Evidence frame for ${selectedEvidenceItem.title}`} />
              <div className="evidence-details">
                <div>
                  <span>Metric</span>
                  <strong>
                    {titleCase(selectedEvidenceItem.metric)}: {formatCoachingValue(selectedEvidenceItem.value)}
                  </strong>
                </div>
                <div>
                  <span>Target</span>
                  <strong>{selectedEvidenceItem.target}</strong>
                </div>
              </div>
              <p>{selectedEvidenceItem.why_it_matters}</p>
            </section>
          </div>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
