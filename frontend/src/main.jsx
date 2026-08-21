import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { isSupabaseConfigured, supabase } from "./supabaseClient";
import {
  API_BASE_URL,
  CAMERA_VIEW_GUIDANCE,
  SAMPLE_RESULTS,
  clearAccessMode,
  readAccessMode,
  writeAccessMode,
} from "./lib/appConfig";
import { groupFeedbackItems } from "./lib/feedback";
import {
  formatCoachingValue,
  formatDelta,
  formatFrameRange,
  formatMetric,
  getRequestErrorMessage,
  getUserName,
  resolveOutputUrl,
  titleCase,
} from "./lib/formatters";
import { cacheReport, getReportId, getReportPath, getSampleById, readCachedReport } from "./lib/reportCache";
import { LoadingPage } from "./pages/LoadingPage";
import { HistoryPage } from "./pages/HistoryPage";
import { WelcomePage } from "./pages/WelcomePage";
import "./styles.css";

function parseRoute(pathname = globalThis.window?.location?.pathname || "/welcome") {
  const normalizedPath = pathname === "/" ? "/welcome" : pathname;
  const analysisMatch = normalizedPath.match(/^\/analysis\/([^/]+)$/);
  if (analysisMatch) {
    return { page: "report", analysisId: decodeURIComponent(analysisMatch[1]) };
  }

  if (normalizedPath === "/history") {
    return { page: "history", analysisId: null };
  }

  if (normalizedPath === "/loading") {
    return { page: "loading", analysisId: null };
  }

  if (normalizedPath === "/upload") {
    return { page: "capture", analysisId: null };
  }

  return { page: "welcome", analysisId: null };
}

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    console.error("App render failed", error);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="app-shell">
          <section className="workspace">
            <section className="welcome-panel">
              <div className="welcome-copy">
                <p className="eyebrow">AI Basketball Shot Analyzer</p>
                <h1>The app hit a startup error.</h1>
                <p>{this.state.error.message || "Refresh the page and try again."}</p>
              </div>
            </section>
          </section>
        </main>
      );
    }

    return this.props.children;
  }
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
  const [isRestoringReport, setIsRestoringReport] = useState(false);
  const [session, setSession] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(!isSupabaseConfigured);
  const [isSigningIn, setIsSigningIn] = useState(false);
  const [route, setRoute] = useState(() => parseRoute());
  const [hasEnteredApp, setHasEnteredApp] = useState(() => {
    return readAccessMode() === "guest";
  });
  const [isGuestMode, setIsGuestMode] = useState(() => {
    return readAccessMode() === "guest";
  });
  const [selectedEvidenceItem, setSelectedEvidenceItem] = useState(null);
  const [deletePrompt, setDeletePrompt] = useState(null);
  const [actionToast, setActionToast] = useState(null);
  const annotatedVideoRef = useRef(null);
  const annotatedVideoSectionRef = useRef(null);
  const resultsSectionRef = useRef(null);
  const authUserIdRef = useRef(null);
  const pendingHistoryRef = useRef([]);

  const appView = isAnalyzing ? "analyzing" : route.page;
  const canUseApp = hasEnteredApp || Boolean(session);
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
  const feedbackGroups = useMemo(() => groupFeedbackItems(result?.feedback || []), [result?.feedback]);

  function navigate(path, options = {}) {
    const nextRoute = parseRoute(path);
    if (window.location.pathname !== path) {
      window.history[options.replace ? "replaceState" : "pushState"]({}, "", path);
    }
    setRoute(nextRoute);
  }

  function showActionToast(title, detail = "", tone = "info") {
    setActionToast({
      id: Date.now(),
      title,
      detail,
      tone,
    });
  }

  function setPendingHistory(nextPending) {
    pendingHistoryRef.current = nextPending;
  }

  function mergeHistoryWithPending(serverHistory, pendingHistory = pendingHistoryRef.current) {
    const normalizedServerHistory = serverHistory.map((analysis) => ({
      ...analysis,
      is_pending: analysis.persistence_status === "saving_media",
      created_at: analysis.persistence_status === "saving_media" ? "Saving media..." : analysis.created_at,
    }));
    const readyServerRunIds = new Set(
      normalizedServerHistory
        .filter((analysis) => analysis.persistence_status !== "saving_media")
        .map((analysis) => analysis.run_id)
    );
    const activePending = pendingHistory.filter((analysis) => {
      const isConfirmed = readyServerRunIds.has(analysis.run_id);
      const isExpired = Date.now() - analysis.pending_started_at > 180000;
      return !isConfirmed && !isExpired;
    });
    const activePendingRunIds = new Set(activePending.map((analysis) => analysis.run_id));
    const visibleServerHistory = normalizedServerHistory.filter((analysis) => !activePendingRunIds.has(analysis.run_id));

    if (activePending.length !== pendingHistoryRef.current.length) {
      setPendingHistory(activePending);
    }

    return [...activePending, ...visibleServerHistory].slice(0, 10);
  }

  function setActiveReport(report, options = {}) {
    setResult(report);
    cacheReport(report);
    if (options.navigate !== false) {
      navigate(getReportPath(report), { replace: options.replace });
    }
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
      setPendingHistory([]);
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

      setAnalysisHistory(mergeHistoryWithPending(data));
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

      setActiveReport(data);
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
        navigate("/upload", { replace: true });
      }
      setSelectedComparisonRuns((currentRuns) => currentRuns.filter((runId) => runId !== analysis.run_id));
      if (comparison?.first?.run_id === analysis.run_id || comparison?.second?.run_id === analysis.run_id) {
        setComparison(null);
      }
      setPendingHistory(pendingHistoryRef.current.filter((item) => item.run_id !== analysis.run_id));
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
      setPendingHistory([]);
      setSelectedComparisonRuns([]);
      setComparison(null);
      setResult(null);
      setSelectedEvidenceItem(null);
      navigate("/upload", { replace: true });
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
    const handlePopState = () => {
      setRoute(parseRoute());
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    if (!isAuthReady) {
      return;
    }

    if (!hasEnteredApp && route.page !== "welcome") {
      navigate("/welcome", { replace: true });
      return;
    }

    if (hasEnteredApp && route.page === "welcome") {
      navigate("/upload", { replace: true });
      return;
    }

    if (hasEnteredApp && route.page === "loading" && !isAnalyzing) {
      navigate("/upload", { replace: true });
    }
  }, [isAuthReady, hasEnteredApp, route.page, isAnalyzing]);

  useEffect(() => {
    if (!isAuthReady || !hasEnteredApp || route.page !== "report" || !route.analysisId) {
      return undefined;
    }

    if (getReportId(result) === route.analysisId) {
      return undefined;
    }

    const abortController = new AbortController();

    async function restoreRouteReport() {
      setIsRestoringReport(true);
      setError("");
      try {
        const cachedReport = readCachedReport(route.analysisId);
        if (cachedReport) {
          setResult(cachedReport);
          return;
        }

        const sample = getSampleById(route.analysisId);
        const response = await fetch(sample ? sample.path : `${API_BASE_URL}/analyses/${route.analysisId}`, {
          headers: sample ? {} : authHeaders,
          signal: abortController.signal,
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Could not restore this report.");
        }

        setResult(data);
        cacheReport(data);
      } catch (restoreError) {
        if (restoreError.name === "AbortError") {
          return;
        }

        setError(getRequestErrorMessage(restoreError));
        showActionToast("Report restore failed", "Returning to the upload page.", "danger");
        navigate("/upload", { replace: true });
      } finally {
        if (!abortController.signal.aborted) {
          setIsRestoringReport(false);
        }
      }
    }

    restoreRouteReport();
    return () => abortController.abort();
  }, [isAuthReady, hasEnteredApp, route.page, route.analysisId, session?.access_token]);

  useEffect(() => {
    if (!supabase) {
      return;
    }

    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      authUserIdRef.current = data.session?.user?.id || null;
      if (data.session) {
        setHasEnteredApp(true);
        setIsGuestMode(false);
        writeAccessMode("signed_in");
      }
      setIsAuthReady(true);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, nextSession) => {
      const previousUserId = authUserIdRef.current;
      const nextUserId = nextSession?.user?.id || null;
      const userChanged = Boolean(previousUserId && nextUserId && previousUserId !== nextUserId);
      const shouldClearWorkspace = event === "SIGNED_OUT" || userChanged;

      setSession(nextSession);
      if (nextSession) {
        setHasEnteredApp(true);
        setIsGuestMode(false);
        writeAccessMode("signed_in");
      }
      if (shouldClearWorkspace) {
        setSelectedComparisonRuns([]);
        setComparison(null);
        setResult(null);
        setSelectedEvidenceItem(null);
        setPendingHistory([]);
      }
      authUserIdRef.current = nextUserId;
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
        redirectTo: `${window.location.origin}/upload`,
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
    clearAccessMode();
    navigate("/welcome", { replace: true });
  }

  function continueAsGuest() {
    setError("");
    setHasEnteredApp(true);
    setIsGuestMode(true);
    writeAccessMode("guest");
    navigate("/upload", { replace: true });
    showActionToast("Guest mode started", "You can analyze a shot without signing in.", "success");
  }

  function startNewAnalysis() {
    setResult(null);
    setComparison(null);
    setSelectedEvidenceItem(null);
    setError("");
    navigate("/upload");
    showActionToast("New analysis", "Upload area is ready.");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function openHistoryView() {
    setError("");
    navigate("/history");
    showActionToast("Shot library", "Opening your recent analyses.");
    loadAnalysisHistory();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function openCaptureView() {
    setError("");
    navigate("/upload");
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
    navigate("/loading");
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

      setActiveReport(data);
      setSelectedEvidenceItem(null);
      showActionToast("Analysis complete", `Shot score ${data.score}. Report is ready.`, "success");
      const optimisticSummary = {
        run_id: data.run_id,
        created_at: data.persistence === "supabase_pending" ? "Saving to cloud..." : "Saving...",
        score: data.score,
        shooting_side: data.shooting_side,
        camera_view: data.camera_view,
        video: file.name,
        pending_started_at: Date.now(),
        is_pending: data.persistence === "supabase_pending",
      };
      if (data.persistence === "supabase_pending") {
        setPendingHistory([
          optimisticSummary,
          ...pendingHistoryRef.current.filter((analysis) => analysis.run_id !== data.run_id),
        ].slice(0, 10));
      }
      setAnalysisHistory((currentHistory) => {
        return [optimisticSummary, ...currentHistory.filter((analysis) => analysis.run_id !== data.run_id)].slice(0, 10);
      });
      if (data.persistence === "supabase_pending") {
        [2500, 7000, 15000, 30000].forEach((delay) => {
          window.setTimeout(loadAnalysisHistory, delay);
        });
      } else {
        loadAnalysisHistory();
      }
    } catch (requestError) {
      setError(getRequestErrorMessage(requestError));
      navigate("/upload", { replace: true });
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

      setActiveReport(data);
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
        {!canUseApp ? (
          <WelcomePage
            error={error}
            isSigningIn={isSigningIn}
            isSupabaseConfigured={isSupabaseConfigured}
            onContinueAsGuest={continueAsGuest}
            onSignInWithGoogle={signInWithGoogle}
          />
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
          <LoadingPage
            analysisElapsedSeconds={analysisElapsedSeconds}
            cameraView={cameraView}
            shootingSide={shootingSide}
          />
        )}

        {appView === "report" && isRestoringReport && !result && (
          <section className="analysis-loading-screen" aria-live="polite">
            <div className="loading-orbit" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div className="loading-copy">
              <p className="eyebrow">Opening report</p>
              <h1>Restoring your shot analysis.</h1>
              <p>Loading the saved report, evidence frames, charts, and annotated video links.</p>
            </div>
            <div className="loading-dashboard">
              <div className="loading-steps" aria-hidden="true">
                <span>Report lookup</span>
                <span>Media links</span>
                <span>Metrics</span>
                <span>Ready</span>
              </div>
              <div className="loading-bar" aria-hidden="true">
                <span />
              </div>
            </div>
          </section>
        )}

        {appView === "report" && result && (
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
          <HistoryPage
            analysisHistory={analysisHistory}
            deletingRunId={deletingRunId}
            isComparing={isComparing}
            isDeletingAllHistory={isDeletingAllHistory}
            isGuestMode={isGuestMode}
            isLoadingHistory={isLoadingHistory}
            openingRunId={openingRunId}
            onCompareSelectedRuns={compareSelectedRuns}
            onDeleteAll={requestDeleteAllSavedAnalyses}
            onDeleteOne={requestDeleteSavedAnalysis}
            onLoadHistory={loadAnalysisHistory}
            onOpenAnalysis={openSavedAnalysis}
            onToggleComparisonRun={toggleComparisonRun}
            selectedComparisonRuns={selectedComparisonRuns}
            session={session}
          />
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

        {appView === "report" && result && (
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
              <div className="section-header">
                <div>
                  <h2>Feedback</h2>
                  <p className="subtle">Grouped so strengths, fixes, and in-between notes are easier to read.</p>
                </div>
                <strong className="feedback-total">{result.feedback.length}</strong>
              </div>
              <div className="feedback-groups">
                {feedbackGroups.map((group) => (
                  <article className={`feedback-group ${group.id}`} key={group.id}>
                    <div className="feedback-group-header">
                      <span>{group.label}</span>
                      <div>
                        <h3>{group.title}</h3>
                        <p>{group.description}</p>
                      </div>
                      <strong>{group.items.length}</strong>
                    </div>
                    {group.items.length > 0 ? (
                      <ul>
                        {group.items.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="empty-feedback">No items in this group.</p>
                    )}
                  </article>
                ))}
              </div>
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

            {/* {ballTracking && (
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
            )} */}

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

const rootElement = document.getElementById("root");

if (rootElement) {
  createRoot(rootElement).render(
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  );
}
