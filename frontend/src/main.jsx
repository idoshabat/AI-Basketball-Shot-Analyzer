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

  return `${API_BASE_URL}${path}`;
}

function getUserName(session) {
  return session?.user?.user_metadata?.full_name || session?.user?.email || "Signed-in player";
}

function App() {
  const [file, setFile] = useState(null);
  const [cameraView, setCameraView] = useState("side");
  const [includeAnnotatedVideo, setIncludeAnnotatedVideo] = useState(true);
  const [result, setResult] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [selectedComparisonRuns, setSelectedComparisonRuns] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [isComparingBest, setIsComparingBest] = useState(false);
  const [session, setSession] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(!isSupabaseConfigured);
  const [isSigningIn, setIsSigningIn] = useState(false);
  const annotatedVideoRef = useRef(null);
  const annotatedVideoSectionRef = useRef(null);

  const chartUrl = resolveOutputUrl(result?.output_urls?.angles_chart);
  const followThroughDebugChartUrl = resolveOutputUrl(result?.output_urls?.follow_through_debug_chart);
  const annotatedVideoUrl = resolveOutputUrl(result?.output_urls?.annotated_video);
  const authHeaders = session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {};

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
    ];

    if (result.camera_view !== "side") {
      metrics.push(
        ["Left Shin Vertical Error", result.metrics.left_shin_vertical_error],
        ["Right Shin Vertical Error", result.metrics.right_shin_vertical_error],
        ["Shin Parallel Error", result.metrics.shin_parallel_error],
        ["Foot Parallel Error", result.metrics.foot_parallel_error],
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
      return;
    }

    video.currentTime = Math.max(0, (item.start_frame - 1) / fps);
    videoSection?.scrollIntoView({ behavior: "smooth", block: "start" });
    video.play().catch(() => {});
  }

  async function loadAnalysisHistory() {
    if (isSupabaseConfigured && !session) {
      setAnalysisHistory([]);
      return;
    }

    setIsLoadingHistory(true);
    try {
      const response = await fetch(`${API_BASE_URL}/analyses?limit=10`, {
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not load recent analyses.");
      }

      setAnalysisHistory(data);
    } catch (historyError) {
      setError(historyError.message);
    } finally {
      setIsLoadingHistory(false);
    }
  }

  async function openSavedAnalysis(runId) {
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/analyses/${runId}`, {
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not open saved analysis.");
      }

      setResult(data);
    } catch (savedAnalysisError) {
      setError(savedAnalysisError.message);
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
    } catch (compareError) {
      setError(compareError.message);
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
    try {
      const response = await fetch(`${API_BASE_URL}/analyses/${result.run_id}/compare-best`, {
        headers: authHeaders,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not compare to your best shot.");
      }

      setComparison(data);
    } catch (bestComparisonError) {
      setError(bestComparisonError.message);
    } finally {
      setIsComparingBest(false);
    }
  }

  async function deleteSavedAnalysis(analysis) {
    const shouldDelete = window.confirm(`Delete analysis "${analysis.run_id}"? This cannot be undone.`);
    if (!shouldDelete) {
      return;
    }

    setError("");
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
      }
      setSelectedComparisonRuns((currentRuns) => currentRuns.filter((runId) => runId !== analysis.run_id));
      if (comparison?.first?.run_id === analysis.run_id || comparison?.second?.run_id === analysis.run_id) {
        setComparison(null);
      }
      loadAnalysisHistory();
    } catch (deleteError) {
      setError(deleteError.message);
    }
  }

  useEffect(() => {
    if (!supabase) {
      return;
    }

    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setIsAuthReady(true);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setSelectedComparisonRuns([]);
      setComparison(null);
      setResult(null);
    });

    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (isAuthReady) {
      loadAnalysisHistory();
    }
  }, [isAuthReady, session?.access_token]);

  async function signInWithGoogle() {
    if (!supabase) {
      setError("Supabase is not configured yet.");
      return;
    }

    setIsSigningIn(true);
    setError("");
    const { error: signInError } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: window.location.origin,
      },
    });

    if (signInError) {
      setError(signInError.message);
      setIsSigningIn(false);
    }
  }

  async function signOut() {
    if (!supabase) {
      return;
    }

    setError("");
    await supabase.auth.signOut();
  }

  async function analyzeShot(event) {
    event.preventDefault();
    if (isSupabaseConfigured && !session) {
      setError("Sign in with Google before analyzing so the shot is saved to your account.");
      return;
    }

    if (!file) {
      setError("Choose a video before analyzing.");
      return;
    }

    const selectedCameraView = event.currentTarget.elements.camera_view?.value || cameraView;
    setIsAnalyzing(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("camera_view", selectedCameraView);

    const params = new URLSearchParams({
      save_chart: "true",
      save_annotated_video: includeAnnotatedVideo ? "true" : "false",
      save_report: "true",
      camera_view: selectedCameraView,
    });

    try {
      const response = await fetch(`${API_BASE_URL}/analyze-shot?${params}`, {
        method: "POST",
        headers: {
          ...authHeaders,
          "X-Camera-View": selectedCameraView,
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

      setResult(data);
      loadAnalysisHistory();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function loadSampleResult() {
    setError("");
    setComparison(null);
    try {
      const response = await fetch("/samples/sample-analysis.json");
      const data = await response.json();
      if (!response.ok) {
        throw new Error("Could not load sample analysis.");
      }

      setResult(data);
    } catch (sampleError) {
      setError(sampleError.message);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="upload-panel">
          <div>
            <p className="eyebrow">AI Basketball Shot Analyzer</p>
            <h1>Upload a shot and get a motion report.</h1>
            <div className="auth-panel">
              {isSupabaseConfigured ? (
                session ? (
                  <>
                    <div>
                      <strong>{getUserName(session)}</strong>
                      <span>Your analyses and comparisons are scoped to this account.</span>
                    </div>
                    <button className="secondary-button" type="button" onClick={signOut}>
                      Sign Out
                    </button>
                  </>
                ) : (
                  <>
                    <div>
                      <strong>Personal shot history</strong>
                      <span>Sign in to save analyses under your own account.</span>
                    </div>
                    <button className="secondary-button" type="button" onClick={signInWithGoogle} disabled={isSigningIn}>
                      {isSigningIn ? "Opening Google..." : "Sign in with Google"}
                    </button>
                  </>
                )
              ) : (
                <div>
                  <strong>Guest mode</strong>
                  <span>Add Supabase env vars to enable Google sign-in and per-user history.</span>
                </div>
              )}
            </div>
          </div>

          <form onSubmit={analyzeShot} className="upload-form">
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
            <button className="sample-button" type="button" onClick={loadSampleResult}>
              Load Sample Result
            </button>
            <p className="processing-note">
              Analysis can take a few minutes on the hosted backend, especially when annotated video is enabled.
            </p>
          </form>

          {error && <p className="error-text">{error}</p>}
        </div>

        {isAnalyzing && (
          <section className="loading-panel" aria-live="polite">
            <div className="loading-spinner" aria-hidden="true" />
            <div>
              <h2>Analyzing your shot</h2>
              <p>
                The backend is processing pose detection, metrics, charts, and optional annotated video. This can take a few
                minutes on Render's low-CPU instance. Camera view: {titleCase(cameraView)}.
              </p>
              <div className="loading-bar" aria-hidden="true">
                <span />
              </div>
            </div>
          </section>
        )}

        <section className="history-panel">
          <div className="section-header">
            <h2>Recent Analyses</h2>
            <div className="header-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={compareSelectedRuns}
                disabled={isComparing || selectedComparisonRuns.length !== 2}
              >
                {isComparing ? "Comparing..." : "Compare"}
              </button>
              <button className="secondary-button" type="button" onClick={loadAnalysisHistory} disabled={isLoadingHistory}>
                {isLoadingHistory ? "Loading..." : "Refresh"}
              </button>
            </div>
          </div>

          {analysisHistory.length > 0 ? (
            <div className="history-list">
              {analysisHistory.map((analysis) => (
                <article className="history-item" key={analysis.run_id}>
                  <label className="compare-check">
                    <input
                      type="checkbox"
                      checked={selectedComparisonRuns.includes(analysis.run_id)}
                      onChange={() => toggleComparisonRun(analysis.run_id)}
                    />
                    <span>Compare</span>
                  </label>
                  <div>
                    <strong>{analysis.video}</strong>
                    <span>{formatRunDate(analysis.created_at)}</span>
                  </div>
                  <div className="history-meta">
                    <span>Score {analysis.score}</span>
                    <span>{analysis.shooting_side}</span>
                  </div>
                  <div className="history-actions">
                    <button className="secondary-button" type="button" onClick={() => openSavedAnalysis(analysis.run_id)}>
                      Open
                    </button>
                    <button className="danger-button" type="button" onClick={() => deleteSavedAnalysis(analysis)}>
                      Delete
                    </button>
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <p className="empty-state">
              {isSupabaseConfigured && !session ? "Sign in to view your saved analyses." : "No saved analyses yet."}
            </p>
          )}
        </section>

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
          <div className="results-grid">
            <section className="score-panel">
              <div>
                <p className="eyebrow">Shot Score</p>
                <div className="score">{result.score}</div>
                <p className="subtle">Shooting side: {result.shooting_side}</p>
                <p className="subtle">Camera view: {titleCase(result.camera_view || "side")}</p>
                <button
                  className="best-shot-button"
                  type="button"
                  onClick={compareToBestShot}
                  disabled={isComparingBest}
                >
                  {isComparingBest ? "Comparing..." : "Compare to Best"}
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
                    <button
                      className="watch-button"
                      type="button"
                      disabled={!annotatedVideoUrl}
                      onClick={() => watchCoachingMoment(item)}
                    >
                      Watch this moment
                    </button>
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
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
