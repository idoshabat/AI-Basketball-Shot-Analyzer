import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD ? "https://ai-basketball-shot-analyzer.onrender.com" : "http://127.0.0.1:8000");

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

function App() {
  const [file, setFile] = useState(null);
  const [includeAnnotatedVideo, setIncludeAnnotatedVideo] = useState(true);
  const [result, setResult] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [selectedComparisonRuns, setSelectedComparisonRuns] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const annotatedVideoRef = useRef(null);
  const annotatedVideoSectionRef = useRef(null);

  const chartUrl = result?.output_urls?.angles_chart
    ? `${API_BASE_URL}${result.output_urls.angles_chart}`
    : null;
  const followThroughDebugChartUrl = result?.output_urls?.follow_through_debug_chart
    ? `${API_BASE_URL}${result.output_urls.follow_through_debug_chart}`
    : null;
  const annotatedVideoUrl = result?.output_urls?.annotated_video
    ? `${API_BASE_URL}${result.output_urls.annotated_video}`
    : null;

  const coreMetrics = useMemo(() => {
    if (!result) {
      return [];
    }

    return [
      ["Release Frame", result.metrics.release_frame],
      ["Release Confidence", result.metrics.release_confidence],
      ["Release Confidence Label", result.metrics.release_confidence_label],
      ["Release Elbow Angle", result.metrics.release_elbow_angle],
      ["Follow-Through End", result.metrics.follow_through_end_frame],
      ["Follow-Through Ratio", result.metrics.follow_through_ratio],
      ["Hip Rise", result.metrics.hip_rise],
      ["Ankle Lift", result.metrics.ankle_lift],
    ];
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
    setIsLoadingHistory(true);
    try {
      const response = await fetch(`${API_BASE_URL}/analyses?limit=10`);
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
      const response = await fetch(`${API_BASE_URL}/analyses/${runId}`);
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
      const response = await fetch(`${API_BASE_URL}/analyses/compare?${params}`);
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

  async function deleteSavedAnalysis(analysis) {
    const shouldDelete = window.confirm(`Delete analysis "${analysis.run_id}"? This cannot be undone.`);
    if (!shouldDelete) {
      return;
    }

    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/analyses/${analysis.run_id}`, {
        method: "DELETE",
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
    loadAnalysisHistory();
  }, []);

  async function analyzeShot(event) {
    event.preventDefault();
    if (!file) {
      setError("Choose a video before analyzing.");
      return;
    }

    setIsAnalyzing(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    const params = new URLSearchParams({
      save_chart: "true",
      save_annotated_video: includeAnnotatedVideo ? "true" : "false",
      save_report: "true",
    });

    try {
      const response = await fetch(`${API_BASE_URL}/analyze-shot?${params}`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed.");
      }

      setResult(data);
      loadAnalysisHistory();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="upload-panel">
          <div>
            <p className="eyebrow">AI Basketball Shot Analyzer</p>
            <h1>Upload a shot and get a motion report.</h1>
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

        {isAnalyzing && (
          <section className="loading-panel" aria-live="polite">
            <div className="loading-spinner" aria-hidden="true" />
            <div>
              <h2>Analyzing your shot</h2>
              <p>
                The backend is processing pose detection, metrics, charts, and optional annotated video. This can take a few
                minutes on Render's low-CPU instance.
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
            <p className="empty-state">No saved analyses yet.</p>
          )}
        </section>

        {comparison && (
          <section className="comparison-panel">
            <div className="section-header">
              <div>
                <h2>Shot Comparison</h2>
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
                <span>First Shot</span>
                <strong>{comparison.first.score}</strong>
              </div>
              <div>
                <span>Second Shot</span>
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
