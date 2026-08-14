import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

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

function App() {
  const [file, setFile] = useState(null);
  const [includeAnnotatedVideo, setIncludeAnnotatedVideo] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const annotatedVideoRef = useRef(null);
  const annotatedVideoSectionRef = useRef(null);

  const chartUrl = result?.output_urls?.angles_chart
    ? `${API_BASE_URL}${result.output_urls.angles_chart}`
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
              {isAnalyzing ? "Analyzing..." : "Analyze Shot"}
            </button>
          </form>

          {error && <p className="error-text">{error}</p>}
        </div>

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
