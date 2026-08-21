import React from "react";
import { formatElapsedTime, titleCase } from "../lib/formatters";

export function LoadingPage({ analysisElapsedSeconds, cameraView, shootingSide }) {
  return (
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
  );
}
