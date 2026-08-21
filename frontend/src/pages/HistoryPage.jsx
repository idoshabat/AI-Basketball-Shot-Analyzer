import React from "react";
import { formatRunDate, titleCase } from "../lib/formatters";
import { isSupabaseConfigured } from "../supabaseClient";

export function HistoryPage({
  analysisHistory,
  deletingRunId,
  isComparing,
  isDeletingAllHistory,
  isGuestMode,
  isLoadingHistory,
  openingRunId,
  onCompareSelectedRuns,
  onDeleteAll,
  onDeleteOne,
  onLoadHistory,
  onOpenAnalysis,
  onToggleComparisonRun,
  selectedComparisonRuns,
  session,
}) {
  return (
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
            onClick={onCompareSelectedRuns}
            disabled={isComparing || selectedComparisonRuns.length !== 2 || isDeletingAllHistory}
          >
            {isComparing && <span className="button-spinner small" aria-hidden="true" />}
            <span>{isComparing ? "Comparing..." : "Compare"}</span>
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={onLoadHistory}
            disabled={isLoadingHistory || isDeletingAllHistory}
          >
            {isLoadingHistory && <span className="button-spinner small" aria-hidden="true" />}
            <span>{isLoadingHistory ? "Loading..." : "Refresh"}</span>
          </button>
          <button
            className="danger-button ghost-danger"
            type="button"
            onClick={onDeleteAll}
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
            const isPendingSave = Boolean(analysis.is_pending);
            const isBusy = isOpening || isDeleting || isDeletingAllHistory || isPendingSave;
            return (
              <article className={`history-item ${isBusy ? "is-busy" : ""}`} key={analysis.run_id}>
                <label className="compare-check">
                  <input
                    type="checkbox"
                    checked={selectedComparisonRuns.includes(analysis.run_id)}
                    disabled={isBusy}
                    onChange={() => onToggleComparisonRun(analysis.run_id)}
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
                    onClick={() => onOpenAnalysis(analysis.run_id)}
                    disabled={isBusy}
                  >
                    {(isOpening || isPendingSave) && <span className="button-spinner small" aria-hidden="true" />}
                    <span>{isPendingSave ? "Saving..." : isOpening ? "Opening..." : "Open"}</span>
                  </button>
                  <button
                    className="danger-button history-action-button"
                    type="button"
                    onClick={() => onDeleteOne(analysis)}
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
  );
}
