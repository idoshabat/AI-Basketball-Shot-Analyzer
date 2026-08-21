import React from "react";

export function WelcomePage({
  error,
  isSigningIn,
  isSupabaseConfigured,
  onContinueAsGuest,
  onSignInWithGoogle,
}) {
  return (
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
        <button type="button" onClick={onSignInWithGoogle} disabled={!isSupabaseConfigured || isSigningIn}>
          {isSigningIn && <span className="button-spinner small" aria-hidden="true" />}
          <span>{isSigningIn ? "Opening Google..." : "Sign in with Google"}</span>
        </button>
        <button className="sample-button" type="button" onClick={onContinueAsGuest}>
          Continue as guest
        </button>
        {!isSupabaseConfigured && (
          <p className="processing-note">Google sign-in needs Supabase env vars. Guest mode is available now.</p>
        )}
        {error && <p className="error-text">{error}</p>}
      </div>
    </section>
  );
}
