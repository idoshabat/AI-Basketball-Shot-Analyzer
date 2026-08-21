# Technical Overview

This project is built as a full analysis pipeline, not a single model call. The system takes a raw basketball shot video, extracts pose and ball signals, computes basketball-specific metrics, and returns an explainable coaching report.

## System Flow

```text
Frontend upload
-> FastAPI endpoint
-> OpenCV video reader
-> MediaPipe Pose keypoints
-> feature extraction
-> camera-aware shot analyzer
-> optional ball tracker
-> charts + evidence frames + annotated video
-> Supabase persistence
-> saved reports and comparisons
```

## Backend Pipeline

1. `src/api.py`
   Receives uploads, validates camera view and shooting hand, calls the analysis pipeline, and returns report data plus media URLs.

2. `src/video_reader.py`
   Reads frames with OpenCV and extracts pose landmarks with MediaPipe.

3. `src/feature_extractor.py`
   Converts landmarks into frame-by-frame joint angles, landmark coordinates, and motion signals.

4. `src/shot_analyzer.py`
   Detects phases, calculates metrics, scores the shot, builds quality warnings, and generates coaching items.

5. `src/ball_detector.py`
   Tracks the basketball using a heuristic backend, optional YOLO, or optional Roboflow hosted detector.

6. `src/analyze_video.py`
   Orchestrates the full run and saves charts, evidence frames, annotated videos, and JSON reports.

## Camera-Aware Scoring

The analyzer does not use one universal checklist for every video. It changes the scoring focus by camera view.

Side view emphasizes:

- release timing
- elbow extension
- knee bend
- hip rise and leg drive
- ankle lift
- follow-through hold
- side-view ball arc hints

Front/back view emphasizes:

- feet direction
- foot stagger
- knee-to-foot alignment
- forearm verticality
- follow-through line drift
- body lean
- sideways ball drift hints

## Explainability

Each improvement item includes:

- metric name
- measured value
- target range
- affected phase
- frame range
- coaching explanation
- practice drill
- optional evidence frame with overlay lines/dots

This is important because the user should not only receive a score. They should understand why the score happened and what action to take next.

## Reliability Layer

The system separates input quality from shooting form. A poor video can produce unstable pose landmarks, so the analyzer reports reliability checks such as:

- release detection confidence
- usable pose frame count
- post-release frames
- full-body landmark availability
- multi-person / busy scene risk
- player size in frame
- camera-specific metric availability

Low reliability can cap the score so bad input videos do not receive misleading high scores.

## Persistence

In local mode, analysis runs are saved under `storage/analyses`.

In deployed mode:

- Supabase Postgres stores report JSON, score, camera view, shooting side, and owner user id.
- Supabase Storage stores UI media such as charts, annotated videos, and evidence frames.
- The API returns signed URLs so saved reports continue working from Vercel even when Render's filesystem is temporary.

## QA Strategy

The project includes `scripts/model_qa.py`, which runs known videos through the analyzer and checks expected behavior.

Current QA set:

- `ft10`: default AI-generated front-view demo
- `ft3`: AI-generated side-view demo
- `ft6`: AI-generated back-view demo

This script is the guardrail for future tuning. If a new threshold fixes one clip but breaks another, the QA table makes that visible.

## Current Limitations

- The score is rule-based, not trained from a large labeled dataset.
- Ball tracking is beta and informational.
- Front/back videos cannot measure true shot arc because camera depth distorts the ball path.
- Side videos cannot reliably judge foot direction toward the basket.
- Very crowded clips can still confuse player tracking, although the reliability layer is designed to catch many of those cases.
