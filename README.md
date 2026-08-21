# AI Basketball Shot Analyzer

An end-to-end basketball shooting analysis app that turns a regular phone video into pose data, camera-aware mechanics feedback, evidence frames, annotated video, saved analysis history, and shot-to-shot comparisons.

The goal of this project is not just to detect pose landmarks. It behaves like a small coaching product: upload a shot, choose the filming angle, get a score, understand what changed, and know exactly what to improve next.

![Sample angle chart](docs/screenshots/sample-angle-chart.png)

## Why It Stands Out

Most pose demos stop at drawing a skeleton. This project goes further:

- It converts raw landmarks into basketball-specific phases, metrics, scoring, and coaching priorities.
- It changes the analysis based on camera view, because side, front, and back videos reveal different mechanics.
- It produces evidence frames for each improvement, so feedback is explainable instead of mysterious.
- It stores user-specific history with Supabase Auth, Postgres, and Storage.
- It includes ball tracking as a beta signal using heuristic, YOLO, or Roboflow-backed detection.
- It has a polished React experience with saved reports, comparisons, loading states, delete flows, and demo samples.

## Product Preview

The frontend includes built-in demo reports, so reviewers can explore the product without uploading a video.

| View | What it demonstrates |
| --- | --- |
| AI front view `ft10` | Feet direction, knee alignment, forearm verticality, follow-through line, body lean |
| AI side view `ft3` | Release timing, elbow extension, knee bend, leg drive, jump lift, follow-through hold |
| AI back view `ft6` | Alignment and shot-line mechanics from behind |

Useful sample assets:

- [Front-view evidence frame](frontend/public/samples/front-ft10/coaching_frame_01.jpg)
- [Side-view angle chart](frontend/public/samples/side-ft3/angles.png)
- [Back-view demo report](frontend/public/samples/back-ft6/analysis.json)
- [Front-view annotated video](frontend/public/samples/front-ft10/annotated.webm)

## Highlights

- Upload a basketball shot video from the browser.
- Detect player pose with MediaPipe.
- Extract body keypoints and joint-angle features.
- Estimate shooting side, release frame, knee bend, jump lift, arm extension, and follow-through timing.
- Track the basketball as a beta signal for visibility, release-frame comparison, and arc hints.
- Generate rule-based coaching feedback and a shot score.
- Show priority improvement cards with target metrics and drills.
- Generate charts and optional annotated video.
- Save analysis runs for later review with Supabase persistence when configured.
- Compare two saved shots manually.
- Compare the current shot to the best saved shot automatically.
- Choose the camera view before analysis so scoring uses metrics that are actually visible from that angle.
- Sign in with Google through Supabase Auth so saved analyses are scoped to the current user.
- Load a built-in sample result, including annotated video, from the frontend for quick demos without uploading a file.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Frontend | React, Vite, custom CSS |
| Backend | FastAPI, Python |
| Computer vision | OpenCV, MediaPipe Pose |
| Data processing | NumPy, pandas |
| Charts | Matplotlib |
| Ball detection | Heuristic tracking, optional YOLO, optional Roboflow hosted detector |
| Auth and persistence | Supabase Auth, Supabase Postgres, Supabase Storage |
| Deployment | Vercel frontend, Render backend |

## Architecture

```text
React / Vite frontend
  -> video upload + camera-view selection
  -> FastAPI backend
  -> OpenCV frame reader
  -> MediaPipe Pose keypoints
  -> feature extraction
  -> camera-aware shot analyzer
  -> ball tracking beta signal
  -> charts, evidence frames, annotated video
  -> Supabase report + media persistence
  -> saved history and shot comparison UI
```

## Demo Flow

For a quick project walkthrough:

1. Start the backend and frontend.
2. Open the app at `http://127.0.0.1:5173`.
3. Choose a demo sample and click `Load Demo` to show the dashboard and annotated video instantly.
4. Upload a real shot video.
5. Review the score, feedback, improvement priorities, charts, and annotated video.
6. Click `Compare to Best` after multiple analyses to compare the current shot with the best saved one.

More presentation notes are in [docs/demo.md](docs/demo.md).
For a deeper implementation walkthrough, read [docs/technical-overview.md](docs/technical-overview.md).

## What The Analyzer Measures

The project separates metrics by camera view so the feedback is grounded in what the camera can actually see.

### Side View

- Release frame and release confidence
- Elbow extension at release
- Knee bend and dip/load phase
- Hip rise and leg drive
- Ankle lift / jump lift
- Follow-through hold duration
- Ball release timing and side-view arc hint when ball detection is stable

### Front / Back View

- Feet direction relative to the shot line
- Foot stagger during the dip load
- Knee-to-foot alignment during the load
- Forearm verticality at release
- Follow-through line drift
- Shoulder/torso lean
- Sideways ball drift when ball detection is stable

## Output Structure

During local analysis, the backend creates a working run folder:

```text
storage/analyses/<run_id>/
  input/original.mp4
  data/keypoints.csv
  data/features.csv
  output/angles.png
  output/follow_through_debug.png
  output/annotated.webm
  report.json
```

In deployed mode, Supabase becomes the persistent source of truth:

- Supabase Postgres stores the analysis report, score, metadata, metrics, feedback, and owner user id.
- Supabase Storage stores the UI assets by default: charts, annotated video, and evidence frame images. Original videos, CSVs, and debug files are optional to keep Render memory usage lower.
- The API returns signed Storage URLs for saved media so Vercel and Render are no longer dependent on Render's temporary filesystem.

Frontend demo data lives under `frontend/public/samples/`. The public demo selector includes only AI-generated samples: `front-ft10`, `side-ft3`, and `back-ft6`. Each demo includes evidence images, charts, and an annotated video.

## Engineering Notes

- The score is rule-based by design. It is explainable, tunable, and easier to debug than a black-box score for the current dataset size.
- Ball tracking is marked beta and does not dominate the grade yet. It is used for visibility, release-frame comparison, and camera-specific flight hints.
- The frontend warns users when hosted analysis may take a few minutes because the Render backend can run on very limited CPU.
- The QA script checks known-good and known-bad videos so future threshold changes can be validated instead of tuned from one clip at a time.

## Known Limitations

- The app is not a make/miss predictor yet.
- The score is a coaching heuristic, not a trained biomechanics model.
- Camera angle matters. A side-view clip cannot reliably judge foot direction, and a front-view clip cannot reliably judge true shot arc.
- Ball tracking quality depends on visibility, motion blur, camera distance, and detector configuration.
- Very crowded or far-away videos may track the wrong player, so the app includes reliability warnings and score caps for poor inputs.

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Optional YOLO ball tracking:

```bash
pip install -r requirements-yolo.txt
```

Then place a basketball/sports-ball YOLO model at:

```text
model/basketball_yolo.pt
```

or set `BALL_DETECTION_MODEL_PATH` to the model file. Without this optional model, the app still runs and uses the heuristic fallback ball tracker.

For reliable ball tracking, fine-tune a basketball-specific model with [docs/yolo-ball-training.md](docs/yolo-ball-training.md). The repo includes scripts to extract frames, split YOLO labels, train, and copy the best model into `model/basketball_yolo_custom.pt`.

Optional Roboflow hosted detector:

```bash
BALL_DETECTION_BACKEND=roboflow
ROBOFLOW_MODEL_ID=basketball-detector-n4omu/1
ROBOFLOW_API_KEY=your-roboflow-api-key
ROBOFLOW_API_URL=https://detect.roboflow.com
ROBOFLOW_FRAME_STRIDE=2
ROBOFLOW_MAX_WORKERS=8
ROBOFLOW_RELEASE_WINDOW_BEFORE=18
ROBOFLOW_RELEASE_WINDOW_AFTER=42
```

Use this when your custom Roboflow model is trained and you want the backend to call the hosted detector while still generating the same ball tracking CSV and annotated video.

The MediaPipe pose model should exist at:

```text
model/pose_landmarker_lite.task
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

## Run Locally

Start the FastAPI backend from the project root:

```bash
venv/bin/uvicorn src.api:app --reload
```

Start the Vite frontend:

```bash
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The frontend expects the backend at:

```text
http://127.0.0.1:8000
```

## Model QA

Before tuning scoring, camera-view rules, shooting-hand detection, or ball tracking, run the known-video QA set:

```bash
PYTHONPATH=src venv/bin/python scripts/model_qa.py
```

The default QA run uses the heuristic ball detector so it stays fast and offline. The public demo set uses only AI-generated clips:

- `ft10` as the default front-view demo
- `ft3` as the side-view demo
- `ft6` as the back-view demo

The script prints score, reliability, camera view, shooting hand, quality warnings, top improvement cards, ball visibility, and pass/fail status.

Use this before and after model changes. If a change improves one video but breaks another, the QA table makes that visible immediately.

To include the hosted Roboflow detector in QA:

```bash
PYTHONPATH=src venv/bin/python scripts/model_qa.py --ball-backend roboflow
```

To run one case:

```bash
PYTHONPATH=src venv/bin/python scripts/model_qa.py --case videos/ft10.mp4
```

To save a JSON report:

```bash
PYTHONPATH=src venv/bin/python scripts/model_qa.py --json-output output/model_qa.json
```

## Auth Setup

Authentication is optional for local demos, but required for real per-user history.

The frontend uses Supabase Auth when these Vite env vars exist:

```bash
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
```

The backend verifies Supabase JWTs when this env var exists:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_JWT_SECRET=your-supabase-jwt-secret
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
SUPABASE_ANALYSES_TABLE=analyses
SUPABASE_STORAGE_BUCKET=shot-analyses
SUPABASE_UPLOAD_ANNOTATED_VIDEO=true
SUPABASE_UPLOAD_ORIGINAL_VIDEO=false
SUPABASE_UPLOAD_DEBUG_FILES=false
```

`SUPABASE_URL` and `SUPABASE_ANON_KEY` are required for newer Supabase projects that use asymmetric JWT signing keys such as `ES256` or `RS256`.

`SUPABASE_SERVICE_ROLE_KEY` is required only on the backend. Keep it secret and set it in Render, never in Vercel. The backend uses it to write reports to Postgres and upload analysis files to Supabase Storage.

For low-memory Render instances, keep `SUPABASE_UPLOAD_ORIGINAL_VIDEO=false` and `SUPABASE_UPLOAD_DEBUG_FILES=false`. Large files are uploaded with streaming, but skipping non-UI files still keeps the service much more stable.

Create the required table and storage bucket by running [supabase/schema.sql](supabase/schema.sql) in the Supabase SQL editor.

If Supabase is not configured, the app runs in `Guest mode` with local filesystem persistence. If it is configured, users can sign in with Google and their saved analyses, comparisons, and best-shot history are scoped to their account.

For Google sign-in, enable the Google provider in Supabase Auth and add your local/deployed frontend URLs to the allowed redirect URLs.

## API

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Analyze an uploaded video:

```bash
curl -X POST "http://127.0.0.1:8000/analyze-shot" \
  -F "file=@videos/ft10.mp4" \
  -F "camera_view=front"
```

Compare two saved analyses:

```bash
curl "http://127.0.0.1:8000/analyses/compare?run_a=<run_id>&run_b=<run_id>"
```

Compare one saved analysis to the best saved shot:

```bash
curl "http://127.0.0.1:8000/analyses/<run_id>/compare-best"
```

## Project Status

This is a polished MVP with a real product loop:

```text
Record shot -> Analyze -> Explain mistakes -> Save report -> Compare progress
```

The next meaningful improvements are dataset-driven:

- run the Model QA suite regularly
- add more labeled ball frames from varied gyms, balls, and camera angles
- add more known-good / known-bad pose clips
- eventually train scoring against human-labeled shot-quality targets

## CLI Commands

Run the complete analysis pipeline:

```bash
python src/analyze_video.py videos/ft10.mp4 --camera-view front --save-chart --save-annotated-video --save-report
```

Read and display a video with pose detection:

```bash
python src/video_reader.py videos/ft10.mp4
```

Save keypoints:

```bash
python src/video_reader.py videos/ft10.mp4 --save-keypoints --no-display
```

Extract features:

```bash
python src/feature_extractor.py data/ft10_keypoints.csv
```

Analyze shot features:

```bash
python src/shot_analyzer.py data/ft10_features.csv
```

Track the ball with the default auto backend:

```bash
python src/ball_detector.py videos/ft10.mp4 data/ft10_features.csv --output data/ft10_ball_tracking.csv --shooting-side right --release-frame 122
```

Track the ball with YOLO:

```bash
python src/ball_detector.py videos/ft10.mp4 data/ft10_features.csv --output data/ft10_ball_tracking.csv --shooting-side right --release-frame 122 --backend yolo --model-path model/basketball_yolo.pt --yolo-confidence 0.05
```

Create an angle chart:

```bash
python src/visualize_features.py data/ft10_features.csv
```

Create an annotated video:

```bash
python src/video_annotator.py videos/ft10.mp4 data/ft10_features.csv
```

## Metrics

The analyzer currently estimates:

- shooting side
- release frame
- release confidence
- release elbow angle
- knee bend
- hip rise into release
- ankle-based jump lift
- follow-through duration
- follow-through end frame
- setup, load, upward motion, release, follow-through, and recovery phases
- arm motion variation

## Camera View Awareness

The app supports `side`, `front`, and `back` camera views.

Side view emphasizes:

- release frame
- elbow extension
- knee bend depth
- hip rise
- ankle-based jump lift
- follow-through timing

Front/back view emphasizes:

- wrist, elbow, and shoulder stacking at release
- knee tracking over the feet
- left-right stance symmetry
- shoulder level
- body lean

This matters because the app should not strongly grade a metric that is not visible from the selected camera angle.

## Deployment Notes

The frontend is designed for Vercel. The backend can run on Render using `render.yaml`.

Render free or very small CPU instances can be slow for video analysis. A full request may take a few minutes because the backend runs pose detection frame by frame and may also generate annotated video. The frontend includes a visible loading state and a note explaining this delay.

If local and deployed scores differ for the same video, first verify that both environments are running the same backend version:

```bash
curl http://127.0.0.1:8000/health
curl https://ai-basketball-shot-analyzer.onrender.com/health
```

Both responses should show the same `analysis_version`. If they do not, redeploy the Render backend. The frontend on Vercel only controls the browser UI; the scoring result comes from whichever FastAPI backend it is configured to call.

`/health` also returns `auth_configured`. On Render, it should be `true` once Supabase backend env vars are set.

For a faster hosted demo:

- Disable annotated video by default for public demos, or
- Use a larger Render instance, or
- Keep the built-in sample selector available through `Load Demo` so visitors can see the AI-generated front, side, and back reports immediately.

## Project Status

This is a polished rule-based MVP with beta ball tracking. Ball tracking is YOLO-first when a configured model is available, with a heuristic fallback when it is not. It does not yet detect the rim or use ball tracking in the score. The next major ML upgrade would be a custom fine-tuned basketball/rim model, but the current version already demonstrates the full product loop: video in, biomechanics and ball-tracking hints out, coaching feedback, saved history, and comparison.
