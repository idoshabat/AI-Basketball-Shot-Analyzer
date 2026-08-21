# Demo Guide

Use this short flow when presenting the project.

## 60-Second Walkthrough

1. Open the frontend.
2. Continue as guest or sign in with Google.
3. Click `Load Demo` to show the coaching dashboard instantly.
4. Point out the score, camera view, reliability, improvement priorities, evidence frames, charts, and annotated video.
5. Open a priority card's evidence frame and explain that each recommendation is tied to a frame and metric.
6. Go to `Recent analyses` and show saved reports, compare, open, and delete flows.
7. Upload a real shot video and explain that the hosted Render backend can take a few minutes on a low-CPU instance.
8. After the result appears, click `Compare to Best`.

## Short Demo Script

"This project analyzes a basketball shot video using OpenCV and MediaPipe Pose. It extracts body keypoints, detects shooting phases, computes mechanics like release timing, elbow extension, knee bend, leg drive, follow-through, and camera-specific alignment, then turns them into a score and coaching priorities. Each priority has an evidence frame, so the user can see exactly where the issue happened.

The frontend behaves like a real coaching product: users can sign in, save reports, open previous analyses, compare two shots, compare against their best saved shot, and load demo samples without uploading anything."

## What To Emphasize

- This is not only skeleton drawing; it is an end-to-end product pipeline.
- The analyzer is camera-aware. Side, front, and back videos are graded differently.
- The feedback is explainable through metrics and evidence frames.
- Input quality is judged separately from form quality, so reliability warnings do not get mixed into mechanics feedback.
- Supabase keeps each user's history scoped to their account.
- The public demo selector uses only AI-generated samples: `front-ft10`, `side-ft3`, and `back-ft6`.

## Best Demo Path

Use this order for the smoothest presentation:

1. Load `AI Front View - FT10`.
2. Show the score and reliability panel.
3. Open `What do I need to improve?`.
4. Click `View frame` on a front-view alignment item.
5. Open the annotated video.
6. Switch to `Recent analyses`.
7. Open a saved report and show that the app jumps back to the report.
8. Mention that uploaded videos go through the same pipeline, just slower on hosted Render.

## Reviewer Checklist

If someone is evaluating the project quickly, point them to:

- `README.md` for product overview and architecture
- `docs/demo.md` for the live demo script
- `scripts/model_qa.py` for regression checks
- `src/analyze_video.py` for the full backend pipeline
- `src/shot_analyzer.py` for scoring, camera-aware metrics, and coaching items
- `src/ball_detector.py` for heuristic / YOLO / Roboflow ball tracking
- `frontend/src/main.jsx` for the React product flow
- `supabase/schema.sql` for persistence setup

## Deployment note

The Vercel frontend is fast, but the Render backend may run on a very small CPU instance. Full pose detection plus annotated video generation can take a few minutes, especially after a cold start.

When presenting the deployed app, use `Load Demo` first so the audience sees the product immediately. Then run a live upload if there is enough time.
