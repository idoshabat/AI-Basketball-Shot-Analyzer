# AI Basketball Shot Analyzer

MVP pipeline for analyzing a basketball shot video with OpenCV and MediaPipe Pose.

The current pipeline:

```text
video
-> pose keypoints CSV
-> joint-angle features CSV
-> rule-based shot analysis
-> phase-aware angle chart
-> annotated video
-> JSON report
```

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The MediaPipe pose model should exist at:

```text
model/pose_landmarker_lite.task
```

## Full Analysis

Run the complete MVP pipeline:

```bash
python src/analyze_video.py videos/ft2.mp4 --save-chart --save-annotated-video --save-report
```

Generated files:

```text
data/ft2_keypoints.csv
data/ft2_features.csv
output/ft2_angles.png
output/ft2_annotated.mp4
output/ft2_report.json
```

## API Server

Start the FastAPI backend:

```bash
venv/bin/uvicorn src.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Analyze an uploaded video:

```bash
curl -X POST "http://127.0.0.1:8000/analyze-shot" \
  -F "file=@videos/ft2.mp4"
```

Optional query params:

```text
save_chart=true
save_annotated_video=true
save_report=true
```

Generated output files are available under:

```text
http://127.0.0.1:8000/output/<filename>
```

## Frontend

Install frontend dependencies:

```bash
cd frontend
npm install
```

Start the frontend dev server:

```bash
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Keep the FastAPI backend running on:

```text
http://127.0.0.1:8000
```

## Individual Commands

Read and display a video with pose detection:

```bash
python src/video_reader.py videos/ft2.mp4
```

Save keypoints:

```bash
python src/video_reader.py videos/ft2.mp4 --save-keypoints --no-display
```

Extract features:

```bash
python src/feature_extractor.py data/ft2_keypoints.csv
```

Analyze shot:

```bash
python src/shot_analyzer.py data/ft2_features.csv
```

Create an angle chart:

```bash
python src/visualize_features.py data/ft2_features.csv
```

Create an annotated video:

```bash
python src/video_annotator.py videos/ft2.mp4 data/ft2_features.csv
```

## Current Metrics

The analyzer currently estimates:

- shooting side
- max elbow extension
- knee bend
- release frame
- release wrist velocity
- follow-through duration
- dip/load phase
- upward motion phase
- recovery phase
- hip rise into release
- ankle-based jump lift estimate

## Notes

This is still a rule-based MVP. It does not detect the ball or rim yet.
