# Custom Basketball YOLO Training

The pretrained YOLO11n model is generic. It sometimes sees a basketball as `sports ball`, but it is not reliable for small, blurry, dark, or partially hidden basketballs. The reliable upgrade is to fine-tune a basketball-specific model on frames from this project.

## 1. Extract Frames

```bash
venv/bin/python src/yolo_dataset.py init
venv/bin/python src/yolo_dataset.py extract videos/ft10.mp4 videos/ft3.mp4 videos/ft6.mp4 --frames-per-video 40
```

Frames are written to:

```text
data/ball_yolo/to_label/images/
```

## 2. Label The Ball

Optionally create draft labels from the current generic YOLO model:

```bash
venv/bin/python src/yolo_dataset.py prelabel-yolo --confidence 0.03 --image-size 1280 --write-empty
```

Then use any YOLO annotation tool, such as CVAT, Roboflow, Label Studio, or labelImg, to correct the labels.

Rules:

- One class only: `basketball`
- Draw a tight box around the ball.
- Label every visible ball, even if it is dark, blurry, or partly covered.
- If the ball is not visible in a frame, leave that frame with an empty `.txt` label file.

Put label files here:

```text
data/ball_yolo/to_label/labels/
```

Each image must have a matching YOLO `.txt` file with the same stem.

## 3. Split The Dataset

```bash
venv/bin/python src/yolo_dataset.py split --val-ratio 0.2 --allow-empty-labels
```

This creates:

```text
data/ball_yolo/images/train/
data/ball_yolo/images/val/
data/ball_yolo/labels/train/
data/ball_yolo/labels/val/
data/ball_yolo/dataset.yaml
```

## 4. Train

```bash
venv/bin/python src/train_ball_yolo.py --epochs 80 --image-size 1280 --batch 4 --device mps
```

On a CPU-only machine, use:

```bash
venv/bin/python src/train_ball_yolo.py --epochs 80 --image-size 960 --batch 2 --device cpu
```

The script copies the best model to:

```text
model/basketball_yolo_custom.pt
```

## 5. Use The Custom Model

If you trained on Roboflow and want to use the hosted endpoint, update `.env`:

```env
BALL_DETECTION_BACKEND=roboflow
ROBOFLOW_MODEL_ID=basketball-detector-n4omu/1
ROBOFLOW_API_KEY=your-roboflow-api-key
ROBOFLOW_API_URL=https://detect.roboflow.com
ROBOFLOW_FRAME_STRIDE=2
ROBOFLOW_MAX_WORKERS=8
ROBOFLOW_RELEASE_WINDOW_BEFORE=18
ROBOFLOW_RELEASE_WINDOW_AFTER=42
```

Restart the backend and run a new analysis.

If you exported or trained local YOLO weights, update `.env`:

```env
BALL_DETECTION_BACKEND=yolo
BALL_DETECTION_MODEL_PATH=model/basketball_yolo_custom.pt
BALL_DETECTION_CONFIDENCE=0.05
BALL_DETECTION_IMAGE_SIZE=1280
```

Restart the backend and run a new analysis.

## Practical Target

Start with 250-400 labeled frames. Include:

- Side, front, and back views
- Good clips and bad/crowded clips
- Orange, gray, brown, and dark basketballs
- Frames before release, at release, and after release
- Frames where the ball is near the body, because that is where generic YOLO fails most often
