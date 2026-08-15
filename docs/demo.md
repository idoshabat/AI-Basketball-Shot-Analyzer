# Demo Guide

Use this short flow when presenting the project.

## 60-second walkthrough

1. Open the frontend.
2. Click `Load Sample Result` to show the coaching dashboard instantly.
3. Point out the score, release confidence, follow-through timing, and improvement priorities.
4. Upload a real shot video and explain that the hosted Render backend can take a few minutes on the free/low-CPU instance.
5. After the result appears, click `Compare to Best` to compare the current shot against the best saved analysis.

## Short demo script

"This project analyzes a basketball shot video using OpenCV and MediaPipe Pose. It extracts body keypoints, computes shooting metrics like release frame, elbow extension, knee bend, jump lift, and follow-through timing, then turns those into a score and coaching priorities. The frontend also keeps recent analyses so a player can compare a new shot to their best saved shot."

## Deployment note

The Vercel frontend is fast, but the Render backend may run on a very small CPU instance. Full pose detection plus annotated video generation can take a few minutes, especially after a cold start.
