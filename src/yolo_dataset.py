import argparse
import random
import shutil
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "ball_yolo"
DEFAULT_CLASS_NAMES = ["basketball"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BALL_CLASS_HINTS = {"basketball", "sports ball", "ball"}


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def create_dataset_dirs(dataset_dir: Path) -> None:
    for relative_dir in (
        "to_label/images",
        "to_label/labels",
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        (dataset_dir / relative_dir).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(dataset_dir: Path, class_names: list[str] | None = None) -> Path:
    class_names = class_names or DEFAULT_CLASS_NAMES
    yaml_path = dataset_dir / "dataset.yaml"
    names = ", ".join(f"'{name}'" for name in class_names)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                f"names: [{names}]",
                "",
            ]
        )
    )
    return yaml_path


def init_dataset(dataset_dir: Path) -> None:
    create_dataset_dirs(dataset_dir)
    yaml_path = write_dataset_yaml(dataset_dir)
    print(f"Dataset initialized: {dataset_dir}")
    print(f"Dataset YAML: {yaml_path}")


def video_frame_count(cap: cv2.VideoCapture) -> int:
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count > 0:
        return count

    current_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    measured_count = 0
    while True:
        success, _ = cap.read()
        if not success:
            break
        measured_count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
    return measured_count


def choose_frame_numbers(total_frames: int, frames_per_video: int, include_edges: bool) -> list[int]:
    if total_frames <= 0:
        return []
    if frames_per_video <= 0 or frames_per_video >= total_frames:
        return list(range(1, total_frames + 1))

    first_frame = 1 if include_edges else max(1, total_frames // 12)
    last_frame = total_frames if include_edges else min(total_frames, total_frames - total_frames // 12)
    if last_frame <= first_frame:
        first_frame, last_frame = 1, total_frames

    step = (last_frame - first_frame) / max(1, frames_per_video - 1)
    return sorted({max(1, min(total_frames, round(first_frame + index * step))) for index in range(frames_per_video)})


def extract_video_frames(video_path: Path, output_dir: Path, frames_per_video: int, include_edges: bool) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipped unreadable video: {video_path}")
        return 0

    try:
        total_frames = video_frame_count(cap)
        frame_numbers = choose_frame_numbers(total_frames, frames_per_video, include_edges)
        saved_count = 0
        for frame_number in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            success, frame = cap.read()
            if not success:
                continue

            image_name = f"{video_path.stem}_frame_{frame_number:05d}.jpg"
            image_path = output_dir / image_name
            if cv2.imwrite(str(image_path), frame):
                saved_count += 1

        return saved_count
    finally:
        cap.release()


def extract_frames(video_paths: list[Path], dataset_dir: Path, frames_per_video: int, include_edges: bool) -> None:
    create_dataset_dirs(dataset_dir)
    output_dir = dataset_dir / "to_label" / "images"
    total_saved = 0
    for video_path in video_paths:
        saved_count = extract_video_frames(video_path, output_dir, frames_per_video, include_edges)
        total_saved += saved_count
        print(f"{video_path.name}: saved {saved_count} frames")

    print(f"Saved {total_saved} frames to: {output_dir}")
    print(f"Label these images with one class: basketball")
    print(f"Put YOLO label .txt files in: {dataset_dir / 'to_label' / 'labels'}")


def image_paths_for_split(source_images_dir: Path) -> list[Path]:
    return sorted(path for path in source_images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def clear_split_dirs(dataset_dir: Path) -> None:
    for relative_dir in ("images/train", "images/val", "labels/train", "labels/val"):
        split_dir = dataset_dir / relative_dir
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in split_dir.iterdir():
            if path.is_file():
                path.unlink()


def split_labeled_dataset(dataset_dir: Path, val_ratio: float, seed: int, allow_empty_labels: bool) -> None:
    create_dataset_dirs(dataset_dir)
    source_images_dir = dataset_dir / "to_label" / "images"
    source_labels_dir = dataset_dir / "to_label" / "labels"
    images = image_paths_for_split(source_images_dir)
    if not images:
        raise RuntimeError(f"No images found in {source_images_dir}")

    labeled_pairs = []
    missing_labels = []
    for image_path in images:
        label_path = source_labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            labeled_pairs.append((image_path, label_path))
        elif allow_empty_labels:
            labeled_pairs.append((image_path, None))
        else:
            missing_labels.append(label_path.name)

    if missing_labels:
        preview = ", ".join(missing_labels[:8])
        raise RuntimeError(
            f"{len(missing_labels)} images are missing label files in {source_labels_dir}. "
            f"Examples: {preview}. Use --allow-empty-labels only for true negative frames."
        )

    random.Random(seed).shuffle(labeled_pairs)
    val_count = max(1, round(len(labeled_pairs) * val_ratio)) if len(labeled_pairs) > 1 else 0
    val_pairs = set(labeled_pairs[:val_count])

    clear_split_dirs(dataset_dir)
    for image_path, label_path in labeled_pairs:
        split = "val" if (image_path, label_path) in val_pairs else "train"
        shutil.copy2(image_path, dataset_dir / "images" / split / image_path.name)
        target_label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
        if label_path:
            shutil.copy2(label_path, target_label_path)
        else:
            target_label_path.write_text("")

    yaml_path = write_dataset_yaml(dataset_dir)
    print(f"Prepared YOLO dataset: {dataset_dir}")
    print(f"Train images: {len(labeled_pairs) - val_count}")
    print(f"Val images: {val_count}")
    print(f"Dataset YAML: {yaml_path}")


def is_ball_label(label: str) -> bool:
    normalized = str(label).strip().lower()
    return normalized in BALL_CLASS_HINTS or normalized.endswith(" ball")


def yolo_box_line(width: int, height: int, xyxy) -> str:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2
    return f"0 {center_x / width:.6f} {center_y / height:.6f} {box_width / width:.6f} {box_height / height:.6f}"


def prelabel_with_yolo(
    dataset_dir: Path,
    model_path: Path,
    confidence: float,
    image_size: int,
    overwrite: bool,
    write_empty: bool,
) -> None:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required. Install it with: pip install -r requirements-yolo.txt") from exc

    source_images_dir = dataset_dir / "to_label" / "images"
    output_labels_dir = dataset_dir / "to_label" / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    images = image_paths_for_split(source_images_dir)
    if not images:
        raise RuntimeError(f"No images found in {source_images_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(str(model_path))
    written_count = 0
    detected_count = 0
    skipped_count = 0
    for image_path in images:
        label_path = output_labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not overwrite:
            skipped_count += 1
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            skipped_count += 1
            continue

        height, width = frame.shape[:2]
        lines = []
        results = model.predict(frame, verbose=False, conf=confidence, imgsz=image_size)
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                label = model.names.get(class_id, class_id)
                if not is_ball_label(label):
                    continue
                lines.append(yolo_box_line(width, height, box.xyxy[0]))

        if lines or write_empty:
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            written_count += 1
            if lines:
                detected_count += 1

    print(f"Prelabel complete: {written_count} label files written")
    print(f"Images with draft ball boxes: {detected_count}")
    print(f"Skipped existing/unreadable images: {skipped_count}")
    print("Review and correct these labels before training.")


def parse_video_paths(values: list[str]) -> list[Path]:
    paths = []
    for value in values:
        resolved = resolve_path(value)
        if resolved.is_dir():
            paths.extend(sorted(resolved.glob("*.mp4")))
        else:
            paths.append(resolved)

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing video paths: {', '.join(str(path) for path in missing)}")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a custom YOLO basketball dataset.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Dataset root directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the YOLO dataset folders and dataset.yaml")

    extract_parser = subparsers.add_parser("extract", help="Extract frames from videos for manual labeling")
    extract_parser.add_argument("videos", nargs="+", help="Video files or directories containing .mp4 files")
    extract_parser.add_argument("--frames-per-video", type=int, default=32, help="Number of frames to sample from each video")
    extract_parser.add_argument("--include-edges", action="store_true", help="Include the first and last video frames")

    split_parser = subparsers.add_parser("split", help="Split labeled images into YOLO train/val folders")
    split_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    split_parser.add_argument("--seed", type=int, default=7, help="Deterministic split seed")
    split_parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Allow images without ball labels as negative examples",
    )

    prelabel_parser = subparsers.add_parser("prelabel-yolo", help="Create draft labels with the current YOLO model")
    prelabel_parser.add_argument("--model-path", default="model/basketball_yolo.pt", help="YOLO model used for draft labels")
    prelabel_parser.add_argument("--confidence", type=float, default=0.03, help="Detection confidence for draft labels")
    prelabel_parser.add_argument("--image-size", type=int, default=1280, help="YOLO inference image size")
    prelabel_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing draft label files")
    prelabel_parser.add_argument("--write-empty", action="store_true", help="Write empty labels for images with no ball detection")

    args = parser.parse_args()
    dataset_dir = resolve_path(args.dataset_dir)

    if args.command == "init":
        init_dataset(dataset_dir)
    elif args.command == "extract":
        extract_frames(parse_video_paths(args.videos), dataset_dir, args.frames_per_video, args.include_edges)
    elif args.command == "split":
        split_labeled_dataset(dataset_dir, args.val_ratio, args.seed, args.allow_empty_labels)
    elif args.command == "prelabel-yolo":
        prelabel_with_yolo(
            dataset_dir,
            resolve_path(args.model_path),
            args.confidence,
            args.image_size,
            args.overwrite,
            args.write_empty,
        )


if __name__ == "__main__":
    main()
