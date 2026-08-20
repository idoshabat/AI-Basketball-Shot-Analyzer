import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_YAML = PROJECT_ROOT / "data" / "ball_yolo" / "dataset.yaml"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "model" / "basketball_yolo.pt"
DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "model" / "basketball_yolo_custom.pt"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def train_model(
    data_yaml: Path,
    base_model: Path,
    output_model: Path,
    epochs: int,
    image_size: int,
    batch: int,
    device: str | None,
) -> Path:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    if not base_model.exists():
        raise FileNotFoundError(f"Base YOLO model not found: {base_model}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required. Install it with: pip install -r requirements-yolo.txt") from exc

    model = YOLO(str(base_model))
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "ball_yolo"),
        name="basketball_custom",
        exist_ok=True,
        patience=20,
        plots=True,
    )

    save_dir = Path(getattr(results, "save_dir", PROJECT_ROOT / "runs" / "ball_yolo" / "basketball_custom"))
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.exists():
        raise RuntimeError(f"Training finished, but best.pt was not found at: {best_model}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output_model)
    return output_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO model for basketball detection.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET_YAML), help="YOLO dataset YAML path")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL), help="Starting YOLO weights")
    parser.add_argument("--output-model", default=str(DEFAULT_OUTPUT_MODEL), help="Where to copy best.pt after training")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--image-size", type=int, default=1280, help="Training/inference image size")
    parser.add_argument("--batch", type=int, default=4, help="Training batch size")
    parser.add_argument("--device", default=None, help="Device passed to Ultralytics, for example cpu, mps, or 0")
    args = parser.parse_args()

    output_model = train_model(
        resolve_path(args.data),
        resolve_path(args.base_model),
        resolve_path(args.output_model),
        args.epochs,
        args.image_size,
        args.batch,
        args.device,
    )
    print(f"Custom YOLO model saved to: {output_model}")
    print("Use it with:")
    print(f"BALL_DETECTION_MODEL_PATH={output_model.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
