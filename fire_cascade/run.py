import argparse
import csv
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from dataset import PROJECT, ROOT, image_paths, prepare, read_json, write_json


DETECTOR_LAST = Path("yolo/weights/last.pt")
PROBE_LAST = Path("clip/last.pt")


def read_completed_epochs(run):
    with (run / "yolo" / "results.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return [int(row["epoch"]) for row in reader]


def check(run, config):
    from cascade import CLIP_WEIGHTS, YOLO_PRETRAIN, sha256, torch
    from train_clip import check_clip
    from train_yolo import check_yolo
    import ultralytics

    if not YOLO_PRETRAIN.is_file() or not CLIP_WEIGHTS.is_file():
        raise FileNotFoundError("Both official pretrained checkpoints are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = read_json(run / "training_data" / "manifest.json")
    check_clip(run, manifest, config)
    torch.cuda.empty_cache()
    check_yolo(run, config)
    write_json(run / "preflight.json", {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "yolo_pretrained_sha256": sha256(YOLO_PRETRAIN), "clip_sha256": sha256(CLIP_WEIGHTS),
        "config_sha256": sha256(run / "config.json"),
    })


def train(run, config):
    from cascade import CLIP_WEIGHTS, YOLO_PRETRAIN, sha256
    from train_yolo import trainer_for

    preflight = read_json(run / "preflight.json")
    for key, path in (("yolo_pretrained_sha256", YOLO_PRETRAIN), ("clip_sha256", CLIP_WEIGHTS),
                      ("config_sha256", run / "config.json")):
        if preflight[key] != sha256(path):
            raise RuntimeError(f"Preflight inputs changed: {path}")
    if (run / "yolo").exists():
        raise FileExistsError(run / "yolo")
    trainer = trainer_for(run, config, "yolo")
    if trainer.save_dir != run / "yolo":
        raise RuntimeError(f"Unexpected official training directory: {trainer.save_dir}")
    trainer.train()
    verify_detector(run, config)


def verify_detector(run, config):
    from cascade import YOLO, torch

    if read_completed_epochs(run) != list(range(1, config["epochs"] + 1)):
        raise ValueError("Expected exactly 50 completed training epochs")
    if (run / "yolo" / "weights" / "best.pt").exists():
        raise ValueError("Unexpected best checkpoint")
    last = run / DETECTOR_LAST
    detector = YOLO(str(last))
    if detector.ckpt["epoch"] != config["epochs"] - 1 or detector.names != {0: "fire"}:
        raise ValueError("Last YOLO checkpoint metadata disagrees with the 50-epoch run")
    if not all(torch.isfinite(value).all().item() for value in detector.model.state_dict().values()):
        raise ValueError("Final YOLO checkpoint contains non-finite tensors")
    return last


def verify(run, config):
    from cascade import load_head, sha256, torch

    last = verify_detector(run, config)
    head_path = run / PROBE_LAST
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    if state["config"] != config or state["detector_sha256"] != sha256(last):
        raise ValueError("Final CLIP head does not match the final detector or locked configuration")
    load_head(head_path)
    return last, head_path


def predict_test(run, config):
    from cascade import FireCascade, batches, sha256

    detector_path, head_path = verify(run, config)
    paths = image_paths(ROOT / "data" / "test" / "images")
    if len(paths) != 5313 or len({path.name for path in paths}) != len(paths):
        raise ValueError("Expected exactly 5313 uniquely named test images")
    submission = run / "submission.json"
    marker = run / "test_inference_started.json"
    if submission.exists():
        raise FileExistsError("Submission already exists")
    model = FireCascade(detector_path, head_path, config)
    with marker.open("x", encoding="utf-8") as stream:
        import json
        json.dump({
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "image_count": len(paths),
            "detector_sha256": sha256(detector_path), "probe_sha256": sha256(head_path),
            "config_sha256": sha256(run / "config.json"),
        }, stream, indent=2)
    predictions = {}
    for index, batch in enumerate(batches(paths, config["predict_batch"]), 1):
        labels = model.predict(batch)
        if len(labels) != len(batch) or any(label not in (0, 1) for label in labels):
            raise ValueError("Invalid image-level prediction")
        predictions.update((path.name, label) for path, label in zip(batch, labels))
        if index % 25 == 0:
            print(f"Test inference: {len(predictions)}/{len(paths)}", flush=True)
    write_json(submission, predictions)
    saved = read_json(submission)
    if saved != predictions or saved.keys() != {path.name for path in paths}:
        raise ValueError("Submission verification failed")
    write_json(run / "test_inference_complete.json", {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "images": len(saved),
        "predicted_fire": sum(saved.values()), "submission_sha256": sha256(submission),
    })
    print(f"Saved {len(predictions)} image predictions to {submission}", flush=True)


def score_boxes(run):
    from cascade import sha256

    submission = run / "submission.json"
    completed = read_json(run / "test_inference_complete.json")
    if completed["submission_sha256"] != sha256(submission):
        raise ValueError("Submission changed after inference")
    predictions = read_json(submission)
    directory = ROOT / "data" / "test" / "labels"
    if not directory.is_dir():
        print("No test truth available", flush=True)
        return
    labels = {p.stem: p for p in directory.glob("*.txt")}
    expected = {Path(name).stem for name in predictions}
    if len(expected) != len(predictions) or labels.keys() != expected:
        raise ValueError("Test labels cannot be independently paired with the complete submission")
    tp = fp = fn = tn = 0
    for name, prediction in predictions.items():
        truth = 0
        for line in labels[Path(name).stem].read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            category, *box = line.split()
            if category not in {"0", "1"} or len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
                raise ValueError(f"Unexpected test label format for {name}")
            truth |= int(category == "0")
        tp += truth * prediction
        fp += (1 - truth) * prediction
        fn += truth * (1 - prediction)
        tn += (1 - truth) * (1 - prediction)
    denominator = 2 * tp + fp + fn
    metrics = {"source": str(directory), "image_count": len(predictions),
               "submission_sha256": completed["submission_sha256"],
               "F1": 2 * tp / denominator if denominator else 0.0,
               "TP": tp, "FP": fp, "FN": fn, "TN": tn}
    write_json(run / "local_test_f1.json", metrics)
    print(metrics, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "check", "train", "probe", "infer", "score"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.stage == "prepare":
        run.mkdir(parents=True, exist_ok=False)
        config = read_json(PROJECT / "config.json")
        write_json(run / "config.json", config)
        prepare(run)
        return
    config = read_json(run / "config.json")
    if config["epochs"] != 50:
        raise ValueError("This run requires 50 detector epochs")
    if args.stage == "check":
        check(run, config)
    elif args.stage == "train":
        train(run, config)
    elif args.stage == "probe":
        from train_clip import fit_probe

        if (run / PROBE_LAST).exists() or (run / "test_inference_started.json").exists():
            raise FileExistsError("The CLIP checkpoint or test run already exists")
        last = verify_detector(run, config)
        manifest = read_json(run / "training_data" / "manifest.json")
        fit_probe(run, last, manifest, config)
        verify(run, config)
    elif args.stage == "infer":
        predict_test(run, config)
    else:
        score_boxes(run)


if __name__ == "__main__":
    sys.exit(main())
