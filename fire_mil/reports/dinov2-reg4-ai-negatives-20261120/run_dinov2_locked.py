"""Run the reviewed DINOv2 implementation and seal one submission before scoring."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time


COMMIT = "d74e54d01884d6d5a44cd42aa7cda2d3f0499626"
BACKBONE_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_checkout(directory, manifest):
    expected = json.loads(manifest.read_text(encoding="utf-8-sig"))
    actual = {}
    for path in directory.iterdir():
        if path.is_file():
            payload = path.read_bytes()
            prefix = f"blob {len(payload)}\0".encode()
            actual[path.name] = hashlib.sha1(prefix + payload).hexdigest()
    if expected["commit"] != COMMIT or expected["blobs"] != actual:
        raise ValueError("Local implementation differs from the reviewed commit")


def score_sealed(submission, seal, labels, predictions, sha256_file):
    if sha256_file(submission) != seal["submission_sha256"]:
        raise ValueError("Submission no longer matches its seal")
    if len({Path(name).stem for name in predictions}) != len(predictions):
        raise ValueError("Ambiguous image stems for test label pairing")
    label_files = {path.stem: path for path in labels.glob("*.txt")}
    if label_files.keys() != {Path(name).stem for name in predictions}:
        raise ValueError("Test label stems do not exactly match the sealed submission")
    tp = fp = fn = tn = 0
    for name, pred in predictions.items():
        truth = 0
        for line in label_files[Path(name).stem].read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5 or fields[0] not in {"0", "1"} or not all(
                math.isfinite(float(value)) for value in fields[1:]
            ):
                raise ValueError(f"Invalid test label for {name}")
            truth |= int(fields[0] == "0")
        tp += truth * pred
        fp += (1 - truth) * pred
        fn += truth * (1 - pred)
        tn += (1 - truth) * (1 - pred)
    return {
        "images": len(predictions), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "submission_sha256": seal["submission_sha256"],
    }


def require_finite(value, torch):
    if isinstance(value, torch.Tensor):
        if not value.isfinite().all().item():
            raise FloatingPointError("Nonfinite tensor in checkpoint")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item, torch)


def run(args):
    verify_checkout(args.checkout, args.git_blobs)
    sys.path.insert(0, str(args.checkout))
    import torch
    from common import config, images, read_json, sha256_file, write_json
    from data import Bags, training_records
    from infer import load_head, predict_image, publish_predictions, submission_names, unique_object
    from model import FrozenDino, source_sha256, weight_sha256
    from train import train

    output = args.run.resolve()
    work = output / "train"
    output.mkdir(parents=True, exist_ok=True)
    start_marker = output / "test_inference_started.json"
    submission = output / "submission.json"
    if start_marker.exists() or submission.exists():
        raise FileExistsError("The test inference for this run was already started")
    settings = config()
    weights_sha = weight_sha256(args.weights, settings["backbone"])
    if weights_sha != BACKBONE_SHA256:
        raise ValueError("Unexpected official checkpoint")
    if source_sha256(args.dinov2_source) != settings["backbone"]["source_sha256"]:
        raise ValueError("Unexpected official DINOv2 source")
    mil = settings["mil"]
    if (settings["seed"], settings["epochs"], mil["batch_size"], mil["learning_rate"],
        mil["weight_decay"], mil["instance_loss_weight"], settings["feature_cache_version"],
        settings["fire_probability_threshold"], settings["submission_image_count"]) != (
        42, 50, 8, 0.0003, 0.01, 0.30, 2, 0.5, 5313
    ):
        raise ValueError("Unexpected training or inference settings")
    samples, fingerprint = training_records(args.data / "train", settings)
    positives = sum(sample.fire for sample in samples)
    annotation = read_json(args.data / "train" / "train_coco.json")
    boxes = sum(box["category_id"] == 0 for box in annotation["annotations"])
    if (len(samples), positives, len(samples) - positives, boxes) != (1100, 864, 236, 1445):
        raise ValueError("Unexpected training annotations")
    identity = {
        "config": settings, "backbone_sha256": weights_sha, "training_fingerprint": fingerprint,
        "images": 1100, "fire_images": 864, "no_fire_images": 236,
    }
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 is required")
    write_json(output / "preflight.json", {
        "commit": COMMIT, "identity": identity, "fire_boxes": boxes, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    })
    print(f"{now()} preflight passed, 1100 images, 864 fire, 236 non-fire, {boxes} fire boxes", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    train(argparse.Namespace(train_data=args.data / "train", dinov2_source=args.dinov2_source,
                             weights=args.weights, run_dir=work, resume=args.resume))
    torch.cuda.synchronize()
    training_stats = {
        "wall_seconds": time.perf_counter() - training_started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    last = work / "last.pt"
    ckpt = torch.load(last, map_location="cpu", weights_only=True)
    history = ckpt["history"]
    if ckpt["epoch"] != 50 or [item["epoch"] for item in history] != list(range(1, 51)):
        raise ValueError("Expected all 50 epochs and the final checkpoint")
    if ckpt["identity"] != identity or read_json(work / "run.json") != identity:
        raise ValueError("Training identity mismatch")
    if not ckpt["optimizer"]["state"] or not ckpt["optimizer"]["param_groups"]:
        raise ValueError("Checkpoint is missing optimizer state")
    require_finite(ckpt["model"], torch)
    require_finite(ckpt["optimizer"], torch)
    with (work / "training.csv").open(encoding="utf-8", newline="") as stream:
        logged = list(csv.DictReader(stream))
    if len(logged) != len(history) or any(
        any(not math.isfinite(float(value)) or float(row[key]) != value for key, value in item.items())
        for item, row in zip(history, logged)
    ):
        raise ValueError("Training history and loss log disagree or contain nonfinite values")
    cache = work / "features-v2"
    expected_cache = {f"{index:06d}.pt" for index in range(len(samples))}
    if {path.name for path in cache.glob("*.pt")} != expected_cache:
        raise ValueError("Feature cache does not cover all training images")
    bags = Bags(cache, samples, settings["backbone"]["feature_dim"])
    for index in range(len(bags)):
        require_finite(bags[index][0], torch)
    model = load_head(last, settings, weights_sha)
    training_stats.update({
        "epochs": 50, "final": history[-1], "epoch_seconds_sum": sum(row["seconds"] for row in history),
        "last_sha256": sha256_file(last), "cache_images": len(bags), "strict_head_load": True,
    })
    write_json(output / "training_result.json", training_stats)
    print(f"{now()} training and strict checkpoint acceptance passed: {training_stats}", flush=True)

    paths = images(args.data / "test" / "images")
    names = submission_names(paths, settings["submission_image_count"])
    encoder = FrozenDino.from_pretrained(args.dinov2_source, args.weights, settings)
    test_started = time.perf_counter()
    with start_marker.open("x", encoding="utf-8") as stream:
        json.dump({"time_utc": now(), "commit": COMMIT, "images": len(paths),
                   "last_sha256": training_stats["last_sha256"]}, stream)
    predictions = {}
    for index, path in enumerate(paths, 1):
        predictions[path.name] = predict_image(path, encoder, model, settings)
        if index % 250 == 0 or index == len(paths):
            print(f"{now()} inference {index}/{len(paths)}", flush=True)
    publish_predictions(submission, predictions, names)
    with submission.open(encoding="utf-8") as stream:
        sealed_predictions = json.load(stream, object_pairs_hook=unique_object)
    if sealed_predictions != predictions:
        raise ValueError("Published submission differs from the validated predictions")
    seal = {
        "time_utc": now(), "images": len(sealed_predictions), "predicted_fire": sum(sealed_predictions.values()),
        "submission_sha256": sha256_file(submission), "last_sha256": training_stats["last_sha256"],
        "inference_wall_seconds": time.perf_counter() - test_started,
    }
    write_json(output / "submission_seal.json", seal)
    print(f"{now()} submission sealed; reading evaluation labels", flush=True)
    metrics = score_sealed(submission, seal, args.data / "test" / "labels", sealed_predictions, sha256_file)
    metrics["same_name_train_test_overlap"] = len(names & {sample.path.name for sample in samples})
    if sha256_file(submission) != seal["submission_sha256"]:
        raise ValueError("Submission changed after scoring")
    write_json(output / "score.json", metrics)
    print(f"{now()} scoring complete: {metrics}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("checkout", "git-blobs", "dinov2-source", "weights", "data", "run"):
        parser.add_argument(f"--{option}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
