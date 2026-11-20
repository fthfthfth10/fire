"""Use the reviewed trainer with user-labelled negative additions and one sampled test."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

from run_dinov2_locked import BACKBONE_SHA256, COMMIT, now, require_finite, score_sealed, verify_checkout


def run(args):
    verify_checkout(args.checkout, args.git_blobs)
    sys.path.insert(0, str(args.checkout))
    import torch
    from PIL import Image
    from common import config, images, json_digest, read_json, sha256_file, write_json
    from data import Bags, TrainingImage, training_records
    from infer import load_head, predict_image, publish_predictions, submission_names, unique_object
    from model import FrozenDino, source_sha256, weight_sha256
    from tiles import IGNORE, tile_boxes, tile_labels
    import train as locked_trainer

    settings = config()
    pool_count = settings["submission_image_count"]
    settings.update(fire_probability_threshold=args.threshold, submission_image_count=1000)
    weights_sha = weight_sha256(args.weights, settings["backbone"])
    if weights_sha != BACKBONE_SHA256 or source_sha256(args.dinov2_source) != settings["backbone"]["source_sha256"]:
        raise ValueError("Unexpected official DINOv2 weight or source")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 is required")

    output = args.run
    work = output / "train"
    submission = output / "submission.json"
    start_marker = output / "test_inference_started.json"
    if args.resume:
        if not (work / "run.json").is_file():
            raise FileNotFoundError("Missing training identity for resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
    if start_marker.exists() or submission.exists():
        raise FileExistsError("Test inference for this run already started")

    original, original_fingerprint = training_records(args.data / "train", settings)
    positives = sum(sample.fire for sample in original)
    boxes = sum(box["category_id"] == 0 for box in read_json(args.data / "train" / "train_coco.json")["annotations"])
    if (len(original), positives, boxes) != (1100, 864, 1445):
        raise ValueError("Unexpected original training annotations")
    negative_counts = {"ai": args.ai_count}
    if args.ai2_count is not None:
        negative_counts["ai2"] = args.ai2_count
    ai_paths = []
    for folder, count in negative_counts.items():
        paths = images(args.data / "train" / folder)
        submission_names(paths, count)
        ai_paths.extend(paths)
    ai_names = {path.name.casefold() for path in ai_paths}
    if len(ai_names) != len(ai_paths) or ai_names & {sample.path.name.casefold() for sample in original}:
        raise ValueError("AI and original training filenames collide")
    added = []
    for path in ai_paths:
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        tiles = tile_boxes(width, height, settings["tiles"])
        targets = tile_labels(tiles, [], settings["tiles"])
        if len(tiles) < 2 or targets != [IGNORE] + [0] * (len(tiles) - 1):
            raise ValueError(f"Invalid no-fire tile labels: {path.name}")
        added.append(TrainingImage(path, width, height, False, tuple(tiles), tuple(targets), sha256_file(path)))
    samples = original + added
    added_manifests = {f"{folder}_negative": [] for folder in negative_counts}
    for sample in added:
        added_manifests[f"{sample.path.parent.name}_negative"].append([sample.path.name, sample.sha256])
    identity = {
        "config": settings, "backbone_sha256": weights_sha,
        "training_fingerprint": json_digest({"original": original_fingerprint, **added_manifests}),
        "images": len(samples), "fire_images": positives, "no_fire_images": len(samples) - positives,
    }
    write_json(output / "preflight.json", {
        "commit": COMMIT, "identity": identity, "fire_boxes": boxes, "ai_negative_images": negative_counts["ai"],
        "ai2_negative_images": negative_counts.get("ai2", 0),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    })
    for name, entries in added_manifests.items():
        write_json(output / f"{name}_manifest.json", entries)
    print(f"{now()} preflight passed: {len(samples)} bags, {positives} fire, {len(samples)-positives} no-fire", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with patch.object(locked_trainer, "config", return_value=settings), patch.object(
        locked_trainer, "training_records", return_value=(samples, identity["training_fingerprint"])
    ):
        locked_trainer.train(argparse.Namespace(train_data=args.data / "train", dinov2_source=args.dinov2_source,
                                               weights=args.weights, run_dir=work, resume=args.resume))
    torch.cuda.synchronize()
    training = {
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    last = work / "last.pt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=True)
    history = checkpoint["history"]
    if (checkpoint["epoch"] != 50 or [item["epoch"] for item in history] != list(range(1, 51))
            or checkpoint["identity"] != identity or read_json(work / "run.json") != identity):
        raise ValueError("Final checkpoint is incomplete or training identity differs")
    if not checkpoint["optimizer"]["state"] or not checkpoint["optimizer"]["param_groups"]:
        raise ValueError("Missing optimizer state")
    require_finite(checkpoint["model"], torch)
    require_finite(checkpoint["optimizer"], torch)
    with (work / "training.csv").open(encoding="utf-8", newline="") as stream:
        logged = list(csv.DictReader(stream))
    if len(logged) != len(history) or any(
        any(not math.isfinite(float(value)) or float(row[key]) != value for key, value in item.items())
        for item, row in zip(history, logged)
    ):
        raise ValueError("Loss history is inconsistent or nonfinite")
    cache = work / "features-v2"
    if {path.name for path in cache.glob("*.pt")} != {f"{index:06d}.pt" for index in range(len(samples))}:
        raise ValueError("Feature cache does not cover all training images")
    bags = Bags(cache, samples, settings["backbone"]["feature_dim"])
    for index in range(len(bags)):
        require_finite(bags[index][0], torch)
    model = load_head(last, settings, weights_sha)
    training.update(epochs=50, final=history[-1], epoch_seconds_sum=sum(row["seconds"] for row in history),
                    last_sha256=sha256_file(last), cache_images=len(bags), strict_head_load=True)
    write_json(output / "training_result.json", training)
    print(f"{now()} strict checkpoint acceptance passed: {training}", flush=True)

    pool = sorted(images(args.data / "test" / "images"), key=lambda path: (path.name.casefold(), path.name))
    submission_names(pool, pool_count)
    sample_seed = settings["seed"]
    paths = random.Random(sample_seed).sample(pool, settings["submission_image_count"])
    names = submission_names(paths, settings["submission_image_count"])
    sample_file = output / "test_sample.json"
    write_json(sample_file, {"seed": sample_seed, "pool_count": len(pool), "files": [path.name for path in paths]})
    encoder = FrozenDino.from_pretrained(args.dinov2_source, args.weights, settings)
    test_started = time.perf_counter()
    with start_marker.open("x", encoding="utf-8") as stream:
        json.dump({"time_utc": now(), "images": len(paths), "threshold": settings["fire_probability_threshold"],
                   "last_sha256": training["last_sha256"], "sample_sha256": sha256_file(sample_file)}, stream)
    predictions = {}
    for index, path in enumerate(paths, 1):
        predictions[path.name] = predict_image(path, encoder, model, settings)
        if index % 100 == 0:
            print(f"{now()} inference {index}/{len(paths)}", flush=True)
    publish_predictions(submission, predictions, names)
    with submission.open(encoding="utf-8") as stream:
        sealed_predictions = json.load(stream, object_pairs_hook=unique_object)
    if sealed_predictions != predictions:
        raise ValueError("Sealed file differs from validated predictions")
    seal = {
        "time_utc": now(), "images": len(sealed_predictions), "threshold": settings["fire_probability_threshold"],
        "sample_sha256": sha256_file(sample_file), "predicted_fire": sum(sealed_predictions.values()),
        "submission_sha256": sha256_file(submission), "last_sha256": training["last_sha256"],
        "inference_wall_seconds": time.perf_counter() - test_started,
    }
    write_json(output / "submission_seal.json", seal)
    print(f"{now()} submission sealed; scoring only the selected labels", flush=True)
    labels = args.data / "test" / "labels"

    class SampledLabels:
        def glob(self, pattern):
            stems = {Path(name).stem for name in names}
            return (path for path in labels.glob(pattern) if path.stem in stems)

    metrics = score_sealed(submission, seal, SampledLabels(), sealed_predictions, sha256_file)
    metrics.update(threshold=seal["threshold"], sampling_seed=sample_seed, sampling_pool_count=len(pool),
                   same_name_train_test_overlap=len({name.casefold() for name in names} &
                                                    {sample.path.name.casefold() for sample in samples}))
    if sha256_file(submission) != seal["submission_sha256"]:
        raise ValueError("Submission changed after scoring")
    write_json(output / "score.json", metrics)
    print(f"{now()} scoring complete: {metrics}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("checkout", "git-blobs", "dinov2-source", "weights", "data", "run"):
        parser.add_argument(f"--{option}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--ai-count", type=int, default=157)
    parser.add_argument("--ai2-count", type=int)
    parser.add_argument("--threshold", type=float, choices=(0.35, 0.50), default=0.35)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
