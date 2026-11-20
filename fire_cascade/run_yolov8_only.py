"""Run YOLOv8n alone on the previously accepted MIL training and test sets."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

from PIL import Image, ImageOps

from cascade import YOLO, YOLO_PRETRAIN, batches, detections, sha256, torch
from dataset import PROJECT, ROOT, image_paths, prepare, read_json, write_json
from run import score_boxes, verify_detector
from train_yolo import TrainOnlyDetector, check_yolo, trainer_for


PRETRAIN_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
PREVIOUS_TRAINING = "6fa9a8c3eeadb4971d1cb7305b72cafe6df71a93a814bba2cbd9023a95f03c4b"
THRESHOLD = 0.50
SAMPLE_COUNT = 1000


class CombinedDetector(TrainOnlyDetector):
    expected_images = 1554


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def prepare_combined(run):
    source = ROOT / "data" / "train"
    manifest = prepare(run)
    training = run / "training_data"
    original_labels = [sha256(source / name) for name in ("train_coco.json", "train_image.json")]
    original_images = [[row["file"], sha256(source / "images" / row["file"])] for row in manifest]
    fingerprint = {"original": digest_json({"labels": original_labels, "images": original_images})}
    filenames = {Path(row["file"]).stem.casefold() for row in manifest}
    for folder, count in (("ai", 254), ("ai2", 200)):
        paths = image_paths(source / folder)
        if len(paths) != count:
            raise ValueError(f"Expected {count} {folder} images, found {len(paths)}")
        entries = []
        for index, src in enumerate(paths, start=1):
            staged = f"{folder}_negative_{index:04d}{src.suffix.lower()}"
            stem = Path(staged).stem
            if stem.casefold() in filenames:
                raise ValueError(f"YOLO label filename collision: {staged}")
            filenames.add(stem.casefold())
            dst = training / "images" / staged
            with Image.open(src) as image:
                normalized = ImageOps.exif_transpose(image)
                normalized.load()
                if min(normalized.size) < 10:
                    raise ValueError(f"Image smaller than YOLO accepts: {src}")
                if image.getexif().get(274, 1) != 1:
                    normalized.convert("RGB").save(dst, quality=100, subsampling=0)
                else:
                    shutil.copyfile(src, dst)
            (training / "labels" / f"{stem}.txt").write_text("", encoding="ascii")
            entries.append([src.name, sha256(src)])
            manifest.append({"file": staged, "source": f"{folder}/{src.name}", "fire": 0, "boxes": []})
        fingerprint[f"{folder}_negative"] = entries
    digest = digest_json(fingerprint)
    audit = read_json(training / "audit.json")
    positives = sum(row["fire"] for row in manifest)
    if digest != PREVIOUS_TRAINING or (len(manifest), positives, audit["box_classes"].get("0")) != (1554, 864, 1445):
        raise ValueError("Training images or labels differ from the last accepted dataset")
    audit.update(images=len(manifest), image_labels={"0": len(manifest) - positives, "1": positives},
                 ai_negative_images=254, ai2_negative_images=200, training_fingerprint=digest)
    write_json(training / "manifest.json", manifest)
    write_json(training / "audit.json", audit)
    print(f"Verified the same training set: {len(manifest)} images, {positives} with fire", flush=True)


def train_detector(run, config):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("The reviewed YOLOv8n trainer requires CUDA with native BF16")
    if sha256(YOLO_PRETRAIN) != PRETRAIN_SHA256:
        raise ValueError("YOLOv8n pretrained checkpoint differs from the previously verified official asset")
    check_yolo(run, config, CombinedDetector)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer = trainer_for(run, config, "yolo", CombinedDetector)
    trainer.train()
    last = verify_detector(run, config)
    with (run / "yolo" / "results.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    losses = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
    if any(not math.isfinite(float(row[key])) for row in history for key in losses):
        raise FloatingPointError("Non-finite training loss in results.csv")
    write_json(run / "training_result.json", {
        "epochs": len(history), "wall_seconds": time.perf_counter() - start,
        "final_losses": {key: float(history[-1][key]) for key in losses},
        "last_sha256": sha256(last), "pretrained_sha256": PRETRAIN_SHA256,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    })
    print("Verified final epoch and checkpoint; beginning the one sampled test", flush=True)
    return last


def test_detector(run, config, last):
    pool = sorted(image_paths(ROOT / "data" / "test" / "images"), key=lambda path: (path.name.casefold(), path.name))
    if len(pool) != 5313 or len({path.name.casefold() for path in pool}) != len(pool):
        raise ValueError("Expected exactly 5313 unique test filenames")
    paths = random.Random(config["seed"]).sample(pool, SAMPLE_COUNT)
    names = {path.name for path in paths}
    sample_file = run / "test_sample.json"
    write_json(sample_file, {"seed": config["seed"], "pool_count": len(pool), "files": [path.name for path in paths]})
    detector = YOLO(str(last))
    metadata = {"images": len(paths), "threshold": THRESHOLD, "sample_sha256": sha256(sample_file),
                "last_sha256": sha256(last)}
    with (run / "test_inference_started.json").open("x", encoding="utf-8") as stream:
        json.dump({**metadata, "time_utc": utc_now()}, stream)
    predictions = {}
    for index, batch in enumerate(batches(paths, config["predict_batch"]), start=1):
        for path, (_, boxes) in zip(batch, detections(detector, batch, config, THRESHOLD)):
            predictions[path.name] = int(bool(boxes))
        if index % 10 == 0 or len(predictions) == SAMPLE_COUNT:
            print(f"Test inference: {len(predictions)}/{SAMPLE_COUNT}", flush=True)
    if predictions.keys() != names or any(type(value) is not int or value not in (0, 1) for value in predictions.values()):
        raise ValueError("Incomplete or invalid image-level predictions")
    submission = run / "submission.json"
    write_json(submission, predictions)
    if read_json(submission) != predictions:
        raise ValueError("Submission read-back failed")
    write_json(run / "test_inference_complete.json", {**metadata, "time_utc": utc_now(),
                                                     "submission_sha256": sha256(submission)})
    score_boxes(run, sampled=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=lambda value: Path(value).resolve())
    run = parser.parse_args().run
    run.mkdir(parents=True, exist_ok=False)
    config = read_json(PROJECT / "config.json")
    if (config["epochs"], config["seed"]) != (50, 42):
        raise ValueError("Unexpected YOLO training protocol")
    write_json(run / "config.json", {**config, "evaluation_confidence": THRESHOLD, "evaluation_count": SAMPLE_COUNT})
    prepare_combined(run)
    last = train_detector(run, config)
    test_detector(run, config, last)


if __name__ == "__main__":
    main()
