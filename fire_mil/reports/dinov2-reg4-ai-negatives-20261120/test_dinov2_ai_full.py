"""Evaluate the accepted AI-negative run on the complete held-out image set."""
import argparse
import json
from pathlib import Path
import sys
import time

from run_dinov2_locked import BACKBONE_SHA256, COMMIT, now, score_sealed, verify_checkout


LAST_SHA256 = "b1288da3827d88677302e3b7816d23baea4ca94d06a30a1e46e2c7b505e940c0"


def run(args):
    verify_checkout(args.checkout, args.git_blobs)
    sys.path.insert(0, str(args.checkout))
    from common import config, images, read_json, sha256_file, write_json
    from infer import load_head, predict_image, publish_predictions, submission_names, unique_object
    from model import FrozenDino, weight_sha256

    settings = config()
    full_count = settings["submission_image_count"]
    settings.update(fire_probability_threshold=0.35, submission_image_count=1000)
    last = args.training / "train" / "last.pt"
    trained = read_json(args.training / "training_result.json")
    identity = read_json(args.training / "train" / "run.json")
    if (sha256_file(last) != LAST_SHA256 or trained["last_sha256"] != LAST_SHA256
            or trained["epochs"] != 50 or not trained["strict_head_load"]
            or (identity["images"], identity["fire_images"], identity["no_fire_images"]) != (1257, 864, 393)
            or identity["config"] != settings):
        raise ValueError("Checkpoint or settings do not match the accepted AI-negative run")
    digest = weight_sha256(args.weights, settings["backbone"])
    if digest != BACKBONE_SHA256:
        raise ValueError("Expected official DINOv2 weights")
    model = load_head(last, settings, digest)
    encoder = FrozenDino.from_pretrained(args.dinov2_source, args.weights, settings)
    inference_settings = {**settings, "fire_probability_threshold": args.threshold}
    paths = images(args.test_data / "images")
    names = submission_names(paths, full_count)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    submission = output / "submission.json"
    manifest = {"commit": COMMIT, "images": len(paths), "threshold": args.threshold,
                "last_sha256": LAST_SHA256, "backbone_sha256": digest}
    write_json(output / "test_inference_started.json", {**manifest, "time_utc": now()})
    print(f"{now()} strict loading passed; starting one inference pass on {len(paths)} images at {args.threshold:.2f}", flush=True)
    started = time.perf_counter()
    predictions = {}
    for index, path in enumerate(paths, 1):
        predictions[path.name] = predict_image(path, encoder, model, inference_settings)
        if index % 250 == 0 or index == len(paths):
            print(f"{now()} inference {index}/{len(paths)}", flush=True)
    publish_predictions(submission, predictions, names)
    with submission.open(encoding="utf-8") as stream:
        sealed = json.load(stream, object_pairs_hook=unique_object)
    if predictions != sealed:
        raise ValueError("Sealed submission differs from predictions")
    seal = {**manifest, "time_utc": now(), "predicted_fire": sum(sealed.values()),
            "submission_sha256": sha256_file(submission), "inference_wall_seconds": time.perf_counter() - started}
    write_json(output / "submission_seal.json", seal)
    print(f"{now()} submission sealed; reading full test labels", flush=True)
    metrics = score_sealed(submission, seal, args.test_data / "labels", sealed, sha256_file)
    metrics.update(threshold=seal["threshold"], last_sha256=LAST_SHA256)
    if sha256_file(submission) != seal["submission_sha256"]:
        raise ValueError("Submission changed during scoring")
    write_json(output / "score.json", metrics)
    print(f"{now()} scoring complete: {metrics}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("checkout", "git-blobs", "dinov2-source", "weights", "training", "test-data", "output"):
        parser.add_argument(f"--{option}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--threshold", type=float, choices=(0.35, 0.50), default=0.35)
    run(parser.parse_args())
