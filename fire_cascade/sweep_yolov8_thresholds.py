"""Compare fixed YOLOv8 confidence thresholds on the sealed 1000-image sample."""

import argparse
import csv
from pathlib import Path

from cascade import YOLO, batches, detections, sha256
from dataset import ROOT, image_paths, read_json, write_json
from run import score_boxes, verify_detector


THRESHOLDS = tuple(value / 100 for value in range(10, 51, 5))
FIELDS = ("threshold", "F1", "precision", "recall", "TP", "FP", "FN", "TN")


def sweep(run):
    config = read_json(run / "config.json")
    seal = read_json(run / "test_inference_complete.json")
    training = read_json(run / "training_result.json")
    last = verify_detector(run, config)
    sample_file = run / "test_sample.json"
    original = run / "submission.json"
    if (seal["threshold"] != 0.5 or sha256(last) != seal["last_sha256"]
            or seal["last_sha256"] != training["last_sha256"]
            or sha256(original) != seal["submission_sha256"] or sha256(sample_file) != seal["sample_sha256"]):
        raise ValueError("The training or test artifacts differ from the accepted run")
    filenames = read_json(sample_file)["files"]
    image_map = {path.name: path for path in image_paths(ROOT / "data" / "test" / "images")}
    baseline = read_json(original)
    if len(filenames) != 1000 or len(set(filenames)) != 1000 or set(filenames) != baseline.keys():
        raise ValueError("The original test sample is incomplete or ambiguous")
    paths = [image_map[name] for name in filenames]
    output = run / "threshold-sweep-010-050"
    output.mkdir(exist_ok=False)
    write_json(output / "manifest.json", {
        "thresholds": THRESHOLDS, "last_sha256": seal["last_sha256"],
        "sample_sha256": seal["sample_sha256"], "images": len(paths),
    })

    model = YOLO(str(last))
    trial_dirs = []
    for threshold in THRESHOLDS:
        predictions = {}
        for batch in batches(paths, config["predict_batch"]):
            for path, (_, boxes) in zip(batch, detections(model, batch, config, threshold)):
                predictions[path.name] = int(bool(boxes))
        if predictions.keys() != baseline.keys():
            raise ValueError(f"Incomplete prediction coverage at {threshold:.2f}")
        if threshold == 0.5 and predictions != baseline:
            raise ValueError("The 0.50 predictions do not reproduce the accepted run")
        trial = output / f"conf-{threshold:.2f}"
        trial.mkdir()
        submission = trial / "submission.json"
        write_json(submission, predictions)
        if read_json(submission) != predictions:
            raise ValueError(f"Submission read-back failed at {threshold:.2f}")
        write_json(trial / "test_inference_complete.json", {
            "images": len(predictions), "threshold": threshold, "sample_sha256": seal["sample_sha256"],
            "last_sha256": seal["last_sha256"], "submission_sha256": sha256(submission),
        })
        trial_dirs.append(trial)
        print(f"Sealed {len(predictions)} predictions at {threshold:.2f}", flush=True)

    results = []
    for trial in trial_dirs:
        score_boxes(trial, sampled=True)
        metrics = read_json(trial / "score.json")
        results.append({key: metrics[key] for key in FIELDS})
    best_f1 = max(row["F1"] for row in results)
    best_thresholds = [row["threshold"] for row in results if row["F1"] == best_f1]
    with (output / "thresholds.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    write_json(output / "summary.json", {"best_f1": best_f1, "best_thresholds": best_thresholds, "results": results})
    print(f"Best F1: {best_f1:.6f}, threshold(s): {best_thresholds}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=lambda value: Path(value).resolve())
    sweep(parser.parse_args().run)
