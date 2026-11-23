"""Run the official image-only entry using the validation-sealed threshold."""
import argparse
import math
from pathlib import Path

import torch

from .data import checked_splits, official_test_records
from .io_utils import ROOT, read_json, sha256, write_json
from .model import GlobalLocalFire
from .train import cuda_device, loader, probabilities, selection_key


def score_confirmed(labels_dir, submission, seal):
    if sha256(submission) != seal["submission_sha256"]:
        raise ValueError("The saved submission no longer matches its seal")
    predictions = read_json(submission)
    paths = {path.stem: path for path in labels_dir.glob("*.txt")}
    stems = {Path(name).stem for name in predictions}
    if len(stems) != len(predictions) or paths.keys() != stems:
        raise ValueError("Confirmed labels must uniquely cover every submitted filename")
    tp = fp = fn = tn = 0
    for name, prediction in predictions.items():
        truth = 0
        for line in paths[Path(name).stem].read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if not fields:
                continue
            if (len(fields) != 5 or fields[0] not in {"0", "1"}
                    or not all(math.isfinite(float(value)) and 0 <= float(value) <= 1 for value in fields[1:])):
                raise ValueError(f"Invalid YOLO truth for {name}")
            truth |= int(fields[0] == "0")
        tp += truth * prediction
        fp += (1 - truth) * prediction
        fn += truth * (1 - prediction)
        tn += (1 - truth) * (1 - prediction)
    return {**seal, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}


def infer(run, output, confirmed_labels=None):
    if output.suffix.lower() != ".json" or output.exists() or (run / "official_test_started.json").exists():
        raise ValueError("Use a new JSON output path; official inference is one pass per training run")
    best, last = run / "best.pt", run / "last.pt"
    config = read_json(run / "config.json")
    identity = read_json(run / "training_identity.json")
    selection = read_json(run / "validation_threshold.json")
    root = ROOT / config["dataset_dir"]
    threshold = selection["selected"]["threshold"]
    if (selection["split"] != "validation" or selection["manifest_sha256"] != identity["validation_manifest_sha256"]
            or config["epochs"] != selection["completed_epochs"] or sha256(best) != selection["best_sha256"]
            or threshold not in config["validation_thresholds"] or not math.isfinite(threshold)
            or sha256(root / "validation.json") != identity["validation_manifest_sha256"]
            or sha256(root / "train.json") != identity["train_manifest_sha256"]):
        raise ValueError("Validation threshold, manifest or weights do not match the completed run")
    final = torch.load(last, map_location="cpu", weights_only=True)
    if (final["format"] != 2 or final["epoch"] != config["epochs"] or final["config"] != config
            or final["identity"] != identity):
        raise ValueError("Final checkpoint does not match the completed run")
    del final
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    if (checkpoint["format"] != 2 or checkpoint["epoch"] != selection["best_epoch"]
            or checkpoint["config"] != config or checkpoint["identity"] != identity
            or checkpoint["validation_selected"] != selection["selected"]
            or checkpoint["validation_grid"] != selection["grid"]
            or [row["threshold"] for row in selection["grid"]] != config["validation_thresholds"]
            or max(selection["grid"], key=selection_key) != selection["selected"]):
        raise ValueError("Best checkpoint is not the selected model")
    device = cuda_device(config)
    model = GlobalLocalFire(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    training, validation = checked_splits(root)
    protected_hashes = {row.digest for row in training + validation}
    records = official_test_records(ROOT / config["official_test_dir"], config["official_test_count"], protected_hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (run / "official_test_started.json").open("x", encoding="utf-8") as stream:
        stream.write(f'{{"images": {len(records)}, "threshold": {threshold}}}\n')
    predictions = {}
    for batch, scores in probabilities(model, loader(records, config), device):
        predictions.update(zip(batch["names"], (scores >= threshold).int().tolist(), strict=True))
        if len(predictions) % 1000 == 0 or len(predictions) == len(records):
            print(f"Official inference: {len(predictions)}/{len(records)}", flush=True)
    if (predictions.keys() != {row.path.name for row in records}
            or any(type(value) is not int or value not in (0, 1) for value in predictions.values())):
        raise ValueError("Incomplete or invalid official submission")
    write_json(output, predictions)
    if read_json(output) != predictions:
        raise ValueError("Submission read-back mismatch")
    seal = {"images": len(predictions), "threshold": threshold, "best_sha256": selection["best_sha256"],
            "submission_sha256": sha256(output)}
    write_json(run / "official_test_complete.json", seal)
    if confirmed_labels is not None:
        score = score_confirmed(confirmed_labels, output, seal)
        write_json(run / "score.json", score)
        print(score, flush=True)
    else:
        print("Submission sealed; no confirmed test labels supplied; no score computed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmed-labels", type=Path, help="Explicitly verified YOLO labels: 0=Fire, 1=Smoke")
    args = parser.parse_args()
    infer(args.run, args.output, args.confirmed_labels)
