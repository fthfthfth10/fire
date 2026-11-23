"""Run the official image-only entry using the validation-sealed threshold."""
import argparse
import math
from pathlib import Path

import torch

from .data import checked_splits, official_test_records
from .io_utils import ROOT, read_json, sha256, write_json
from .model import GlobalLocalFire
from .train import cuda_device, loader, probabilities


def infer(run, output):
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
    if (final["format"] != 1 or final["epoch"] != config["epochs"] or final["config"] != config
            or final["identity"] != identity):
        raise ValueError("Final checkpoint does not match the completed run")
    del final
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    if (checkpoint["format"] != 1 or checkpoint["epoch"] != selection["best_epoch"]
            or checkpoint["config"] != config or checkpoint["identity"] != identity):
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
    if (predictions.keys() != {row.path.name for row in records}
            or any(type(value) is not int or value not in (0, 1) for value in predictions.values())):
        raise ValueError("Incomplete or invalid official submission")
    write_json(output, predictions)
    if read_json(output) != predictions:
        raise ValueError("Submission read-back mismatch")
    write_json(run / "official_test_complete.json", {"images": len(predictions), "threshold": threshold,
               "best_sha256": selection["best_sha256"], "submission_sha256": sha256(output)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    infer(args.run, args.output)
