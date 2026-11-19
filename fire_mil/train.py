import argparse
import csv
import math
import random
import time
from pathlib import Path

import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from common import config, read_json, save_pt, write_json
from data import Bags, collate_bags, fill_cache, training_records
from model import FireMIL, FrozenDino, objective, weight_sha256


def train_epoch(model, loader, optimizer, device, pos_weight, instance_weight):
    model.train()
    totals = torch.zeros(3, device=device)
    for features, mask, bag_targets, tile_targets in loader:
        features, mask, bag_targets, tile_targets = (
            tensor.to(device) for tensor in (features, mask, bag_targets, tile_targets)
        )
        optimizer.zero_grad(set_to_none=True)
        bag_logits, tile_logits = model(features, mask)
        loss, bag_loss, tile_loss = objective(bag_logits, tile_logits, bag_targets, tile_targets,
                                              mask, pos_weight, instance_weight)
        if not loss.isfinite().item():
            raise FloatingPointError("Nonfinite MIL training loss")
        loss.backward()
        optimizer.step()
        totals += torch.stack([loss.detach(), bag_loss.detach(), tile_loss.detach()]) * len(bag_targets)
    return (totals / len(loader.dataset)).tolist()


def write_log(path, history):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "bag_loss", "tile_loss", "seconds"])
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def initialize_run(run, identity, resume):
    manifest = run / "run.json"
    if resume:
        if not manifest.is_file() or read_json(manifest) != identity:
            raise ValueError("Resume requires the same input images, labels, weights and configuration")
    else:
        if any(path.name != ".run.lock" for path in run.iterdir()):
            raise FileExistsError(f"Use an empty run directory or --resume: {run}")
        write_json(manifest, identity)


def train(args):
    settings = config()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    weight_digest = weight_sha256(args.weights, settings["backbone"])
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    samples, fingerprint = training_records(args.train_data, settings)
    positives = sum(sample.fire for sample in samples)
    negatives = len(samples) - positives
    if not positives or not negatives:
        raise ValueError("Training requires both visible-fire and no-fire images")
    identity = {
        "config": settings, "backbone_sha256": weight_digest, "training_fingerprint": fingerprint,
        "images": len(samples), "fire_images": positives, "no_fire_images": negatives,
    }
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run / ".run.lock"), timeout=0):
        initialize_run(run, identity, args.resume)
        last = run / "last.pt"
        history = []
        device = torch.device(settings["device"])
        options = settings["mil"]
        model = FireMIL(settings).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"],
                                      weight_decay=options["weight_decay"])
        if last.is_file():
            checkpoint = torch.load(last, map_location="cpu", weights_only=True)
            if checkpoint["identity"] != identity:
                raise ValueError("last.pt run identity mismatch")
            history = checkpoint["history"]
            if not (1 <= checkpoint["epoch"] <= settings["epochs"]
                    and [row["epoch"] for row in history] == list(range(1, checkpoint["epoch"] + 1))):
                raise ValueError("Invalid epoch history in last.pt")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            write_log(run / "training.csv", history)
        if len(history) == settings["epochs"]:
            print("50 epochs are already complete.", flush=True)
            return

        cache = run / "features"
        fill_cache(cache, samples, lambda: FrozenDino.from_pretrained(args.dinov3_source, args.weights, settings),
                   settings)
        torch.cuda.empty_cache()
        dataset = Bags(cache, samples, settings["backbone"]["feature_dim"])
        pos_weight = torch.tensor(negatives / positives, device=device)
        for epoch in range(len(history) + 1, settings["epochs"] + 1):
            started = time.perf_counter()
            # Per-epoch seeding reproduces the shuffle across epoch-boundary resumes.
            shuffle = torch.Generator().manual_seed(settings["seed"] + epoch)
            loader = DataLoader(dataset, batch_size=options["batch_size"], shuffle=True,
                                generator=shuffle, num_workers=0, collate_fn=collate_bags)
            loss, bag_loss, tile_loss = train_epoch(model, loader, optimizer, device, pos_weight,
                                                   options["instance_loss_weight"])
            if not all(map(math.isfinite, (loss, bag_loss, tile_loss))):
                raise FloatingPointError(f"Nonfinite losses at epoch {epoch}")
            record = {"epoch": epoch, "loss": loss, "bag_loss": bag_loss, "tile_loss": tile_loss,
                      "seconds": time.perf_counter() - started}
            history.append(record)
            save_pt(last, {"identity": identity, "epoch": epoch, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "history": history})
            write_log(run / "training.csv", history)
            print(record, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train MIL heads on all 1100 training images.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--dinov3-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
