"""Train one Global-Local Fire model after the dataset prerequisites are met."""
import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FireImages, checked_splits, collate
from .io_utils import PROJECT, ROOT, read_json, sha256, write_json
from .model import GlobalLocalFire, JointLoss, ULTRALYTICS_VERSION


def cuda_device(settings):
    if settings["precision"] != "bf16" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
            including_emulation=False):
        raise RuntimeError("This configuration requires native CUDA BF16; no automatic FP16 fallback")
    return torch.device("cuda:0")


def loader(records, settings, shuffle=False):
    return DataLoader(FireImages(records, settings["image_size"]), batch_size=settings["batch_size"],
                      shuffle=shuffle, num_workers=settings["workers"], pin_memory=True, collate_fn=collate,
                      generator=torch.Generator().manual_seed(settings["seed"]),
                      persistent_workers=settings["workers"] > 0)


def images_on(batch, device):
    return batch["images"].to(device, dtype=torch.float32, non_blocking=True).div_(255.0)


@torch.inference_mode()
def probabilities(model, batches, device):
    model.eval()
    for batch in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images_on(batch, device))["fusion"]
        if not logits.isfinite().all().item():
            raise FloatingPointError("Nonfinite fusion logits")
        yield batch, logits.float().sigmoid().cpu()


def collect_validation(model, batches, device):
    scores, targets = [], []
    for batch, values in probabilities(model, batches, device):
        scores.append(values)
        targets.append(batch["labels"].bool())
    return torch.cat(scores), torch.cat(targets)


def image_f1(scores, targets, threshold):
    predicted = scores >= threshold
    tp = (predicted & targets).sum().item()
    denominator = predicted.sum().item() + targets.sum().item()
    return 2 * tp / denominator if denominator else 0.0


def save_checkpoint(path, model, optimizer, epoch, settings, identity, f1):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"format": 1, "epoch": epoch, "config": settings, "identity": identity, "val_f1_at_fixed": f1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict()}, temporary)
    temporary.replace(path)


def train(args):
    settings = read_json(PROJECT / "config.json")
    if (settings["seed"], settings["epochs"]) != (42, 100):
        raise ValueError("This entry point requires seed 42 and 100 epochs")
    root = ROOT / settings["dataset_dir"]
    training, validation = checked_splits(root, args.smoke_manifest)
    documented_smoke = sum(len(row.smoke) for row in training)
    if not documented_smoke:
        raise ValueError("No verified Smoke boxes. Supply --smoke-manifest for this dataset before training")
    for row in training + validation:
        if sha256(row.path) != row.digest:
            raise ValueError(f"Dataset image bytes disagree with the manifest: {row.path.name}")
    identity = {"ultralytics": ULTRALYTICS_VERSION, "train_manifest_sha256": sha256(root / "train.json"),
                "validation_manifest_sha256": sha256(root / "validation.json"),
                "smoke_manifest_sha256": sha256(args.smoke_manifest), "train_images": len(training),
                "validation_images": len(validation), "smoke_boxes": documented_smoke,
                "smoke_reviewed_images": sum(row.smoke_reviewed for row in training)}
    if len({row.label for row in validation}) != 2:
        raise ValueError("Validation must contain both image-level classes")
    device = cuda_device(settings)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = GlobalLocalFire(settings, args.pretrained).to(device)
    identity["transferred_layers"] = model.pretrained_layers
    criterion = JointLoss(model, settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], foreach=False)
    batches = loader(training, settings, shuffle=True)
    validate = loader(validation, settings)
    args.run.mkdir(parents=True, exist_ok=False)
    write_json(args.run / "config.json", settings)
    write_json(args.run / "training_identity.json", identity)
    names = ("detection", "global", "mil", "fusion")
    best_f1 = -1.0
    last = args.run / "last.pt"
    best = args.run / "best.pt"
    with (args.run / "training.csv").open("x", encoding="utf-8", newline="") as stream:
        log = csv.writer(stream)
        log.writerow(("epoch", *names, "validation_f1_at_fixed", "peak_allocated_bytes"))
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            averages = dict.fromkeys(names, 0.0)
            accumulation = settings["accumulate"]
            for index, batch in enumerate(batches):
                batch_size = len(batch["labels"])
                for key in ("labels", "known", "bboxes", "cls", "batch_idx"):
                    batch[key] = batch[key].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(images_on(batch, device), with_detection=True)
                loss, parts = criterion(output, batch)
                if not loss.isfinite().item():
                    raise FloatingPointError(f"Nonfinite joint loss at epoch {epoch}, batch {index + 1}")
                window_start = index - index % accumulation
                window_end = min(window_start + accumulation, len(batches))
                window_images = min(window_end * settings["batch_size"], len(training)) - (
                    window_start * settings["batch_size"])
                (loss * (batch_size / window_images)).backward()
                if index + 1 == window_end:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for key in names:
                    averages[key] += parts[key].detach().item() * batch_size / len(training)
                del output, loss, parts
            values, targets = collect_validation(model, validate, device)
            metric = image_f1(values, targets, settings["checkpoint_threshold"])
            save_checkpoint(last, model, optimizer, epoch, settings, identity, metric)
            if metric > best_f1:
                best_f1 = metric
                save_checkpoint(best, model, optimizer, epoch, settings, identity, metric)
            log.writerow((epoch, *(averages[name] for name in names), metric, torch.cuda.max_memory_allocated(device)))
            stream.flush()
            print(f"Epoch {epoch:03d}: {averages}; validation F1={metric:.6f}", flush=True)
    chosen = torch.load(best, map_location="cpu", weights_only=True)
    model.load_state_dict(chosen["model"], strict=True)
    values, targets = collect_validation(model, validate, device)
    results = [{"threshold": threshold, "f1": image_f1(values, targets, threshold)}
               for threshold in settings["validation_thresholds"]]
    selected = max(results, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5), -row["threshold"]))
    write_json(args.run / "validation_threshold.json", {
        "split": "validation", "manifest_sha256": identity["validation_manifest_sha256"],
        "best_epoch": chosen["epoch"], "completed_epochs": settings["epochs"], "best_sha256": sha256(best),
        "selected": selected, "grid": results,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("pretrained", "smoke-manifest", "run"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    train(parser.parse_args())
