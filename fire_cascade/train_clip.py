import random
import warnings
from collections import Counter

import numpy as np
from PIL import Image

from cascade import (
    CLIP_SHA256, DEVICE, ClipEncoder, YOLO, batches, detections, extract_crop, open_rgb, sha256, torch,
)
from dataset import write_json
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def background_box(image, rng):
    width, height = image.size
    w = rng.randint(min(64, width), max(min(64, width), width // 2))
    h = rng.randint(min(64, height), max(min(64, height), height // 2))
    x = rng.randint(0, width - w)
    y = rng.randint(0, height - h)
    return x, y, x + w, y + h


def check_clip(run, manifest, config):
    encoder = ClipEncoder()
    images = run / "training_data" / "images"
    positive = next(record for record in manifest if record["fire"])
    negative = next(record for record in manifest if not record["fire"])
    a = open_rgb(images / positive["file"])
    b = open_rgb(images / negative["file"])
    crops = [
        extract_crop(a, positive["boxes"][0]["xyxy"], config["crop_margin"]),
        extract_crop(b, background_box(b, random.Random(config["seed"])), config["crop_margin"]),
    ]
    head = nn.Linear(512, 1).to(DEVICE)
    torch.cuda.reset_peak_memory_stats()
    inputs = encoder.encode(crops)
    logits = head(inputs).flatten()
    loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([1.0, 0.0], device=DEVICE))
    loss.backward()
    if not torch.isfinite(head.weight.grad).all().item() or not loss.isfinite().item():
        raise RuntimeError("CLIP probe loss or gradients failed")
    if any(p.requires_grad or p.grad is not None for p in encoder.visual.parameters()):
        raise RuntimeError("CLIP visual encoder is not frozen")
    print(f"CLIP one batch: loss={loss.item():.4f}, features={list(inputs.shape)}, "
          f"peak allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


def fit_probe(run, detector_path, manifest, config):
    encoder = ClipEncoder()
    images = run / "training_data" / "images"
    rng = random.Random(config["seed"])
    detector = YOLO(str(detector_path))
    if detector.names != {0: "fire"}:
        raise ValueError("Expected class 0 fire in the final YOLO checkpoint")
    margin = config["crop_margin"]
    crops, features, samples = [], [], []
    counts = Counter()

    def add(image, box, filename, label, origin, confidence=None):
        crops.append(extract_crop(image, box, margin))
        samples.append({"file": filename, "xyxy": box, "fire": label, "source": origin,
                        "yolo_confidence": confidence})
        counts[origin] += 1
        if len(crops) == config["clip_batch"]:
            features.append(encoder.encode(crops).cpu().numpy())
            crops.clear()

    negative_paths = []
    for record in tqdm(manifest, desc="CLIP training GT/background", mininterval=4):
        path = images / record["file"]
        image = open_rgb(path)
        for box in record["boxes"]:
            category = box["category"]
            if category == 0 or not record["fire"]:
                add(image, box["xyxy"], path.name, int(category == 0), "fire_gt" if category == 0 else "smoke_gt")
        if not record["fire"]:
            negative_paths.append(path)
            add(image, background_box(image, rng), path.name, 0, "random_background")

    # D5 mines detector false positives exclusively on training images without a fire box.
    for paths in tqdm(list(batches(negative_paths, config["predict_batch"])), desc="Mining training FPs"):
        for path, (pixels, boxes) in zip(paths, detections(detector, paths, config, config["mining_confidence"])):
            if not boxes:
                continue
            image = Image.fromarray(pixels[:, :, ::-1])
            for box in boxes:
                if int(box[5]) == 0:
                    add(image, box[:4], path.name, 0, "mined_false_positive", box[4])
    if crops:
        features.append(encoder.encode(crops).cpu().numpy())
    x = np.concatenate(features, axis=0)
    y = np.array([sample["fire"] for sample in samples])
    if len(x) != len(y) or not np.isfinite(x).all() or set(y) != {0, 1}:
        raise ValueError("Invalid CLIP probe training set")

    # D5's class-balanced L2 logistic regression. Map its final coefficients
    # to a portable PyTorch linear layer for the torch-only inference path.
    probe = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                               solver="lbfgs", random_state=config["seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        probe.fit(x, y)
    head = nn.Linear(512, 1)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(probe.coef_))
        head.bias.copy_(torch.from_numpy(probe.intercept_))
        probabilities = head(torch.from_numpy(x)).sigmoid().numpy().ravel()
    np.testing.assert_allclose(probabilities, probe.predict_proba(x)[:, 1], rtol=2e-5, atol=2e-6)
    output = run / "clip"
    output.mkdir(exist_ok=False)
    write_json(output / "samples.json", samples)
    report = {"training_samples": len(y), "source_counts": dict(counts),
              "lbfgs_iterations": int(probe.n_iter_[0]),
              "mined_at_or_above_detection_gate": sum(
                  s["yolo_confidence"] is not None and s["yolo_confidence"] >= config["candidate_confidence"]
                  for s in samples),
              "mined_at_or_above_auto_keep_gate": sum(
                  s["yolo_confidence"] is not None and s["yolo_confidence"] >= config["auto_keep_confidence"]
                  for s in samples)}
    write_json(output / "training.json", report)
    temporary = output / "last.pt.tmp"
    torch.save({"head": head.state_dict(), "clip_sha256": CLIP_SHA256,
                "detector_sha256": sha256(detector_path), "config": config}, temporary)
    temporary.replace(output / "last.pt")
    print(f"CLIP fit complete: {report}", flush=True)
