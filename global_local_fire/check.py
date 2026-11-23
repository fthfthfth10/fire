"""Metadata and one synthetic CUDA BF16 forward/backward; never train or test."""
import argparse
from pathlib import Path

import torch

from .data import checked_splits
from .io_utils import PROJECT, ROOT, read_json
from .model import GlobalLocalFire, JointLoss, STRIDES
from .train import cuda_device, image_f1


def check(pretrained):
    settings = read_json(PROJECT / "config.json")
    train, val = checked_splits(ROOT / settings["dataset_dir"])
    counts = len(train), sum(row.label for row in train), sum(len(row.fire) for row in train), len(val), sum(
        row.label for row in val)
    if counts != (8707, 4164, 7625, 259, 71) or sum(bool(row.fire) for row in train) != 4163:
        raise ValueError(f"Unexpected training/validation metadata: {counts}")
    if image_f1(torch.tensor([0.5, 0.2, 0.7]), torch.tensor([True, False, False]), 0.5) != 2 / 3:
        raise AssertionError("Image F1 threshold or denominator mismatch")
    torch.set_num_threads(2)
    torch.manual_seed(settings["seed"])
    device = cuda_device(settings)
    model = GlobalLocalFire(settings, pretrained).to(device).train()
    criterion = JointLoss(model, settings)
    size = settings["image_size"]
    images = torch.rand((2, 3, size, size), device=device)
    batch = {"labels": torch.tensor([1.0, 0.0], device=device),
             "known": torch.tensor([[True, False], [True, True]], device=device),
             "bboxes": torch.tensor([[0.5, 0.4, 0.08, 0.06], [0.4, 0.5, 0.12, 0.1]], device=device),
             "cls": torch.tensor([0.0, 1.0], device=device), "batch_idx": torch.tensor([0, 1], device=device)}
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(images, with_detection=True)
    expected = tuple((size // stride, size // stride) for stride in STRIDES)
    raw = output["detection"]
    anchors = sum(height * width for height, width in expected)
    if (tuple(tuple(feat.shape[-2:]) for feat in raw["feats"]) != expected
            or raw["scores"].shape != (2, 2, anchors) or raw["boxes"].shape != (2, 64, anchors)
            or raw["scores"].dtype != torch.bfloat16 or any(output[key].shape != (2,) for key in
                                                                   ("global", "mil", "smoke", "fusion"))):
        raise AssertionError("Unexpected pyramid or evidence shapes")
    output["smoke"].retain_grad()
    loss, terms = criterion(output, batch)
    if not loss.isfinite().item() or any(not term.isfinite().item() or term.item() <= 0 for term in terms.values()):
        raise FloatingPointError("Synthetic losses must be finite and positive")
    loss.backward()
    gradients = {"fusion": model.fusion[-1].weight.grad, "smoke_via_fusion": output["smoke"].grad,
                 "global": model.global_head.weight.grad, "P2_gate": model.gates[0].weight.grad,
                 "detector": model.detect.cv3[0][-1].weight.grad, "shared": next(model.neck.parameters()).grad}
    for name, gradient in gradients.items():
        if gradient is None or not gradient.isfinite().all().item() or not gradient.abs().max().item() > 0:
            raise AssertionError(f"Invalid gradient: {name}")
    torch.cuda.synchronize(device)
    print(f"Manifest counts: {counts}; official Smoke sidecar not supplied to this check")
    print(f"Transferred layers: {model.pretrained_layers}; strides: {STRIDES}; maps: {expected}; anchors: {anchors}")
    print(f"Losses: { {key: value.item() for key, value in terms.items()} }; gradients: {list(gradients)}")
    print(f"CUDA peak: {torch.cuda.max_memory_allocated(device) / 2**30:.3f} GiB allocated, "
          f"{torch.cuda.max_memory_reserved(device) / 2**30:.3f} GiB reserved (no optimizer)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    check(parser.parse_args().pretrained)
