import argparse
from pathlib import Path

import torch

from common import config, images, write_json
from model import FireMIL, FrozenDino, fire_decisions, weight_sha256
from tiles import open_rgb


def load_head(checkpoint_path, settings, backbone_sha256):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    identity = checkpoint["identity"]
    if checkpoint["epoch"] != settings["epochs"]:
        raise ValueError("Inference requires last.pt from epoch 50")
    if identity["config"] != settings or identity["backbone_sha256"] != backbone_sha256:
        raise ValueError("The locked configuration or backbone differs from last.pt")
    model = FireMIL(settings)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().to(settings["device"])


@torch.inference_mode()
def predict_image(path, encoder, model, settings):
    features = encoder.encode(open_rgb(path)).unsqueeze(0).to(settings["device"])
    mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
    logits, _ = model(features, mask)
    if not logits.isfinite().all().item():
        raise FloatingPointError(f"Nonfinite image logit: {path.name}")
    return int(fire_decisions(logits, settings).item())


def main():
    parser = argparse.ArgumentParser(description="Predict visible fire directly from images.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--images", type=Path)
    parser.add_argument("--dinov3-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--last", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = [args.image] if args.image is not None else images(args.images)
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError("No input images")
    output = args.output
    if args.images is not None and output is None:
        output = Path("submission.json")
    if output is not None and output.exists():
        raise FileExistsError(output)

    settings = config()
    digest = weight_sha256(args.weights, settings["backbone"])
    model = load_head(args.last, settings, digest)
    encoder = FrozenDino.from_pretrained(args.dinov3_source, args.weights, settings)
    predictions = {path.name: predict_image(path, encoder, model, settings) for path in paths}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, predictions)
    if args.image is not None:
        print(predictions[args.image.name])


if __name__ == "__main__":
    main()
