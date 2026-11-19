import argparse
import json
import tempfile
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


def submission_names(paths, expected_count):
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} test images, found {len(paths)}")
    names = {path.name for path in paths}
    if len({name.casefold() for name in names}) != expected_count:
        raise ValueError("Test image filenames must be unique (case-insensitive)")
    return names


def validate_predictions(predictions, expected_names):
    if not isinstance(predictions, dict) or predictions.keys() != expected_names:
        raise ValueError("Predictions must cover exactly the input image filenames")
    if any(type(value) is not int or value not in (0, 1) for value in predictions.values()):
        raise ValueError("Every image prediction must be an integer 0 or 1")


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate keys in prediction JSON")
    return result


def publish_predictions(output, predictions, expected_names):
    validate_predictions(predictions, expected_names)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "predictions.json"
        write_json(staged, predictions)
        with staged.open(encoding="utf-8") as stream:
            decoded = json.load(stream, object_pairs_hook=unique_object)
        validate_predictions(decoded, expected_names)
        if decoded != predictions:
            raise ValueError("Serialized predictions differ from the original predictions")
        staged.replace(output)


def main():
    parser = argparse.ArgumentParser(description="Predict visible fire directly from images.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--images", type=Path, help="Directory of exactly 5313 test images")
    parser.add_argument("--dinov3-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--last", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = config()
    if args.image is not None:
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        paths = [args.image]
        expected_names = {args.image.name}
        output = args.output
    else:
        paths = images(args.images)
        expected_names = submission_names(paths, settings["submission_image_count"])
        output = args.output or Path("submission.json")
    if output is not None and output.exists():
        raise FileExistsError(output)

    digest = weight_sha256(args.weights, settings["backbone"])
    model = load_head(args.last, settings, digest)
    encoder = FrozenDino.from_pretrained(args.dinov3_source, args.weights, settings)
    predictions = {path.name: predict_image(path, encoder, model, settings) for path in paths}
    if output is not None:
        publish_predictions(output, predictions, expected_names)
    if args.image is not None:
        print(predictions[args.image.name])


if __name__ == "__main__":
    main()
