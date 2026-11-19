import hashlib
import math
import os
from itertools import islice

from dataset import PROJECT

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "ultralytics_settings"))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
from ultralytics import YOLO

from openai_clip_model import build_model


CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
CLIP_WEIGHTS = PROJECT / "assets" / "ViT-B-32.pt"
YOLO_PRETRAIN = PROJECT / "assets" / "yolov8n.pt"
DEVICE = "cuda:0"


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batches(values, size):
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def open_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def extract_crop(image, box, margin):
    # D5 cell i2-15. Expand each full side length by margin around the box center.
    left, top, right, bottom = box
    dx = (right - left) * margin / 2
    dy = (bottom - top) * margin / 2
    return image.crop((max(0, math.floor(left - dx)), max(0, math.floor(top - dy)),
                       min(image.width, math.ceil(right + dx)), min(image.height, math.ceil(bottom + dy))))


class ClipEncoder:
    def __init__(self):
        if sha256(CLIP_WEIGHTS) != CLIP_SHA256:
            raise ValueError("OpenAI ViT-B/32 checkpoint checksum mismatch")
        # Same reconstruction as OpenAI clip.load(jit=False), using the upstream model.py.
        archive = torch.jit.load(str(CLIP_WEIGHTS), map_location="cpu")
        model = build_model(archive.state_dict())
        self.visual = model.visual.requires_grad_(False).eval().to(DEVICE)
        if self.visual.input_resolution != 224 or self.visual.output_dim != 512:
            raise ValueError("Expected OpenAI CLIP ViT-B/32 visual features")
        self.preprocess = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def encode(self, crops):
        pixels = torch.stack([self.preprocess(crop) for crop in crops]).to(DEVICE)
        features = self.visual(pixels.to(self.visual.conv1.weight.dtype)).float()
        return F.normalize(features, dim=-1)


def detections(detector, paths, config, confidence):
    results = detector.predict(
        source=[str(path) for path in paths], device=DEVICE, imgsz=config["image_size"],
        conf=confidence, iou=config["nms_iou"], classes=[0], half=True, verbose=False,
    )
    if len(results) != len(paths):
        raise RuntimeError("Unexpected YOLO prediction count")
    for path, result in zip(paths, results):
        if path.name != os.path.basename(result.path):
            raise RuntimeError(f"YOLO prediction order mismatch for {path}")
        yield result.orig_img, result.boxes.data.cpu().tolist()


def load_head(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["clip_sha256"] != CLIP_SHA256:
        raise ValueError("Probe was fit with a different CLIP checkpoint")
    head = nn.Linear(512, 1)
    head.load_state_dict(state["head"])
    return head.requires_grad_(False).eval().to(DEVICE)


class FireCascade:
    def __init__(self, detector_path, head_path, config):
        self.detector = YOLO(str(detector_path))
        if self.detector.names != {0: "fire"}:
            raise ValueError("The detector checkpoint does not map class 0 to fire")
        self.clip = ClipEncoder()
        self.head = load_head(head_path)
        self.config = config

    @torch.inference_mode()
    def predict(self, paths):
        config = self.config
        predictions = []
        owners = []
        crops = []
        for pixels, boxes in detections(self.detector, paths, config, config["candidate_confidence"]):
            image_index = len(predictions)
            fire = [box for box in boxes if int(box[5]) == 0]
            auto_keep = any(box[4] >= config["auto_keep_confidence"] for box in fire)
            predictions.append(int(auto_keep))
            if fire and not auto_keep:
                # Reuse the exact decoded pixels and coordinate frame from Ultralytics.
                image = Image.fromarray(pixels[:, :, ::-1])
                for box in fire:
                    crops.append(extract_crop(image, box[:4], config["crop_margin"]))
                    owners.append(image_index)
        offset = 0
        for batch in batches(crops, config["clip_batch"]):
            probabilities = self.head(self.clip.encode(batch)).sigmoid().flatten().tolist()
            for owner, probability in zip(owners[offset:offset + len(batch)], probabilities):
                if probability >= config["clip_fire_probability"]:
                    predictions[owner] = 1
            offset += len(batch)
        return predictions
