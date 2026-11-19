import hashlib
import sys
from importlib import import_module
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchmil.models import ABMIL

from common import sha256_file
from tiles import tile_boxes, tile_tensor


def source_sha256(source):
    digest = hashlib.sha256()
    paths = [source / "hubconf.py", *sorted((source / "dinov3").rglob("*.py"))]
    for path in paths:
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).digest())
    return digest.hexdigest()


def build_backbone(source, settings):
    source = source.resolve()
    if source_sha256(source) != settings["source_sha256"]:
        raise ValueError(f"DINOv3 source does not match commit {settings['commit']}")
    # Import the exact factory exported by hubconf.py without importing its
    # unrelated segmentation, detection and text model entry points.
    sys.path.insert(0, str(source))
    try:
        module = import_module("dinov3")
        if Path(module.__file__).resolve().parent != source / "dinov3":
            raise RuntimeError("A different DINOv3 package is already imported")
        backbones = import_module("dinov3.hub.backbones")
        backbone = backbones.dinov3_vits16(pretrained=False)
    finally:
        sys.path.pop(0)
    if backbone.embed_dim != settings["feature_dim"] or backbone.patch_size != 16:
        raise ValueError("Expected a 384-dimensional DINOv3 ViT-S/16")
    return backbone


def weight_sha256(weights, settings):
    if not weights.is_file():
        raise FileNotFoundError(f"Official Meta S/16 checkpoint required: {weights}")
    digest = sha256_file(weights)
    if not digest.startswith(settings["weights_sha256_prefix"]):
        raise ValueError(f"Checkpoint SHA-256 does not match {settings['weights_name']}: {digest}")
    return digest


class FrozenDino:
    def __init__(self, backbone, settings, device):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("The encoder requires native CUDA BF16 support")
        self.backbone = backbone.eval().requires_grad_(False).to(self.device)
        self.backbone_settings = settings["backbone"]
        self.tiles = settings["tiles"]

    @classmethod
    def from_pretrained(cls, source, weights, settings):
        weight_sha256(weights, settings["backbone"])
        backbone = build_backbone(source, settings["backbone"])
        state = torch.load(weights, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        return cls(backbone, settings, settings["device"])

    @torch.inference_mode()
    def encode(self, image):
        boxes = tile_boxes(*image.size, self.tiles)
        return self.encode_tiles(image, boxes)

    @torch.inference_mode()
    def encode_tiles(self, image, boxes):
        chunk_size = self.backbone_settings["encode_batch_size"]
        input_size = self.backbone_settings["input_size"]
        pieces = []
        for offset in range(0, len(boxes), chunk_size):
            batch = torch.stack([
                tile_tensor(image, box, input_size, full_image=(offset + index == 0))
                for index, box in enumerate(boxes[offset:offset + chunk_size])
            ]).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                tokens = self.backbone.forward_features(batch)["x_norm_clstoken"]
            pieces.append(tokens.to(device="cpu", dtype=torch.bfloat16))
        features = torch.cat(pieces, dim=0)
        if features.shape != (len(boxes), self.backbone_settings["feature_dim"]):
            raise RuntimeError(f"Unexpected DINOv3 features {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise FloatingPointError("DINOv3 produced nonfinite features")
        return features


class FireMIL(nn.Module):
    def __init__(self, settings):
        super().__init__()
        feature_dim = settings["backbone"]["feature_dim"]
        self.projection = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.projection.weight)
        self.bag = ABMIL(in_shape=(feature_dim,), att_dim=settings["mil"]["attention_dim"],
                         gated=True)
        self.instance = nn.Linear(feature_dim, 1)

    def forward(self, features, mask):
        shared = self.projection(features.float())
        return self.bag(shared, mask=mask), self.instance(shared).squeeze(-1)


def fire_decisions(logits, settings):
    return (logits.sigmoid() >= settings["fire_probability_threshold"]).to(torch.int64)


def objective(bag_logits, tile_logits, bag_targets, tile_targets, mask, bag_pos_weight, instance_weight):
    bag_loss = F.binary_cross_entropy_with_logits(bag_logits, bag_targets, pos_weight=bag_pos_weight)
    positive = mask & (tile_targets == 1)
    negative = mask & (tile_targets == 0)
    item_loss = F.binary_cross_entropy_with_logits(tile_logits, positive.float(), reduction="none")
    positives = positive.sum()
    negatives = negative.sum()
    per_class = (item_loss[positive].sum() / positives.clamp_min(1)
                 + item_loss[negative].sum() / negatives.clamp_min(1))
    classes_present = (positives > 0).int() + (negatives > 0).int()
    tile_loss = per_class / classes_present.clamp_min(1)
    return bag_loss + instance_weight * tile_loss, bag_loss, tile_loss
