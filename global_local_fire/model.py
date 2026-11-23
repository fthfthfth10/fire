"""Global-local evidence on Ultralytics YOLOv8m-P2; pin the raw-head API."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
import ultralytics
from ultralytics.nn.tasks import DetectionModel, load_checkpoint, yaml_model_load
from ultralytics.utils.loss import v8DetectionLoss

from .data import FIRE, SMOKE
from .io_utils import sha256


ULTRALYTICS_VERSION = "8.4.0"
STRIDES = (4, 8, 16, 32)


class GlobalLocalFire(nn.Module):
    def __init__(self, settings, pretrained=None):
        super().__init__()
        if ultralytics.__version__ != ULTRALYTICS_VERSION:
            raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS_VERSION}, got {ultralytics.__version__}")
        arch = yaml_model_load("yolov8m-p2.yaml")
        if arch["scale"] != "m":
            raise RuntimeError("The upstream YAML did not resolve to YOLOv8m-P2")
        detector = DetectionModel(arch, nc=2, verbose=False)
        head = detector.model[-1]
        if tuple(head.stride.tolist()) != STRIDES or head.nc != 2 or head.end2end:
            raise RuntimeError("Unexpected P2/P3/P4/P5 Detect head")
        self.pretrained_layers = self.transfer(detector, pretrained, settings["pretrained_sha256"]) if pretrained else 0
        self.neck = detector
        self.detect = head
        feature_tap = nn.Identity()
        feature_tap.f, feature_tap.i = head.f, head.i
        self.neck.model[-1] = feature_tap
        strides = head.stride
        del head.stride
        head.register_buffer("stride", strides, persistent=False)
        channels = [branch[0].conv.in_channels for branch in head.cv2]
        hidden = settings["hidden_dim"]
        self.top_k = settings["top_k"]
        self.scene = nn.Sequential(nn.LayerNorm(sum(channels)), nn.Linear(sum(channels), hidden), nn.SiLU())
        self.global_head = nn.Linear(hidden, 1)
        self.gates = nn.ModuleList(nn.Linear(hidden, width) for width in channels)
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.fusion = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))

    @staticmethod
    def transfer(target, weights, expected_sha):
        if not weights.is_file() or sha256(weights) != expected_sha:
            raise ValueError("Missing or unverified official YOLOv8m COCO weights")
        source, _ = load_checkpoint(weights, device="cpu")
        source_specs = source.yaml["backbone"] + source.yaml["head"]
        target_specs = target.yaml["backbone"] + target.yaml["head"]
        max_channels = target.yaml["scales"]["m"][2]

        def effective_spec(spec):
            inputs, repeats, layer, args = spec
            if layer in {"Conv", "C2f", "SPPF"}:
                args = [min(args[0], max_channels), *args[1:]]
            return inputs, repeats, layer, args

        matched = 0
        for spec_src, spec_dest, src, dest in zip(source_specs, target_specs, source.model, target.model):
            old, new = src.state_dict(), dest.state_dict()
            if (effective_spec(spec_src) != effective_spec(spec_dest) or type(src) is not type(dest) or old.keys() != new.keys()
                    or any(old[key].shape != new[key].shape for key in old)):
                break
            dest.load_state_dict(old, strict=True)
            matched += 1
        if matched < len(target.yaml["backbone"]):
            raise ValueError("Official COCO checkpoint did not match the full YOLOv8m backbone")
        return matched

    def forward(self, images, *, with_detection=False):
        pyramid = self.neck(images)
        scene = self.scene(torch.cat([feature.mean(dim=(2, 3), dtype=torch.float32) for feature in pyramid], dim=1))
        global_logit = self.global_head(scene).squeeze(1).float()
        local = []
        for feature, gate in zip(pyramid, self.gates):
            multiplier = 2 * gate(scene).sigmoid().to(feature.dtype)
            local.append(feature * multiplier[:, :, None, None])
        raw = self.detect.forward_head(local, **self.detect.one2many) if with_detection else None
        if raw is None:
            scores = torch.cat([classifier(feature).flatten(2) for classifier, feature in zip(self.detect.cv3, local)],
                               dim=2)
        else:
            scores = raw["scores"]
        evidence = scores.float().topk(min(self.top_k, scores.shape[-1]), dim=-1).values.mean(dim=-1)
        fire_logit, smoke_logit = evidence[:, FIRE], evidence[:, SMOKE]
        fusion_logit = self.fusion(torch.stack((global_logit, fire_logit, smoke_logit), dim=1)).squeeze(1).float()
        return {"detection": raw, "global": global_logit, "mil": fire_logit, "smoke": smoke_logit,
                "fusion": fusion_logit}


class JointLoss:
    def __init__(self, model, settings):
        detector_view = SimpleNamespace(parameters=model.parameters, args=SimpleNamespace(**settings["detection_gains"]),
                                        model=[model.detect])
        self.detection = v8DetectionLoss(detector_view)
        self.weights = settings["loss_weights"]

    def __call__(self, output, batch):
        raw = output["detection"]
        fp32 = {"scores": raw["scores"].float(), "boxes": raw["boxes"].float(),
                "feats": [feature[:, :1].float() for feature in raw["feats"]]}
        known = batch["known"][:, None, :]
        original_bce = self.detection.bce
        self.detection.bce = lambda scores, targets: original_bce(scores, targets) * known
        try:
            _, components, _ = self.detection.get_assigned_targets_and_loss(fp32, batch)
        finally:
            self.detection.bce = original_bce
        losses = {"detection": components.sum()}
        for name in ("global", "mil", "fusion"):
            losses[name] = F.binary_cross_entropy_with_logits(output[name], batch["labels"])
        total = losses["detection"] + sum(self.weights[name] * losses[name] for name in self.weights)
        return total, losses
