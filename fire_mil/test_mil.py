import os
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from common import config, read_json, save_pt, write_json
from data import Bags, collate_bags, fill_cache, training_records
from infer import load_head, predict_image
from model import (
    FireMIL, FrozenDino, build_backbone, fire_decisions, objective, restore_backbone, weight_sha256,
)
from tiles import IGNORE, coco_box_to_xyxy, open_rgb, tile_boxes, tile_labels


SETTINGS = config()
SOURCE = os.environ.get("FIRE_MIL_DINOV3_SOURCE")
WEIGHTS = os.environ.get("FIRE_MIL_DINOV3_WEIGHTS")
TEMP_ROOT = os.environ.get("FIRE_MIL_TEST_TMP")


class GeometryTests(unittest.TestCase):
    def test_grid_and_one_pixel_fires(self):
        width, height = 547, 321
        boxes = tile_boxes(width, height, SETTINGS["tiles"])
        self.assertEqual(boxes[0], (0, 0, width, height))
        self.assertEqual({b[2] - b[0] for b in boxes[1:]}, {224, 321})
        for x, y in ((0, 0), (168, 168), (width - 1, height - 1), (width - 1, 0)):
            gt = coco_box_to_xyxy((x, y, 1, 1), width, height)
            labels = tile_labels(boxes, [gt], SETTINGS["tiles"])
            self.assertEqual(labels[0], IGNORE)
            self.assertTrue(any(
                label == 1 and left <= x < right and top <= y < bottom
                for (left, top, right, bottom), label in zip(boxes, labels)
            ))

    def test_partial_overlap_and_large_box_fallback(self):
        tiles = [(0, 0, 440, 224), (0, 0, 224, 224), (216, 0, 440, 224)]
        self.assertEqual(tile_labels(tiles, [(222, 60, 236, 80)], SETTINGS["tiles"]), [IGNORE, IGNORE, 1])
        self.assertEqual(tile_labels(tiles, [(222, 60, 236, 80), (10, 10, 12, 12)], SETTINGS["tiles"]),
                         [IGNORE, 1, 1])
        tiles = tile_boxes(1800, 1000, SETTINGS["tiles"])
        labels = tile_labels(tiles, [(0, 0, 1800, 1000)], SETTINGS["tiles"])
        self.assertEqual(labels.count(1), 1)
        self.assertEqual(labels.count(0), 0)

    def test_box_clipping(self):
        self.assertEqual(coco_box_to_xyxy((-2, 100, 10, 30), 200, 110), (0, 100, 8, 110))
        with self.assertRaises(ValueError):
            coco_box_to_xyxy((250, 100, 10, 30), 200, 110)

    def test_class_semantics_exif_and_cache_round_trip(self):
        class EncoderStub:
            def encode_tiles(self, image, boxes):
                self.last_boxes = boxes
                return torch.zeros(len(boxes), 384, dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp)
            images_dir = root / "images"
            images_dir.mkdir()
            exif = Image.Exif()
            exif[274] = 8
            Image.new("RGB", (173, 251)).save(images_dir / "fire.jpg", exif=exif)
            for name in ("smoke.jpg", "empty.jpg"):
                Image.new("RGB", (251, 173)).save(images_dir / name)
            write_json(root / "train_coco.json", {
                "categories": [{"id": 0}, {"id": 1}],
                "images": [{"id": i, "file_name": name, "width": 251, "height": 173}
                           for i, name in enumerate(("fire.jpg", "smoke.jpg", "empty.jpg"))],
                "annotations": [
                    {"id": 9, "image_id": 0, "category_id": 0, "bbox": [250, 172, 2, 2]},
                    {"id": 8, "image_id": 1, "category_id": 1, "bbox": [30, 30, 50, 50]},
                ],
            })
            write_json(root / "train_image.json", {"fire.jpg": 1, "smoke.jpg": 0, "empty.jpg": 0})
            samples, digest = training_records(root, SETTINGS, expected_count=3)
            self.assertEqual(len(digest), 64)
            self.assertEqual([s.fire for s in samples], [True, False, False])
            self.assertIn(1, samples[0].targets[1:])
            self.assertTrue(all(not any(s.targets[1:]) for s in samples[1:]))
            encoder = EncoderStub()
            fill_cache(root / "features", samples, lambda: encoder, SETTINGS)
            fill_cache(root / "features", samples, lambda: self.fail("Complete cache rebuilt the encoder"),
                       SETTINGS)
            expected = tile_boxes(*open_rgb(images_dir / "fire.jpg").size, SETTINGS["tiles"])
            self.assertEqual(encoder.last_boxes, expected)
            bag, target, _ = Bags(root / "features", samples, 384)[0]
            self.assertEqual((tuple(bag.shape), target), ((len(expected), 384), 1.0))


class ModelTests(unittest.TestCase):
    @torch.no_grad()
    def test_abmil_shapes_mask_and_loss(self):
        features, mask, targets, tile_targets = collate_bags([
            (torch.zeros(2, 384, dtype=torch.bfloat16), 1.0, torch.tensor([-1, 1])),
            (torch.ones(4, 384, dtype=torch.bfloat16), 0.0, torch.tensor([-1, 0, 0, 0])),
        ])
        model = FireMIL(SETTINGS).eval()
        logits, tile_logits = model(features, mask)
        self.assertEqual((tuple(logits.shape), tuple(tile_logits.shape)), ((2,), (2, 4)))
        modified = features.clone()
        modified[~mask] = 100
        torch.testing.assert_close(model(modified, mask)[0], logits)
        loss_args = (targets, tile_targets, mask, torch.tensor(1.0), 0.3)
        losses = objective(logits, tile_logits, *loss_args)
        self.assertTrue(all(torch.isfinite(loss).item() for loss in losses))
        changed = tile_logits.clone()
        changed[tile_targets == IGNORE] = 100
        for expected, actual in zip(losses, objective(logits, changed, *loss_args)):
            torch.testing.assert_close(expected, actual)
        self.assertEqual(fire_decisions(torch.tensor([-2.0, 0.0, 2.0]), SETTINGS).tolist(), [0, 1, 1])

    def test_last_head_load_complete_and_reject_mismatch(self):
        settings = {**SETTINGS, "device": "cpu"}
        model = FireMIL(settings)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            path = Path(temp) / "last.pt"
            checkpoint = {"epoch": 50, "identity": {"config": settings, "backbone_sha256": "fixture"},
                          "model": model.state_dict()}
            save_pt(path, checkpoint)
            restored = load_head(path, settings, "fixture")
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name])
            with self.assertRaises(ValueError):
                load_head(path, settings, "different")
            checkpoint["epoch"] = 49
            save_pt(path, checkpoint)
            with self.assertRaises(ValueError):
                load_head(path, settings, "fixture")

    @unittest.skipUnless(SOURCE, "Set FIRE_MIL_DINOV3_SOURCE to the pinned Meta source tree")
    def test_official_code_bf16_forward_and_synthetic_weight_load(self):
        device = SETTINGS["device"] if torch.cuda.is_available() else "cpu"
        settings = {**SETTINGS, "device": device}
        source = Path(SOURCE)
        backbone = build_backbone(source, settings["backbone"])
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp)
            fixture = root / "synthetic.pth"
            save_pt(fixture, backbone.state_dict())
            other = build_backbone(source, settings["backbone"])
            restore_backbone(other, fixture)
            torch.testing.assert_close(backbone.cls_token, other.cls_token)
            with self.assertRaises(ValueError):
                weight_sha256(fixture, settings["backbone"])
            del backbone
            encoder = FrozenDino(other, settings, device)
            self.assertTrue(all(not p.requires_grad for p in encoder.backbone.parameters()))
            path = root / "synthetic.png"
            Image.new("RGB", (1024, 512), color=(105, 120, 135)).save(path)
            image = open_rgb(path)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            tensors = encoder.encode(image)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                peak_mib = torch.cuda.max_memory_allocated() / 2**20
                print(f"CUDA synthetic encoder peak allocated: {peak_mib:.1f} MiB")
                self.assertLess(peak_mib, 4096)
            tiles = tile_boxes(*image.size, settings["tiles"])
            self.assertGreater(len(tiles), settings["backbone"]["encode_batch_size"])
            self.assertEqual(tensors.shape, (len(tiles), 384))
            self.assertEqual(tensors.dtype, torch.bfloat16)
            model = FireMIL(settings).eval().to(device)
            decision = predict_image(path, encoder, model, settings)
            self.assertIn(decision, (0, 1))
            submission = root / "submission.json"
            write_json(submission, {path.name: decision})
            self.assertEqual(read_json(submission), {path.name: decision})

    @unittest.skipUnless(SOURCE and WEIGHTS, "Set FIRE_MIL_DINOV3_WEIGHTS to an authorized Meta S/16 checkpoint")
    def test_pretrained_checkpoint_strict_load(self):
        device = SETTINGS["device"] if torch.cuda.is_available() else "cpu"
        settings = {**SETTINGS, "device": device}
        path = Path(WEIGHTS)
        weight_sha256(path, settings["backbone"])
        encoder = FrozenDino.from_pretrained(Path(SOURCE), path, settings)
        result = encoder.encode(Image.new("RGB", (224, 224)))
        self.assertEqual(result.shape, (2, 384))


if __name__ == "__main__":
    unittest.main()
