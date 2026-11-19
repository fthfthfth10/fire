import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from common import config, read_json, save_pt, write_json
from data import Bags, collate_bags, fill_cache, training_records
from infer import load_head, publish_predictions, submission_names, validate_predictions
from model import FireMIL, FrozenDino, fire_decisions, objective, weight_sha256
from tiles import IGNORE, coco_box_to_xyxy, open_rgb, tile_boxes, tile_labels
from train import initialize_run


SETTINGS = config()
SOURCE = os.environ.get("FIRE_MIL_DINOV2_SOURCE")
WEIGHTS = os.environ.get("FIRE_MIL_DINOV2_WEIGHTS")
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

    def test_instance_only_gradient_and_shared_abmil_input(self):
        model = FireMIL(SETTINGS)
        features = torch.ones(1, 3, SETTINGS["backbone"]["feature_dim"], dtype=torch.bfloat16)
        mask = torch.tensor([[True, True, False]])
        observed = {}
        projector = model.projection.register_forward_hook(
            lambda module, inputs, output: observed.update(projected=output))
        bag = model.bag.register_forward_pre_hook(
            lambda module, inputs: observed.update(bag=inputs[0]))
        instance = model.instance.register_forward_pre_hook(
            lambda module, inputs: observed.update(instance=inputs[0]))
        with torch.no_grad():
            model.instance.weight.fill_(1 / features.shape[-1])
            model.instance.bias.zero_()
        try:
            bag_logits, tile_logits = model(features, mask)
        finally:
            projector.remove()
            bag.remove()
            instance.remove()
        self.assertIs(observed["bag"], observed["projected"])
        self.assertIs(observed["instance"], observed["projected"])
        _, _, instance_loss = objective(bag_logits, tile_logits, torch.tensor([1.0]),
                                         torch.tensor([[IGNORE, 1, IGNORE]]),
                                         mask, torch.tensor(1.0), 0.30)
        instance_loss.backward()
        gradient = model.projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(gradient.isfinite().all().item())
        self.assertGreater(gradient.norm().item(), 0.0)
        self.assertTrue(all(p.grad is None for p in model.bag.parameters()))

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
            checkpoint["identity"]["config"] = {**settings, "schema_version": 2}
            save_pt(path, checkpoint)
            with self.assertRaisesRegex(ValueError, "locked configuration"):
                load_head(path, settings, "fixture")
            checkpoint["identity"]["config"] = settings
            checkpoint["epoch"] = 49
            save_pt(path, checkpoint)
            with self.assertRaises(ValueError):
                load_head(path, settings, "fixture")

    def test_model_rejects_unverified_native_weight_file(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp)
            metadata = SETTINGS["backbone"]
            path = root / metadata["weights_name"]
            path.write_bytes(b"not a verified checkpoint")
            with patch("model.build_backbone") as build:
                with self.assertRaisesRegex(ValueError, "checkpoint size"):
                    FrozenDino.from_pretrained(root, path, SETTINGS)
                with path.open("r+b") as stream:
                    stream.truncate(metadata["weights_bytes"])
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    FrozenDino.from_pretrained(root, path, SETTINGS)
                build.assert_not_called()

    def test_previous_cache_identity_rejected_on_resume(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            run = Path(temp)
            write_json(run / "run.json", {"config": {**SETTINGS, "schema_version": 2}})
            with self.assertRaisesRegex(ValueError, "Resume requires the same"):
                initialize_run(run, {"config": SETTINGS}, resume=True)

    @unittest.skipUnless(SOURCE and WEIGHTS, "Set FIRE_MIL_DINOV2_SOURCE and FIRE_MIL_DINOV2_WEIGHTS")
    def test_pretrained_checkpoint_strict_load(self):
        device = SETTINGS["device"] if torch.cuda.is_available() else "cpu"
        settings = {**SETTINGS, "device": device}
        path = Path(WEIGHTS)
        digest = weight_sha256(path, settings["backbone"])
        encoder = FrozenDino.from_pretrained(Path(SOURCE), path, settings)
        self.assertTrue(all(not p.requires_grad for p in encoder.backbone.parameters()))
        image = Image.new("RGB", (1024, 512))
        result = encoder.encode(image)
        self.assertEqual(result.shape, (len(tile_boxes(*image.size, settings["tiles"])), 384))
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(result.isfinite().all().item())
        print(f"Verified official {path.name} SHA-256 {digest}, strict load, shape {tuple(result.shape)}")


class SubmissionTests(unittest.TestCase):
    def test_5313_synthetic_filenames_round_trip(self):
        count = SETTINGS["submission_image_count"]
        self.assertEqual(count, 5313)
        paths = [Path(f"synthetic_{index:04}.jpg") for index in range(count)]
        names = submission_names(paths, count)
        predictions = {path.name: index % 2 for index, path in enumerate(paths)}
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            output = Path(temp) / "submission.json"
            publish_predictions(output, predictions, names)
            self.assertEqual(read_json(output), predictions)
            with self.assertRaises(FileExistsError):
                publish_predictions(output, predictions, names)

    def test_submission_rejects_count_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "Expected 5313"):
            submission_names([Path("one.jpg")], SETTINGS["submission_image_count"])
        for duplicate in ("one.jpg", "ONE.JPG"):
            with self.subTest(duplicate=duplicate):
                with self.assertRaisesRegex(ValueError, "unique"):
                    submission_names([Path("one.jpg"), Path("other") / duplicate], 2)

    def test_submission_rejects_bad_keyset_or_labels(self):
        expected = {"one.jpg", "two.jpg"}
        for predictions in (
            {"one.jpg": 0}, {"one.jpg": 0, "extra.jpg": 1},
            {"one.jpg": 0, "two.jpg": 1, "extra.jpg": 0},
            {"one.jpg": 0, "two.jpg": True}, {"one.jpg": 0, "two.jpg": 1.0},
            {"one.jpg": 0, "two.jpg": "1"}, {"one.jpg": 0, "two.jpg": 2},
        ):
            with self.subTest(predictions=predictions):
                with self.assertRaises(ValueError):
                    validate_predictions(predictions, expected)

    def test_invalid_staged_json_never_published(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            output = Path(temp) / "submission.json"
            for content in ('{"one.jpg": 1, "one.jpg": 1}', '{"one.jpg": 0}', '{"one.jpg":'):
                with self.subTest(content=content), patch("infer.write_json") as writer:
                    writer.side_effect = lambda path, values: path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        publish_predictions(output, {"one.jpg": 1}, {"one.jpg"})
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
