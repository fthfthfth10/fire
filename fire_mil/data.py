from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import torch
from torch.utils.data import Dataset

from common import images, json_digest, read_json, save_pt, sha256_file
from tiles import IGNORE, coco_box_to_xyxy, open_rgb, tile_boxes, tile_labels


@dataclass(frozen=True)
class TrainingImage:
    path: Path
    width: int
    height: int
    fire: bool
    tiles: tuple
    targets: tuple
    sha256: str


def training_records(directory, settings, expected_count=1100):
    images_dir = directory / "images"
    disk = {p.name: p for p in images(images_dir)}
    coco_path = directory / "train_coco.json"
    image_labels_path = directory / "train_image.json"
    coco = read_json(coco_path)
    image_labels = read_json(image_labels_path)
    by_id = {record["id"]: record for record in coco["images"]}
    names = {record["file_name"] for record in by_id.values()}
    if not (
        len(coco["images"]) == len(by_id) == len(names) == expected_count
        and names == disk.keys() == image_labels.keys()
    ):
        raise ValueError(f"Expected {expected_count} unique training images paired with both annotations")
    categories = {category["id"] for category in coco["categories"]}
    if 0 not in categories or categories - {0, 1}:
        raise ValueError("Training classes must use 0 for fire and may use 1 for smoke")
    if len({box["id"] for box in coco["annotations"]}) != len(coco["annotations"]):
        raise ValueError("Duplicate COCO annotation IDs")

    fire_boxes = defaultdict(list)
    for annotation in coco["annotations"]:
        image_id, category = annotation["image_id"], annotation["category_id"]
        if image_id not in by_id or category not in categories:
            raise ValueError(f"Invalid COCO annotation {annotation['id']}")
        record = by_id[image_id]
        box = coco_box_to_xyxy(annotation["bbox"], record["width"], record["height"])
        if category == 0:
            fire_boxes[image_id].append(box)

    samples = []
    for image_id, record in sorted(by_id.items()):
        name, width, height = record["file_name"], record["width"], record["height"]
        truth = bool(fire_boxes[image_id])
        if type(image_labels[name]) is not int or image_labels[name] != int(truth):
            raise ValueError(f"Image-level label disagrees with visible-fire boxes: {name}")
        boxes = tile_boxes(width, height, settings["tiles"])
        targets = tile_labels(boxes, fire_boxes[image_id], settings["tiles"])
        samples.append(TrainingImage(disk[name], width, height, truth, tuple(boxes), tuple(targets),
                                     sha256_file(disk[name])))
    fingerprint = json_digest({
        "labels": [sha256_file(coco_path), sha256_file(image_labels_path)],
        "images": [[sample.path.name, sample.sha256] for sample in samples],
    })
    return samples, fingerprint


def cache_path(directory, index):
    return directory / f"{index:06d}.pt"


def encode_training_image(sample, encoder, settings):
    contents = sample.path.read_bytes()
    if sha256(contents).hexdigest() != sample.sha256:
        raise ValueError(f"Training image changed while building the cache: {sample.path.name}")
    image = open_rgb(BytesIO(contents))
    if image.size != (sample.width, sample.height):
        raise ValueError(f"EXIF-oriented dimensions disagree with COCO for {sample.path.name}: {image.size}")
    boxes = tile_boxes(*image.size, settings["tiles"])
    if tuple(boxes) != sample.tiles:
        raise RuntimeError(f"Training image and inference geometry disagree for {sample.path.name}")
    return encoder.encode_tiles(image, boxes)


def fill_cache(directory, samples, encoder_factory, settings):
    directory.mkdir(exist_ok=True)
    missing = [index for index in range(len(samples)) if not cache_path(directory, index).is_file()]
    if not missing:
        return
    encoder = encoder_factory()
    for count, index in enumerate(missing, 1):
        features = encode_training_image(samples[index], encoder, settings)
        save_pt(cache_path(directory, index), features)
        if count % 50 == 0 or count == len(missing):
            print(f"DINO cache: {count}/{len(missing)} newly encoded images", flush=True)


class Bags(Dataset):
    def __init__(self, directory, samples, feature_dim):
        self.directory = directory
        self.samples = samples
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        features = torch.load(cache_path(self.directory, index), map_location="cpu", weights_only=True)
        if features.shape != (len(sample.tiles), self.feature_dim) or features.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected cache tensor for {sample.path.name}")
        return features, float(sample.fire), torch.tensor(sample.targets, dtype=torch.int8)


def collate_bags(batch):
    count = len(batch)
    capacity = max(x.shape[0] for x, _, _ in batch)
    features = torch.zeros(count, capacity, batch[0][0].shape[1], dtype=torch.bfloat16)
    valid = torch.zeros(count, capacity, dtype=torch.bool)
    bag_targets = torch.empty(count, dtype=torch.float32)
    tile_targets = torch.full((count, capacity), IGNORE, dtype=torch.int8)
    for index, (bag, label, targets) in enumerate(batch):
        length = bag.shape[0]
        features[index, :length] = bag
        valid[index, :length] = True
        bag_targets[index] = label
        tile_targets[index, :length] = targets
    return features, valid, bag_targets, tile_targets
