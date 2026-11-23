"""Fixed dataset manifests, explicit annotation coverage, shared letterbox."""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from ultralytics.data.augment import LetterBox

from .io_utils import read_json, sha256


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Record:
    path: Path
    label: int
    digest: str
    pixel_digest: str
    group: str
    fire: tuple


def boxes(raw):
    if not isinstance(raw, list):
        raise ValueError("Boxes must be a JSON array of normalized xyxy coordinates")
    output = []
    for box in raw:
        if not isinstance(box, list) or len(box) != 4 or any(type(value) not in (int, float) for value in box):
            raise ValueError(f"Invalid box: {box!r}")
        if any(not math.isfinite(value) or not (-0.003 <= value <= 1.003) for value in box):
            raise ValueError(f"Invalid normalized box: {box!r}")
        clamped = tuple(min(1.0, max(0.0, value)) for value in box)
        if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
            raise ValueError(f"Invalid clipped box: {box!r}")
        output.append(clamped)
    return tuple(output)


def load_split(root, split):
    if split not in ("train", "validation"):
        raise ValueError("Only fixed train and validation manifests are accepted")
    root = root.resolve()
    rows = read_json(root / f"{split}.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Invalid dataset manifest")
    records, seen_paths, seen_digests = [], set(), set()
    for row in rows:
        key = row["path"]
        relative = Path(key)
        label = row["y"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Invalid Fire label: {key}")
        path = (root / relative).resolve()
        scene = "fire" if label else "no_fire"
        if (len(relative.parts) != 4 or relative.parts[:3] != ("images", split, scene)
                or relative.name in (".", "..") or path.parent != root / "images" / split / scene
                or path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file()):
            raise ValueError(f"Invalid or missing {split} image: {key}")
        digest = row["sha256"]
        if path in seen_paths or digest in seen_digests:
            raise ValueError(f"Duplicate path or declared hash: {key}")
        seen_paths.add(path)
        seen_digests.add(digest)
        fire = boxes(row.get("boxes", [])) if split == "train" else ()
        if fire and not label:
            raise ValueError(f"Fire boxes disagree with image label: {key}")
        records.append(Record(path, label, digest, row.get("pixel_sha256", ""), row["group"], fire))
    return records


def checked_splits(root):
    train = load_split(root, "train")
    val = load_split(root, "validation")
    for attribute in ("path", "digest", "pixel_digest", "group"):
        training = {getattr(row, attribute) for row in train if getattr(row, attribute)}
        validation = {getattr(row, attribute) for row in val if getattr(row, attribute)}
        if training & validation:
            raise ValueError(f"Train/validation overlap in {attribute}")
    return train, val


class FireImages(Dataset):
    def __init__(self, records, size):
        self.records = records
        self.size = size
        self.letterbox = LetterBox(new_shape=(size, size), auto=False, scaleup=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        height, width = rgb.shape[:2]
        ratio = min(self.size / width, self.size / height)
        pad_x = round((self.size - round(width * ratio)) / 2 - 0.1)
        pad_y = round((self.size - round(height * ratio)) / 2 - 0.1)
        coords = [(((left + right) * width * ratio / 2 + pad_x) / self.size,
                   ((top + bottom) * height * ratio / 2 + pad_y) / self.size,
                   (right - left) * width * ratio / self.size,
                   (bottom - top) * height * ratio / self.size) for left, top, right, bottom in record.fire]
        image = torch.from_numpy(np.ascontiguousarray(self.letterbox(image=rgb).transpose(2, 0, 1))).clone()
        return {"images": image, "label": record.label, "known": bool(record.fire) or record.label == 0,
                "bboxes": torch.tensor(coords, dtype=torch.float32).reshape(-1, 4),
                "cls": torch.zeros(len(coords), dtype=torch.float32), "name": record.path.name}


def collate(rows):
    return {"images": torch.stack([row["images"] for row in rows]),
            "labels": torch.tensor([row["label"] for row in rows], dtype=torch.float32),
            "known": torch.tensor([row["known"] for row in rows], dtype=torch.bool),
            "bboxes": torch.cat([row["bboxes"] for row in rows]), "cls": torch.cat([row["cls"] for row in rows]),
            "batch_idx": torch.repeat_interleave(torch.arange(len(rows)),
                                                  torch.tensor([len(row["cls"]) for row in rows])),
            "names": [row["name"] for row in rows]}


def official_test_records(path, count, protected_hashes):
    files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if len(files) != count or len({item.name.casefold() for item in files}) != count:
        raise ValueError(f"Expected {count} unique official test filenames")
    if any(sha256(item) in protected_hashes for item in files):
        raise ValueError("Official test contains a byte-identical training or validation image")
    return [Record(item, -1, "", "", "", ()) for item in files]
