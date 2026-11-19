import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def image_paths(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def prepare(run):
    source = ROOT / "data" / "train"
    coco = read_json(source / "train_coco.json")
    target = read_json(source / "train_image.json")
    original = {p.name: p for p in image_paths(source / "images")}
    records = {i["id"]: i for i in coco["images"]}
    names = {i["file_name"] for i in records.values()}
    if not (len(coco["images"]) == len(records) == len(names) == 1100 and names == original.keys() == target.keys()):
        raise ValueError("Expected 1100 unique COCO IDs and exact filename agreement with both training inputs")
    if len({p.stem.casefold() for p in original.values()}) != len(original):
        raise ValueError("The training filenames would collide as YOLO label stems")
    if {c["id"] for c in coco["categories"]} - {0, 1}:
        raise ValueError("Unknown COCO category")
    if len({a["id"] for a in coco["annotations"]}) != len(coco["annotations"]):
        raise ValueError("Duplicate annotation IDs")

    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in records or annotation["category_id"] not in (0, 1):
            raise ValueError(f"Invalid annotation {annotation['id']}")
        annotations[annotation["image_id"]].append(annotation)

    staging = run / "training_data"
    images = staging / "images"
    labels = staging / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    manifest, clipped, oriented = [], [], []
    counts = Counter()
    for image_id, record in sorted(records.items()):
        name = record["file_name"]
        src = original[name]
        width, height = record["width"], record["height"]
        boxes = []
        for annotation in annotations[image_id]:
            x, y, w, h = annotation["bbox"]
            if not all(map(math.isfinite, (x, y, w, h))) or min(w, h) <= 0:
                raise ValueError(f"Invalid bounding box in {name}: {annotation['id']}")
            xyxy = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
            x1, y1, x2, y2 = xyxy
            if min(x2 - x1, y2 - y1) <= 0:
                raise ValueError(f"Bounding box lies outside {name}: {annotation['id']}")
            if xyxy != [x, y, x + w, y + h]:
                clipped.append({"file": name, "annotation_id": annotation["id"], "xyxy": xyxy})
            category = annotation["category_id"]
            counts[category] += 1
            boxes.append({"category": category, "xyxy": xyxy})
        fire = [b["xyxy"] for b in boxes if b["category"] == 0]
        if type(target[name]) is not int or target[name] != int(bool(fire)):
            raise ValueError(f"Image-level fire label disagrees with COCO boxes: {name}")

        dst = images / name
        with Image.open(src) as image:
            orientation = image.getexif().get(274, 1)
            normalized = ImageOps.exif_transpose(image)
            if normalized.size != (width, height):
                raise ValueError(f"Image dimensions disagree with COCO: {name}: {normalized.size}")
            normalized.load()
            if orientation != 1:
                normalized.save(dst, quality=100, subsampling=0)
                oriented.append({"file": name, "exif_orientation": orientation})
            else:
                # Ultralytics can repair JPEGs in place, so stage copies of the originals.
                shutil.copyfile(src, dst)
        lines = []
        for x1, y1, x2, y2 in fire:
            values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                      (x2 - x1) / width, (y2 - y1) / height)
            lines.append("0 " + " ".join(f"{v:.10g}" for v in values))
        (labels / f"{src.stem}.txt").write_text("".join(line + "\n" for line in lines), encoding="ascii")
        manifest.append({"file": name, "image_id": image_id, "fire": target[name], "boxes": boxes})

    write_json(staging / "manifest.json", manifest)
    write_json(staging / "audit.json", {
        "images": len(manifest), "image_labels": dict(Counter(target.values())),
        "box_classes": dict(counts), "clipped": clipped, "exif_normalized": oriented,
    })
    print(f"Training data: {len(manifest)} images, {counts[0]} fire boxes, "
          f"{sum(not r['fire'] for r in manifest)} fire-negative images", flush=True)
    return manifest
