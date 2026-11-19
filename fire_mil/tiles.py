import math

from PIL import Image, ImageOps
from torchvision.transforms import Compose, Normalize, ToTensor


IGNORE = -1
NORMALIZE = Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def axis_starts(length, side, stride):
    last = length - side
    return sorted(set(range(0, last + 1, stride)) | {last})


def tile_boxes(width, height, settings):
    if min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive")
    whole = (0, 0, width, height)
    tiles = []
    for side in dict.fromkeys(min(scale, width, height) for scale in settings["scales_px"]):
        stride = max(1, int(side * settings["stride_fraction"]))
        for y in axis_starts(height, side, stride):
            for x in axis_starts(width, side, stride):
                # Keep a local tile even when it is the whole square image so it
                # can receive instance supervision.
                tiles.append((x, y, x + side, y + side))
    return [whole, *tiles]


def coco_box_to_xyxy(bbox, width, height):
    x, y, w, h = bbox
    if not all(math.isfinite(v) for v in bbox) or min(w, h) <= 0:
        raise ValueError("Invalid COCO box extent")
    box = (max(0.0, x), max(0.0, y), min(width, x + w), min(height, y + h))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("COCO box lies outside the image")
    return box


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def tile_labels(tiles, fire_boxes, settings):
    local = tiles[1:]
    labels = [0] * len(local)
    threshold = settings["positive_box_coverage"]
    for fire in fire_boxes:
        coverages = [intersection(tile, fire) / area(fire) for tile in local]
        if not any(coverages):
            raise ValueError("A fire box has no overlapping local tile")
        positive = {index for index, coverage in enumerate(coverages) if coverage >= threshold}
        if not positive:
            # Guarantee supervision for each box, including fires larger than a
            # tile and fires cut by multiple grid boundaries.
            positive.add(max(range(len(local)), key=coverages.__getitem__))
        for index, coverage in enumerate(coverages):
            if index in positive:
                labels[index] = 1
            elif coverage > 0 and labels[index] == 0:
                labels[index] = IGNORE
    return [IGNORE, *labels]


def open_rgb(source):
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def tile_tensor(image, box, input_size, *, full_image):
    size = (input_size, input_size)
    if full_image:
        # Preserve framing and aspect ratio without cropping off edge objects.
        resized = ImageOps.pad(image, size, method=Image.Resampling.BICUBIC, color=(124, 116, 104))
    else:
        resized = image.crop(box).resize(size, resample=Image.Resampling.BICUBIC)
    return NORMALIZE(resized)
