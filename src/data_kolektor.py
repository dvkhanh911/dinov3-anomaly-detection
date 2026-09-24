"""KolektorSDD parsing: layout, defect flags, masks, fixed train/test split."""

import os
import re
import struct

import numpy as np
from PIL import Image


def _mask_has_defect(mask_path):
    """Fast raw BMP check: any nonzero pixel after the pixel-data offset."""
    with open(mask_path, "rb") as fh:
        header = fh.read(30)
        offset = struct.unpack("<I", header[10:14])[0]
        fh.seek(offset)
        while True:
            chunk = fh.read(65536)
            if not chunk:
                return False
            if any(chunk):
                return True


def list_items(root):
    """[(rel_path, is_defective)] for all 399 images. Boards kos01..kos50."""
    boards = sorted(d for d in os.listdir(root) if re.fullmatch(r"kos\d+", d))
    items = []
    for board in boards:
        folder = os.path.join(root, board)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".jpg"):
                continue
            mask = os.path.join(folder, name.replace(".jpg", "_label.bmp"))
            items.append((f"{board}/{name}", _mask_has_defect(mask)))
    return items


def load_mask(mask_path, width, height):
    """BMP label (values 0/1) -> boolean mask at (width, height)."""
    arr = np.array(Image.open(mask_path))
    binary = (arr > 0).astype(np.uint8)
    resized = np.array(Image.fromarray(binary * 255).resize((width, height), Image.NEAREST))
    return resized > 127


def fixed_split(items, n_train_good=50, seed=42):
    """Same split as the 0.9507 baseline: first 50 shuffled goods for train."""
    import random
    rng = random.Random(seed)
    goods = [p for p, d in items if not d]
    defects = [p for p, d in items if d]
    rng.shuffle(goods)
    train = goods[:n_train_good]
    test = defects + goods[n_train_good:]
    rng.shuffle(test)
    return train, test
