from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

def list_images(folder: str | Path):
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def read_bgr(path: str | Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def save_bgr(path: str | Path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def resize_keep_aspect(img, target_w: int, target_h: int):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas, (x0, y0, new_w, new_h)
