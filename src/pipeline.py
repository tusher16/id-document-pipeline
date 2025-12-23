from dataclasses import dataclass
from pathlib import Path
import cv2

from .utils import read_bgr, save_bgr
from .deskew import deskew_hough

try:
    from .ocr import ocr_lines
except Exception:
    ocr_lines = None

@dataclass
class PipelineConfig:
    enable_ocr: bool = False
    deskew: bool = True

class DocumentPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def process_one(self, image_path: str | Path, out_dir: str | Path):
        out_dir = Path(out_dir)
        bgr = read_bgr(image_path)

        # 1) Deskew (classical CV) — works immediately
        if self.config.deskew:
            bgr = deskew_hough(bgr)

        # 2) Save “processed” image (later you’ll replace with cut-out, masks, etc.)
        out_img = out_dir / "processed" / Path(image_path).name
        save_bgr(out_img, bgr)

        # 3) OCR (optional)
        ocr = []
        if self.config.enable_ocr and ocr_lines is not None:
            ocr = ocr_lines(bgr)
            (out_dir / "ocr").mkdir(parents=True, exist_ok=True)
            (out_dir / "ocr" / (Path(image_path).stem + ".txt")).write_text("\n".join(ocr), encoding="utf-8")

        return {"image": str(image_path), "saved": str(out_img), "ocr_lines": ocr}
