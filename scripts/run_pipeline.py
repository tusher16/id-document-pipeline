import argparse
from pathlib import Path
from src.utils import list_images
from src.pipeline import DocumentPipeline, PipelineConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of images")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR (requires tesseract)")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskew step")
    args = parser.parse_args()

    imgs = list_images(args.input)
    if not imgs:
        print(f"No images found in {args.input}")
        return

    cfg = PipelineConfig(enable_ocr=args.ocr, deskew=not args.no_deskew)
    pipe = DocumentPipeline(cfg)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in imgs:
        result = pipe.process_one(p, out_dir)
        print(result["saved"])

if __name__ == "__main__":
    main()
