import argparse
from pathlib import Path

from src.utils import list_images
from src.pipeline import DocumentPipeline, PipelineConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Folder of images")
    p.add_argument("--output", default="outputs", help="Output folder")

    p.add_argument("--no-classify", action="store_true")
    p.add_argument("--no-segment", action="store_true")
    p.add_argument("--no-deskew", action="store_true")

    p.add_argument("--ocr", action="store_true", help="Enable OCR (requires tesseract)")
    p.add_argument("--classifier-model", default=None, help="Path to saved classifier model")
    p.add_argument("--segmenter-model", default=None, help="Path to saved segmenter model")

    args = p.parse_args()

    imgs = list_images(args.input)
    if not imgs:
        print(f"No images found in {args.input}")
        return

    cfg = PipelineConfig(
        do_classify=not args.no_classify,
        do_segment=not args.no_segment,
        do_deskew=not args.no_deskew,
        do_ocr=args.ocr,
        classifier_model_path=args.classifier_model,
        segmenter_model_path=args.segmenter_model,
    )

    pipe = DocumentPipeline(cfg)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img in imgs:
        r = pipe.process_one(img, out_dir)
        print(r.get("saved", r))

if __name__ == "__main__":
    main()
