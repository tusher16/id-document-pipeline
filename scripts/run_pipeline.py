import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run ID document processing pipeline")
    parser.add_argument("--input", required=True, help="Path to input image folder")
    parser.add_argument("--output", default="outputs", help="Output folder")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    if not images:
        print(f"No images found in: {input_dir}")
        return

    print(f"Found {len(images)} images")
    print("Next: connect this runner to src/pipeline.py")

    # Placeholder loop
    for img_path in images:
        # Later: call your pipeline here and save outputs
        print(f"Processing: {img_path.name}")

    print(f"Done. Outputs should go to: {output_dir}")

if __name__ == "__main__":
    main()
