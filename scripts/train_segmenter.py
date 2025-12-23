import argparse
from pathlib import Path
from src.segmenter import IDCardSegmenter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", required=True, help="Dataset base path")
    ap.add_argument("--out", default="models/segmenter.keras")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    model = IDCardSegmenter().build()
    model.train_from_basepath(args.base_path, epochs=args.epochs, batch_size=args.batch)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
