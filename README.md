# ID Document Pipeline — Classification → Segmentation → Deskew → Cleaning → OCR

End-to-end ID document processing pipeline built for the Image Processing for Document Automation (IPDA) coursework.

Given an input image folder, the pipeline can:
1. detect whether an image is "ID-like" (classification)
2. segment and crop the ID card (segmentation)
3. deskew the cropped card (Hough/SIFT-based methods)
4. clean the card to improve text readability (U-Net-style cleaning)
5. run OCR (Tesseract) and export text to JSON


## Project structure
```
id-document-pipeline/
├── scripts/
│   ├── run_pipeline.py
│   ├── train_classifier.py
│   └── train_segmenter.py
├── src/
│   ├── pipeline.py
│   ├── classifier.py
│   ├── segmenter.py
│   ├── deskew.py
│   ├── ocr.py
│   └── utils.py
├── sample_data/
│   └── images/
├── outputs/
├── requirements.txt
└── README.md
```

## Setup

### 1) Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Install Tesseract (OCR engine)

**macOS:**
```bash
brew install tesseract
```

## Running the pipeline

### Fix for `ModuleNotFoundError: No module named 'src'`

If you run:
```bash
python scripts/run_pipeline.py ...
```

Python may not find the `src/` package depending on how your environment sets `PYTHONPATH`.

Use one of the options below:

**Option A (recommended): run with `PYTHONPATH`:**
```bash
PYTHONPATH=. python scripts/run_pipeline.py --input sample_data/images --output outputs
```

**Option B: run as a module (requires `scripts/__init__.py`):**
```bash
python -m scripts.run_pipeline --input sample_data/images --output outputs
```

### Run with optional OCR + optional deskew toggle
```bash
PYTHONPATH=. python scripts/run_pipeline.py \
  --input sample_data/images \
  --output outputs \
  --ocr
```

Disable deskew if needed:
```bash
PYTHONPATH=. python scripts/run_pipeline.py \
  --input sample_data/images \
  --output outputs \
  --ocr \
  --no-deskew
```

## Outputs

* Intermediate images saved into the output directory (segmented/deskewed/cleaned)
* OCR results saved as JSON (filename → list of text lines)

## Notes on ML models

This repo contains the pipeline code. Model weights (`.keras`) are not included unless you add them manually.

If you train models yourself, store them under a folder like:
```
models/
  id_classifier.keras
  id_segmentation.keras
  best_cleaning_model.keras
```

…and point your config/script to those paths.

## Requirements / Compatibility note (TensorFlow)

If you hit "Could not find a version that satisfies the requirement tensorflow", it usually means your Python version is too new for the available TensorFlow builds.

Use a stable Python version commonly supported by TensorFlow (e.g., 3.10/3.11) when training/running the ML models.

## License

MIT