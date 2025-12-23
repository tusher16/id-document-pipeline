# ID Document Processing Pipeline (Document Automation)

Prototype pipeline for document image processing:
- ID-like vs non-ID classification
- ID region segmentation (U-Net)
- Deskew / alignment (classical CV)
- (Optional) OCR step

> Note: Course datasets are not included in this repository due to distribution restrictions.  
> Use your own images or the sample_data folder.

## Project structure
- `src/` core pipeline modules
- `scripts/` training + running scripts
- `notebooks/` experiments / exploration
- `sample_data/` small demo inputs (non-restricted)
- `assets/` diagrams + example outputs

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py --input sample_data/images --output outputs
