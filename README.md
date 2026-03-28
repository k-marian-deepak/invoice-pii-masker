# Invoice PII Masker (OCR + REGEX)

A from-scratch, local-first tool to **mask personal/sensitive information in invoice images** using OCR + regex + lightweight training memory. 
- **Training (paired)**: Reads `images/*.png` and `labels/*.txt` (same basenames), learns what fields to find, and stores helpful vendor-specific cues in `data/memory.json`.
- **Inference (unpaired)**: Only images are provided; the app uses OCR + regex and the learned memory to find and mask PII.
- **Masking modes**: black (default), blur, pixelate, or any solid color.
- **Downloadable** outputs (single or batch). 
- **FastAPI** backend with a minimalist frontend for testing.

> Tested with Python 3.10+. Requires **Tesseract OCR** installed on your system and in PATH.

## 1) Setup (virtual environment)

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

### Dependency troubleshooting (recommended)

If setup fails due to package version mismatches, run:

```bash
sed -i 's/^numpy==1\.24\.3$/numpy==1.26.4/' requirements.txt
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

Notes:
- If `requirements.txt` already contains `numpy==1.26.4`, the `sed` step is a no-op and safe to run.
- Prefer `python -m pip ...` to ensure installs go into the currently active virtual environment.

### Install Tesseract OCR
- Windows: download from UB Mannheim build or tesseract-ocr.github.io, ensure `tesseract.exe` is in PATH.
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

If Tesseract isn't on PATH, set it in **`.env`** or as an env var:
```
TESSERACT_CMD=/usr/local/bin/tesseract
```

## 2) Project structure

```
invoice-pii-masker/
  app/
    main.py                # FastAPI server with upload & batch endpoints
  core/
    ocr.py                 # OCR helpers (word boxes, line boxes)
    masking.py             # black/blur/pixelate/color mask helpers
    patterns.py            # built-in regex for GSTIN, PAN, phone, email, etc.
    alignment.py           # maps field labels to likely value regions
  training/
    train_memory.py        # learns vendor cues from PNG/TXT pairs
  scripts/
    batch_process.py       # CLI: process a folder of images
  static/
    index.html             # simple UI for upload/test
  data/
    memory.json            # learned cues stored here
  requirements.txt
  README.md
```

## 3) Training with paired data

Place files like:
```
dataset/
  images/
    1.png
    2.png
    ...
  labels/
    1.txt
    2.txt
    ...
```

Each `labels/*.txt` can list **fields to mask** (one per line), e.g.:
```
GSTIN
Phone
Email
PAN
```

Run trainer:
```
python -m training.train_memory --images dataset/images --labels dataset/texts
```
This updates `data/memory.json` with vendor cues (e.g., synonyms and nearby anchors).

## 4) Run the server

```bash
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000 to use the minimal UI.

## 5) Batch process via CLI

```bash
python -m scripts.batch_process --input some_images --out out_masks --mode black --color "#000000"
# modes: black | blur | pixelate | color
```

## 6) Notes about detection

- OCR extracts **word boxes**. 
- Regex in `core/patterns.py` finds sensitive **values** like GSTIN, PAN, email, phone, PIN code, dates, amounts.
- **Label anchoring**: for known labels (e.g., "GST", "GSTIN", "Goods and Services Tax"), the value is usually to the **right on the same line** or **directly below**; `alignment.py` tries both.
- Training adds *vendor-specific synonyms* encountered in the TXT files to improve label matching later.

## 7) Frontend integration

A minimal form is included at `/` for quick testing. You can replace it with your own Cloud AI template—just keep the POST endpoint `/mask` and fields `file`, `mode`, `color`, `pixel_size`.

## 8) License

MIT (you can use/modify freely).
