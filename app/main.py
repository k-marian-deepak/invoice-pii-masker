\
import io, os, json
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import cv2

from core.ocr import ocr_words, configure_tesseract
from core.patterns import PATTERNS, LABEL_SYNONYMS
from core.alignment import find_label_words, right_of, below, box_of_word
from core.masking import apply_mask  # type: ignore

app = FastAPI(title="Invoice PII Masker")

app.mount("/static", StaticFiles(directory="static"), name="static")

def load_memory() -> Dict[str, Any]:
    if os.path.exists("data/memory.json"):
        return json.load(open("data/memory.json","r"))
    return {"label_synonyms": LABEL_SYNONYMS}

def detect_boxes(img: Image.Image, mem: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    words = ocr_words(img)
    boxes: List[Tuple[int, int, int, int]] = []
    for w in words:
        txt = w["text"]
        for _, rgx in PATTERNS.items():
            if rgx.search(txt):
                boxes.append((w["left"], w["top"], w["width"], w["height"]))
    synonyms = mem.get("label_synonyms", LABEL_SYNONYMS)
    for _, terms in synonyms.items():
        label_hits = find_label_words(words, terms)
        for lh in label_hits:
            lb = box_of_word(lh)
            candidates = right_of(lb, words) or below(lb, words)  # type: ignore
            for c in candidates[:4]:  # type: ignore
                boxes.append((c["left"], c["top"], c["width"], c["height"]))  # type: ignore
    return merge_boxes(boxes)

def merge_boxes(boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.1) -> List[Tuple[int, int, int, int]]:
    if not boxes: 
        return []
    
    def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter
        return inter / union if union else 0.0
    
    merged: List[Tuple[int, int, int, int]] = []
    for b in sorted(boxes, key=lambda k: (k[1], k[0])):
        merged_into = False
        for i, m in enumerate(merged):
            if iou(b, m) > iou_thresh:
                x = min(m[0], b[0])
                y = min(m[1], b[1])
                w = max(m[0] + m[2], b[0] + b[2]) - x
                h = max(m[1] + m[3], b[1] + b[3]) - y
                merged[i] = (x, y, w, h)
                merged_into = True
                break
        if not merged_into:
            merged.append(b)
    return merged

@app.get("/", response_class=HTMLResponse)
async def root():
    return open("static/index.html","r",encoding="utf-8").read()

@app.post("/mask")
async def mask_endpoint(
    file: UploadFile,
    mode: str = Form("black"),
    color: str = Form("#000000"),
    pixel_size: int = Form(12),
    tesseract_cmd: Optional[str] = Form(None)
):
    if tesseract_cmd:
        configure_tesseract(tesseract_cmd)

    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    mem = load_memory()
    boxes = detect_boxes(img, mem)

    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    rgb = hex_to_rgb(color)
    for box in boxes:
        # Type cast mode to satisfy the literal type requirement
        apply_mask(img_bgr, box, mode=mode, color=rgb, pixel_size=pixel_size)  # type: ignore

    out = cv2.imencode(".png", img_bgr)[1].tobytes()
    return StreamingResponse(io.BytesIO(out), media_type="image/png",
                             headers={"Content-Disposition": f'attachment; filename="masked_{file.filename}"'})

def hex_to_rgb(hx: str) -> Tuple[int, int, int]:
    hx = hx.lstrip("#")
    if len(hx) == 3:
        hx = "".join([c*2 for c in hx])
    return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))  # type: ignore
