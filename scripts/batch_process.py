\
import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Literal
from PIL import Image
import numpy as np
import cv2

from core.ocr import ocr_words
from core.patterns import PATTERNS, LABEL_SYNONYMS
from core.alignment import find_label_words, right_of, below, box_of_word
from core.masking import apply_mask  # type: ignore

def detect_boxes(img: Image.Image, mem: Dict[str, Any]) -> List[Tuple[int,int,int,int]]:
    words = ocr_words(img)
    boxes: List[Tuple[int,int,int,int]] = []

    # 1) value-first: regex on concatenated words (map back by per-word matches)
    # simple per-word matching:
    for w in words:
        txt = w["text"]
        for _, rgx in PATTERNS.items():
            if rgx.search(txt):
                boxes.append((w["left"], w["top"], w["width"], w["height"]))

    # 2) label anchoring:
    synonyms = mem.get("label_synonyms", LABEL_SYNONYMS)
    for _, terms in synonyms.items():
        label_hits = find_label_words(words, terms)
        for lh in label_hits:
            lb = box_of_word(lh)
            # try right-side same line first, then below
            candidates: List[Dict[str, Any]] = right_of(lb, words) or below(lb, words)  # type: ignore
            # take the next few words as the value span
            for c in candidates[:4]:
                boxes.append((c["left"], c["top"], c["width"], c["height"]))
    # merge boxes (optional simple merge by proximity)
    merged = merge_boxes(boxes, iou_thresh=0.1)
    return merged

def merge_boxes(boxes: List[Tuple[int,int,int,int]], iou_thresh: float = 0.1) -> List[Tuple[int,int,int,int]]:
    if not boxes: 
        return []
    # convert to x1,y1,x2,y2
    b: List[List[int]] = []
    for x,y,w,h in boxes:
        b.append([x,y,x+w,y+h])
    b = sorted(b, key=lambda k: (k[1], k[0]))
    merged: List[List[int]] = []
    for bx in b:
        merged_into_existing = False
        for i, m in enumerate(merged):
            if iou(bx, m) > iou_thresh:
                merged[i] = [min(m[0], bx[0]), min(m[1], bx[1]), max(m[2], bx[2]), max(m[3], bx[3])]
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(bx)
    # back to x,y,w,h
    out: List[Tuple[int,int,int,int]] = [(x1, y1, x2-x1, y2-y1) for x1,y1,x2,y2 in merged]
    return out

def iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0

def main(input_dir: str, out_dir: str, mode: Literal["black", "blur", "pixelate", "color"]="black", color: str="#000000", pixel_size: int=12, tesseract_cmd: Optional[str]=None) -> None:
    mem_path = Path("data/memory.json")
    mem: Dict[str, Any] = json.load(open(mem_path, "r")) if mem_path.exists() else {"label_synonyms": LABEL_SYNONYMS}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.jpeg"))
    for p in paths:
        img = Image.open(p).convert("RGB")
        boxes = detect_boxes(img, mem)
        # mask
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR).astype(np.uint8)
        rgb = hex_to_rgb(color)
        for box in boxes:
            apply_mask(img_bgr, box, mode=mode, color=rgb, pixel_size=pixel_size)
        out_path = Path(out_dir) / p.name
        cv2.imwrite(str(out_path), img_bgr)
        print(f"Saved {out_path} ({len(boxes)} regions masked)")

def hex_to_rgb(hx: str) -> Tuple[int, int, int]:
    hx = hx.lstrip("#")
    if len(hx)==3:
        hx = "".join([c*2 for c in hx])
    return tuple(int(hx[i:i+2],16) for i in (0,2,4))  # type: ignore

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="black", choices=["black","blur","pixelate","color"])
    ap.add_argument("--color", default="#000000")
    ap.add_argument("--pixel_size", type=int, default=12)
    args = ap.parse_args()
    main(args.input, args.out, args.mode, args.color, args.pixel_size)
