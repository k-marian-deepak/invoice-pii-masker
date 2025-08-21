\
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import pytesseract
from pytesseract import Output  # type: ignore

def configure_tesseract(cmd: Optional[str] = None):
    """Configure tesseract executable path."""
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    else:
        pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def ocr_words(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Returns a list of word dicts with text and bounding boxes (x, y, w, h).
    """
    words: List[Dict[str, Any]] = []
    # Type annotation to help the type checker
    data: Dict[str, List[Any]] = pytesseract.image_to_data(image, output_type=Output.DICT)  # type: ignore
    
    for i in range(len(data["text"])):  # type: ignore
        txt = str(data["text"][i]).strip()  # type: ignore
        if not txt:
            continue
        conf = float(data["conf"][i]) if str(data["conf"][i]) != '-1' else -1.0  # type: ignore
        words.append({
            "text": txt,
            "conf": conf,
            "left": int(data["left"][i]),  # type: ignore
            "top": int(data["top"][i]),  # type: ignore
            "width": int(data["width"][i]),  # type: ignore
            "height": int(data["height"][i]),  # type: ignore
            "line_num": int(data["line_num"][i]),  # type: ignore
            "block_num": int(data["block_num"][i]),  # type: ignore
            "par_num": int(data["par_num"][i]),  # type: ignore
            "page_num": int(data["page_num"][i]),  # type: ignore
        })
    return words

def line_boxes(words: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Rough line bounding boxes aggregated by (block_num, par_num, line_num).
    Returns dict keyed by a composite int key with (x, y, w, h).
    """
    boxes: Dict[Tuple[int, int, int], List[int]] = {}
    for w in words:
        key = (w["block_num"], w["par_num"], w["line_num"])
        if key not in boxes:
            boxes[key] = [w["left"], w["top"], w["left"] + w["width"], w["top"] + w["height"]]
        else:
            x1, y1, x2, y2 = boxes[key]
            x1 = min(x1, w["left"])
            y1 = min(y1, w["top"])
            x2 = max(x2, w["left"] + w["width"])
            y2 = max(y2, w["top"] + w["height"])
            boxes[key] = [x1, y1, x2, y2]
    
    # convert to x,y,w,h
    boxes_xywh: Dict[int, Tuple[int, int, int, int]] = {}
    for i, (_, coords) in enumerate(boxes.items(), 1):
        x1, y1, x2, y2 = coords
        boxes_xywh[i] = (x1, y1, x2 - x1, y2 - y1)
    return boxes_xywh
