\
from typing import Tuple, Literal
import cv2
import numpy as np
from numpy.typing import NDArray

Mode = Literal["black", "blur", "pixelate", "color"]

def clamp_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def apply_mask(img_bgr: NDArray[np.uint8], box: Tuple[int,int,int,int], mode: Mode = "black", color: Tuple[int,int,int] = (0,0,0), pixel_size: int = 12) -> None:
    x, y, w, h = box
    H, W = img_bgr.shape[:2]
    x, y, w, h = clamp_box(x, y, w, h, W, H)
    roi = img_bgr[y:y+h, x:x+w]

    if mode == "black":
        img_bgr[y:y+h, x:x+w] = (0, 0, 0)
    elif mode == "color":
        img_bgr[y:y+h, x:x+w] = color[::-1]  # input as RGB, OpenCV is BGR
    elif mode == "blur":
        # kernel size must be odd
        kx = max(3, (w//10)*2 + 1)
        ky = max(3, (h//10)*2 + 1)
        img_bgr[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (kx, ky), 0)
    elif mode == "pixelate":
        ps = max(4, int(pixel_size))
        small = cv2.resize(roi, (max(1, w//ps), max(1, h//ps)), interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        img_bgr[y:y+h, x:x+w] = pix
    else:
        img_bgr[y:y+h, x:x+w] = (0, 0, 0)
