import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

def imread_unicode(path: str):
    arr = np.fromfile(path, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\uploads\cell_p1_fig0_r18_c0.jpg"
img = imread_unicode(img_path)

# TEST A: Modo Actual (ES, det=False, cls=True)
print("\n--- TEST A: ES, det=False, cls=True ---")
ocr_es = PaddleOCR(lang='es', use_angle_cls=True, use_gpu=False, show_log=False)
res_a = ocr_es.ocr(img, det=False, rec=True, cls=True)
print(f"RESULT A: {res_a}")

# TEST B: Modo Ingles (EN, det=True)
print("\n--- TEST B: EN, det=True, cls=True ---")
ocr_en = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False, show_log=False)
res_b = ocr_en.ocr(img, det=True, rec=True, cls=True)
print(f"RESULT B: {res_b}")

# TEST C: Rotacion Manual 180 + EN
print("\n--- TEST C: Rot 180 + EN, det=False ---")
rot = cv2.rotate(img, cv2.ROTATE_180)
res_c = ocr_en.ocr(rot, det=False, rec=True, cls=True)
print(f"RESULT C: {res_c}")
