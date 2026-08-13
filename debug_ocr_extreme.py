import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

def imread_unicode(path: str):
    arr = np.fromfile(path, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\uploads\cell_p1_fig0_r18_c0.jpg"
img = imread_unicode(img_path)

# TEST D: PP-OCRv3 (Suele ser mas robusto en imagenes dificiles)
print("\n--- TEST D: PP-OCRv3, EN, det=False ---")
ocr_v3 = PaddleOCR(lang='en', use_gpu=False, show_log=False, ocr_version='PP-OCRv3')
res_d = ocr_v3.ocr(img, det=False, rec=True, cls=True)
print(f"RESULT D: {res_d}")

# TEST E: Mejora extrema de imagen + PP-OCRv3
print("\n--- TEST E: CLAHE + Upscale x4 + PP-OCRv3 ---")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
h, w = cl1.shape
cl1 = cv2.resize(cl1, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
# Volver a BGR para Paddle
cl1_bgr = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
res_e = ocr_v3.ocr(cl1_bgr, det=False, rec=True, cls=True)
print(f"RESULT E: {res_e}")

# TEST F: Probar si un modelo stock sin 'lang' funciona
print("\n--- TEST F: Default model, det=True ---")
ocr_def = PaddleOCR(use_gpu=False, show_log=False)
res_f = ocr_def.ocr(img, det=True)
print(f"RESULT F: {res_f}")
