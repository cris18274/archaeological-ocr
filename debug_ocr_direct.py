import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

def imread_unicode(path: str):
    arr = np.fromfile(path, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# Configurar motor igual que en main.py (PP-OCRv4)
ocr = PaddleOCR(lang='es', use_angle_cls=True, use_gpu=True, show_log=False, ocr_version='PP-OCRv4')

img_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\uploads\cell_p1_fig0_r18_c0.jpg"

if not os.path.exists(img_path):
    print(f"ERROR: Archivo no encontrado {img_path}")
    # Buscar cualquier otro archivo .jpg en la carpeta uploads
    import glob
    files = glob.glob(r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\uploads\cell_p1_*.jpg")
    if files:
        img_path = files[0]
        print(f"Usando alternativa: {img_path}")
    else:
        exit()

img = imread_unicode(img_path)
print(f"Imagen cargada: {img.shape} desde {os.path.basename(img_path)}")

print("\n--- TEST: det=False ---")
res = ocr.ocr(img, det=False, rec=True, cls=True)
print(f"RESULT: {res}")

print("\n--- TEST: Upscale (x2) + det=False ---")
h, w = img.shape[:2]
resized = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
res2 = ocr.ocr(resized, det=False, rec=True, cls=True)
print(f"RESULT x2: {res2}")
