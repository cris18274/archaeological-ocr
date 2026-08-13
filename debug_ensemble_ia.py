import easyocr
import cv2
import numpy as np
import os
import time

def imread_unicode(path: str):
    arr = np.fromfile(path, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# 1. Cargar imagen
img_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\uploads\cell_p1_fig0_r18_c0.jpg"
if not os.path.exists(img_path):
    print("ERROR: Celda no encontrada")
    exit()
img = imread_unicode(img_path)

# 2. Inicializar EasyOCR (IA Transformers)
print("Inicializando EasyOCR (GPU=True)...")
t0 = time.time()
reader = easyocr.Reader(['es','en'], gpu=True)
print(f"EasyOCR listo en {time.time()-t0:.2f}s")

# 3. Reconocimiento EasyOCR
print("\n--- TEST EasyOCR: readtext ---")
res_easy = reader.readtext(img)
for (bbox, text, conf) in res_easy:
    print(f"EasyOCR: '{text}' (conf={conf:.3f})")

# 4. Comparar con Paddle (CPU)
from paddleocr import PaddleOCR
print("\nInicializando PaddleOCR (CPU)...")
ocr_paddle = PaddleOCR(lang='es', use_gpu=False, show_log=False)
res_paddle = ocr_paddle.ocr(img, det=False, rec=True, cls=True)
print(f"PaddleOCR: {res_paddle}")
