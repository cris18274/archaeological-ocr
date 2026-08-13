import cv2
import numpy as np
import logging
from paddleocr import PaddleOCR
import os

logging.getLogger('ppocr').setLevel(logging.ERROR)
print('Inicializando OCR en test...')
ocr = PaddleOCR(use_angle_cls=True, lang='es', use_gpu=True, show_log=False)

# Crear imagen dummy con texto para asegurar que no es la imagen
img_dummy = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(img_dummy, "TEXTO PRUEBA", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

print("\n--- PRUEBA 1: NUMPY ARRAY DIRECTO ---")
try:
    res1 = ocr.ocr(img_dummy, cls=True)
    print("Resultado:", res1)
except Exception as e:
    print("Error:", e)

print("\n--- PRUEBA 2: ARCHIVO DESDE DISCO ---")
cv2.imwrite("test_pad.jpg", img_dummy)
try:
    res2 = ocr.ocr("test_pad.jpg", cls=True)
    print("Resultado:", res2)
except Exception as e:
    print("Error:", e)
    
print("\n--- PRUEBA 3: NUMPY GRISES ---")
img_gray = cv2.cvtColor(img_dummy, cv2.COLOR_BGR2GRAY)
img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
try:
    res3 = ocr.ocr(img_bgr, cls=True)
    print("Resultado:", res3)
except Exception as e:
    print("Error:", e)

print("\nHecho.")
