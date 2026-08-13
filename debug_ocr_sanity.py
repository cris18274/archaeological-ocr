import cv2
import numpy as np
from paddleocr import PaddleOCR

# Crear imagen sintética: Texto Negro sobre Fondo Blanco
canvas = np.ones((100, 400, 3), dtype=np.uint8) * 255
cv2.putText(canvas, "PRUEBA OCR", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
cv2.imwrite("synthetic_test.png", canvas)

print("Iniciando motor stock (CPU)...")
ocr = PaddleOCR(use_gpu=False, show_log=False)

print("--- TEST SINTETICO: det=True ---")
res1 = ocr.ocr(canvas, det=True)
print(f"RESULT Synthetic det=True: {res1}")

print("\n--- TEST SINTETICO: det=False ---")
res2 = ocr.ocr(canvas, det=False)
print(f"RESULT Synthetic det=False: {res2}")
