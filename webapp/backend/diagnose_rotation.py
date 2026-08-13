import os
import cv2
import sys
from paddleocr import PaddleOCR

def diagnose():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    small = cv2.resize(img, (0,0), fx=0.4, fy=0.4)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='es', use_gpu=False, show_log=False)
    # Paddle CLS solo detecta 0 o 180 si no hay layout.
    # Pero si el texto está a 90, ¿qué reporta? 
    # Usualmente los cuadros de texto (det=True) tienen una orientación.
    
    # Probamos con det=True
    result = ocr.ocr(small, cls=True, det=True, rec=True)
    
    if result and result[0]:
        for i, line in enumerate(result[0]):
            box = line[0]
            txt = line[1][0]
            conf = line[1][1]
            # Extraer ángulo del clasificador de ángulo si existe
            # result[0] format: [ [box], (text, conf) ] 
            # Si cls=True, Paddle corre el clasificador de ángulo antes del reconocimiento
            print(f"Line {i}: '{txt}' (conf {conf:.2f})")
    else:
        print("No text detected for orientation.")

if __name__ == "__main__":
    diagnose()
