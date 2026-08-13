import cv2
import numpy as np
import os

def locate_numbers():
    img_path = r"../uploads/25f4b1e53b2890fd4012b37001b3002c_prueba1.JPG"
    img = cv2.imread(img_path)
    if img is None: return
    
    h, w = img.shape[:2]
    # Guardamos el cuadrante superior izquierdo donde suelen estar los primeros fragmentos
    area = img[0:int(h*0.3), 0:int(w*0.7)]
    cv2.imwrite("search_area.png", area)
    print(f"Area guardada: {area.shape}")

if __name__ == "__main__":
    locate_numbers()
