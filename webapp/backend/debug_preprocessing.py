import cv2
import numpy as np
import os

def preprocess_cell_debug(cell_img):
    # 1. ELIMINACIÓN DE CUADRÍCULA (Grid Removal)
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    lower_cyan = np.array([80, 20, 20])
    upper_cyan = np.array([130, 255, 255])
    lower_red1 = np.array([0, 20, 20]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 20, 20]); upper_red2 = np.array([180, 255, 255])
    
    mask_grid = cv2.inRange(hsv, lower_cyan, upper_cyan) | \
                cv2.inRange(hsv, lower_red1, upper_red1) | \
                cv2.inRange(hsv, lower_red2, upper_red2)
    
    ink_only = cv2.bitwise_and(cell_img, cell_img, mask=cv2.bitwise_not(mask_grid))
    img_clean = ink_only.copy()
    img_clean[mask_grid > 0] = [255, 255, 255]

    # 2. UPSCALING
    h, w = img_clean.shape[:2]
    target_h = 64
    if h < target_h:
        scale = target_h / h
        img_clean = cv2.resize(img_clean, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive Threshold (C=12 is the suspect)
    adaptive_c12 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    adaptive_c7 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)
    
    return img_clean, adaptive_c12, adaptive_c7

# Simular carga de celda
# Ruta relativa desde backend/
img_path = r"../uploads/25f4b1e53b2890fd4012b37001b3002c_prueba1.JPG"
if os.path.exists(img_path):
    img = cv2.imread(img_path)
    # Extraer una celda tipica (estimada)
    h, w = img.shape[:2]
    cell = img[int(h*0.5):int(h*0.5)+50, int(w*0.5):int(w*0.5)+100]
    
    clean, c12, c7 = preprocess_cell_debug(cell)
    
    cv2.imwrite("debug_cell_original.png", cell)
    cv2.imwrite("debug_cell_clean.png", clean)
    cv2.imwrite("debug_cell_c12.png", c12)
    cv2.imwrite("debug_cell_c7.png", c7)
    print("Debug images saved.")
else:
    print(f"File not found: {img_path}")
