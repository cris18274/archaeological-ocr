import os
import cv2
import sys
import numpy as np

# Root
sys.path.append(os.getcwd())
from main import preprocess_cell, recognize_cells_batch, easyocr_engine

def manual_test_page3():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    img = cv2.imread(img_path)
    h_orig, w_orig = img.shape[:2]
    print(f"Testing Page 3 Ground Truth: {w_orig}x{h_orig}")

    # Definimos ROIs basados en la imagen visualizada (Pag 1 de 356)
    # x, y, w, h (Aproximados para 1654x2338)
    ROIs = {
        "header_bor": [100, 100, 200, 600], # Area de cabeceras verticales
        "row1_num5": [235, 395, 45, 35],    # El "5" en la fila 1
        "row2_num4": [235, 430, 45, 35],    # El "4" en la fila 2
        "row12_colonial": [140, 875, 75, 35], # Texto "Colonial"
        "lasca_3": [715, 470, 45, 35],      # El "3" en columna Lasca (Roja)
    }

    print("\n--- Resultados de OCR en ROIs de Control ---")
    for name, box in ROIs.items():
        x, y, w, h = box
        crop = img[y:y+h, x:x+w]
        
        # 1. OCR RAW
        res_raw = easyocr_engine.readtext(crop, detail=0)
        
        # 2. OCR Preprocessed (Grid Removal)
        prep = preprocess_cell(crop)
        cv2.imwrite(f"gt_prep_{name}.png", prep)
        res_prep = easyocr_engine.readtext(prep, detail=0)
        
        # 3. Rotation Test (for vertical text)
        r90 = cv2.rotate(prep, cv2.ROTATE_90_CLOCKWISE)
        res_r90 = easyocr_engine.readtext(r90, detail=0)
        
        print(f"ROI '{name}':")
        print(f"  RAW:  {res_raw}")
        print(f"  PREP: {res_prep}")
        print(f"  R90:  {res_r90}")

if __name__ == "__main__":
    manual_test_page3()
