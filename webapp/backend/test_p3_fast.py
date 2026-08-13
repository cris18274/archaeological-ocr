import os
import cv2
import sys
import json

sys.path.append(os.getcwd())
from main import (
    reconstruct_table, 
    preprocess_page_forensic,
    easyocr_engine,
    ocr_engine
)

def test_page3_fast():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    print(f"Fast Test Page 3: {img.shape}")

    # Saltamos la detección de Layout y procesamos el ROI de la tabla directamente
    # La tabla ocupa casi toda la página en Portrait
    # x,y,w,h = 20, 40, 1600, 2200
    roi = img[40:2240, 20:1620]
    
    print("Reconstruyendo tabla directamente sobre ROI...")
    table_data = reconstruct_table(roi, "page3_fast")
    
    # Comprobación de Ground Truth
    rows = table_data.get("rows", [])
    print(f"Filas extraídas: {len(rows)}")
    
    # Muestra de validación (Filas 1 y 2)
    if len(rows) > 1:
        print(f"Fila 1: {rows[0]}")
        print(f"Fila 2: {rows[1]}")
        
    with open("fast_result_p3.json", "w", encoding="utf-8") as f:
        json.dump(table_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    test_page3_fast()
