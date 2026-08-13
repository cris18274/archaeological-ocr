import cv2
import os
import sys
import numpy as np

# Ajustar path para importar main
sys.path.append(os.getcwd())
from main import recognize_cells_batch, easyocr_engine

def test_page_3():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Imagen no encontrada")
        return
    
    h, w = img.shape[:2]
    print(f"Dimensiones de Pagina 3: {w}x{h}")
    
    # Vamos a probar con unas cabeceras y unos datos
    # En search_area vimos que estaban rotadas.
    # En una pagina completa normal, probaremos ROIs genericos
    ROIs = [
        [100, 100, 200, 50],  # Cabecera
        [100, 200, 50, 30],   # Dato 1
        [100, 250, 50, 30],   # Dato 2
    ]
    
    print("\n--- Ejecutando OCR en ROIs de Prueba ---")
    results = recognize_cells_batch(img, ROIs)
    
    for i, res in enumerate(results):
        print(f"ROI {i}: '{res}'")

    # Guardar una seccion para ver si la cuadricula molesta
    cv2.imwrite("debug_p3_samples.png", img[0:500, 0:500])

if __name__ == "__main__":
    test_page_3()
