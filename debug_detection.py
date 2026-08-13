import os
import sys
import cv2
import numpy as np
import pandas as pd
import io

# DLL Fixes for Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUDNN_PATH = r"C:\Program Files\SPGlobal\KingdomSuite\TKS 2023"
MKL_PATH = os.path.join(BASE_DIR, "venv", "Library", "bin")

if hasattr(os, "add_dll_directory"):
    for p in [CUDNN_PATH, MKL_PATH]:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
                print(f"DLL path added: {p}")
            except Exception as e:
                print(f"Error adding {p}: {e}")

from paddleocr import PPStructure

# Configuración idéntica al backend
print("Inicializando motor OCR (Español - PPStructureV2)...")
try:
    # Cambiamos a lang='en' porque el modelo de layout (segmentación) 
    # de PP-StructureV2 solo soporta 'en' y 'ch'.
    # El OCR interno aún podrá reconocer caracteres españoles.
    table_engine = PPStructure(
        show_log=True, 
        use_gpu=False, 
        lang='en', 
        layout=True, 
        table=True, 
        layout_score_threshold=0.3,
        structure_version='PP-StructureV2'
    )
    print("Motor inicializado con lang='en'.")
except Exception as e:
    print(f"Error al inicializar: {e}")
    exit(1)

test_images = ["page_1.jpg", "page_2.jpg", "page_3.jpg", "prueba.png"]
base_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version"

for img_name in test_images:
    img_path = os.path.join(base_path, img_name)
    print(f"\n--- Probando: {img_name} ---")
    if not os.path.exists(img_path):
        print(f"ERROR: No existe {img_path}")
        continue

    # Lectura robusta para rutas con tildes/espacios en Windows
    try:
        nparr = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"ERROR al decodificar imagen: {e}")
        continue

    if img is None:
        print(f"ERROR: No se pudo leer {img_name} (imread retornó None)")
        continue

    result = table_engine(img)
    print(f"Regiones detectadas: {len(result)}")
    
    for i, region in enumerate(result):
        rtype = region['type']
        bbox = region['bbox']
        score = region.get('score', 'N/A')
        print(f"[{i}] Tipo: {rtype}, Score: {score}, BBox: {bbox}")
        
        if rtype in ['table', 'table_caption']:
            html = region['res'].get('html')
            if html:
                print(f"   -> Tabla detectada con HTML")
            else:
                print(f"   -> Tabla detectada pero SIN HTML!")
        elif rtype == 'text':
            text = [line['text'] for line in region['res']]
            print(f"   -> Texto (primeras 2 líneas): {text[:2]}")

print("\nDebug finalizado.")
