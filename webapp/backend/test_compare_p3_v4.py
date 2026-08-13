
import os, sys, cv2, json
import numpy as np

# Mocking parts of the backend to test the logic
sys.path.append(r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\backend")
from main import process_image

# Path to the image
img_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\uploads\pagina_3.jpg"

if not os.path.exists(img_path):
    print(f"Error: No se encuentra {img_path}")
    # Intentar buscar en el directorio actual o backend/uploads
    possible_paths = [
        r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\backend\uploads\pagina_3.jpg",
        os.path.join(os.getcwd(), "uploads", "pagina_3.jpg")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            img_path = p
            print(f"Encontrado en fallback: {p}")
            break
    else:
        sys.exit(1)

print("--- Iniciando Prueba de Página 3 con IA Ensemble v3 y 8 Reglas ---")
result = process_image(img_path, 3)

# Mostrar resumen de la tabla extraída
if "table" in result:
    table = result["table"]
    print(f"\nResultado de la Tabla ({len(table)} filas):")
    for i, row in enumerate(table):
        print(f"Fila {i}: {row}")
    
    # Comprobar si keywords esperados están presentes
    flat_text = " ".join([" ".join(row) for row in table]).upper()
    keywords = ["CATAL", "VARIEDAD", "TIESTO", "CUERPO", "TOTAL"]
    found = [kw for kw in keywords if kw in flat_text]
    print(f"\nKeywords detectados: {found}")
    
    # Guardar para inspección
    output_f = os.path.join(os.path.dirname(__file__), "debug_p3_result_v4.json")
    with open(output_f, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nDetalle completo guardado en {output_f}")
else:
    print(f"Error en procesamiento: {result.get('error')}")
    if "steps" in result:
         print(f"Pasos generados: {result['steps']}")
