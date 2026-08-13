
import os, sys, cv2, json
import numpy as np

# Mocking parts of the backend to test the logic
sys.path.append(r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\backend")
from main import process_image

# Usamos uno de los archivos que SI existe segun list_dir
img_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\uploads\7112bb3b-7a4b-4b96-9b18-a94065ab2dc9_page_3.jpg"

if not os.path.exists(img_path):
    print(f"Error: No se encuentra {img_path}")
    sys.exit(1)

print(f"--- Iniciando Prueba de Página 3 ({os.path.basename(img_path)}) ---")
print("--- IA Ensemble v3 + 8 Reglas Críticas ---")

result = process_image(img_path, 3)

# Mostrar resumen de la tabla extraída
if "table" in result:
    table = result["table"]
    print(f"\n[OK] Resultado de la Tabla ({len(table)} filas):")
    for i, row in enumerate(table):
        if i < 5 or i > len(table) - 3: # Mostrar solo cabecera y final para log
            print(f"Fila {i}: {row}")
        elif i == 5:
            print("...")
    
    # Comprobar si keywords esperados están presentes
    flat_text = " ".join([" ".join(row) for row in table]).upper()
    keywords = ["CATALOGO", "VARIEDAD", "TIESTO", "CUERPO", "TOTAL"]
    found = [kw for kw in keywords if kw in flat_text]
    print(f"\nKeywords detectados: {found}")
    
    # Guardar para inspección
    output_f = os.path.join(os.path.dirname(__file__), "debug_p3_result_v5.json")
    with open(output_f, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Detalle completo guardado en {output_f}")
else:
    print(f"[ERROR] En procesamiento: {result.get('error')}")
    if "steps" in result:
         print(f"Pasos generados: {result['steps']}")
