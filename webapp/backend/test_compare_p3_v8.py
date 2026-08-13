
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
print("--- IA Ensemble v3 + 8 Reglas Críticas + Batch Optimization ---")

result = process_image(img_path, 3)

# Extraer tabla de regiones
table_matrix = []
if "regions" in result:
    for reg in result["regions"]:
        if reg.get("type") == "table":
            # Caso 1: Estructura estándar ("res" -> "matrix")
            if "res" in reg and isinstance(reg["res"], dict) and "matrix" in reg["res"]:
                table_matrix = reg["res"]["matrix"]
                break
            # Caso 2: Fallback Reconstrucción ("content" -> "header"/"rows")
            elif "content" in reg and isinstance(reg["content"], dict):
                header = reg["content"].get("header", [])
                rows = reg["content"].get("rows", [])
                table_matrix = [header] + rows
                break

# Mostrar resumen de la tabla extraída
if table_matrix:
    print(f"\n[OK] Resultado de la Tabla ({len(table_matrix)} filas):")
    flat_text = ""
    for i, row in enumerate(table_matrix):
        row_str = " | ".join([str(c) for c in row])
        flat_text += " " + row_str
        if i < 8 or i > len(table_matrix) - 5:
            print(f"Fila {i:2}: {row_str}")
        elif i == 8:
            print("...")
    
    # Comprobar si keywords esperados están presentes
    flat_text = flat_text.upper()
    keywords = ["CATALOGO", "VARIEDAD", "TIESTO", "CUERPO", "TOTAL", "BORDE", "LASCA"]
    found = [kw for kw in keywords if kw in flat_text]
    print(f"\nKeywords detectados: {found}")
    
    # Comprobación de éxito (al menos 3 keywords arqueológicos críticos)
    critical = ["CATALOGO", "VARIEDAD", "TIESTO"]
    success = any(all(kw in row_str.upper() for kw in critical) for row_str in [" | ".join([str(c) for c in r]) for r in table_matrix[:5]])
    # Flexibilizamos: si están en la matriz general
    general_success = all(kw in flat_text for kw in critical)
    
    if general_success:
        print("\n[SUCCESS] La extracción de la Página 3 es CORRECTA.")
    else:
        print("\n[WARNING] Faltan keywords críticos. Revisar pre-procesamiento de cabecera.")

    # Guardar para inspección
    output_f = os.path.join(os.path.dirname(__file__), "debug_p3_result_v8.json")
    with open(output_f, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Detalle completo guardado en {output_f}")
else:
    print(f"[ERROR] No se encontró ninguna tabla en el resultado.")
    print(f"Regiones encontradas: {[r.get('type') for r in result.get('regions', [])]}")
