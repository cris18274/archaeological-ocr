import os
import sys
import json
import pandas as pd
import time

# Agregar directorio actual al path para importar main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import process_image, get_paddle_ocr
except ImportError as e:
    print(f"Error importando main: {e}")
    exit(1)

# Imagen de prueba (Página 3)
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.normpath(os.path.join(current_dir, "..", "uploads", "affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"))

if not os.path.exists(image_path):
    print(f"ERROR: No se encontró la imagen en {image_path}")
    exit(1)

print(f"[*] Iniciando prueba de integración final sobre {image_path}...")
print("[*] Esto usará la lógica de ocr2.py integrada en main.py (GPU)...")

# Ejecutar proceso
start_time = time.time()
result = process_image(image_path, 3)
end_time = time.time()

print(f"[*] Proceso completado en {end_time - start_time:.2f} segundos.")

# Guardar resultado para inspección
output_json = os.path.join(current_dir, "integration_test_p3_result.json")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"[+] Resultado guardado en {output_json}")

# Generar una vista previa de la primera tabla detectada
for i, region in enumerate(result.get("regions", [])):
    if region["type"] == "table" and region.get("content"):
        print(f"\n[TABLA {i+1}] Detectada:")
        content = region["content"]
        header = content.get("header", [])
        rows = content.get("rows", [])
        
        # Mostrar primeras 10 filas para verificar
        df = pd.DataFrame(rows, columns=header if header else None)
        print(df.head(15).to_string())
        
        # Guardar a CSV para que el usuario pueda descargarlo si quiere
        csv_path = os.path.join(current_dir, f"integration_test_p3_table_{i+1}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[+] Tabla {i+1} exportada a {csv_path}")
