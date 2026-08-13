import os
import cv2
import sys
import json
import numpy as np

# ROOT
sys.path.append(os.getcwd())
from main import process_image

# GROUND TRUTH (Verdad de Campo) para affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg
# Basado en inspeccin visual minuciosa
GROUND_TRUTH = {
    "1": {"Borde decorado (+10%)": "5", "TOTAL CERAMICA": "9"},
    "2": {"Borde decorado (+10%)": "4", "TOTAL CERAMICA": "12"},
    "3": {"LITICA TALLADA - Lasca": "3", "TOTAL CERAMICA": "3"},
    "12": {"Cat-variedad": "Colonial", "TOTAL CERAMICA": "3"},
    "13": {"Cat-variedad": "Colonial Panzaleo", "TOTAL CERAMICA": "25"}
}

def run_validation():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    print(f"[*] Iniciando validacin automatizada sobre: {img_path}")
    
    # 1. Ejecutar el modelo completo
    result, steps = process_image(img_path, page_num=3)
    
    rows = result.get("table", {}).get("rows", [])
    headers = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    
    print(f"[*] Tabla extrada con {len(data_rows)} filas.")
    
    # Mapear columnas de inters
    col_map = {name: i for i, name in enumerate(headers)}
    idx_catalogo = col_map.get("CATALOGO", 0)
    
    # 2. Comparar
    stats = {"match": 0, "fail": 0, "total": 0}
    
    for row in data_rows:
        cat_val = row[idx_catalogo].strip()
        if cat_val in GROUND_TRUTH:
            expected = GROUND_TRUTH[cat_val]
            print(f"\n[Fila CATALOGO {cat_val}] Verificando...")
            
            for col_name, exp_val in expected.items():
                stats["total"] += 1
                col_idx = col_map.get(col_name)
                if col_idx is not None:
                    got_val = row[col_idx].strip()
                    # Limpieza bsica para comparacin (ignorar espacios)
                    if got_val == exp_val:
                        print(f"  OK: {col_name} -> '{got_val}'")
                        stats["match"] += 1
                    else:
                        print(f"  FAIL: {col_name} -> Esperaba '{exp_val}', Obtuve '{got_val}'")
                        stats["fail"] += 1
                else:
                    print(f"  WARN: Columna '{col_name}' no encontrada en la extraccin.")
                    stats["fail"] += 1

    # 3. Reporte Final
    print("\n" + "="*40)
    print("RESUMEN DE PRECISIN")
    print("="*40)
    if stats["total"] > 0:
        accuracy = (stats["match"] / stats["total"]) * 100
        print(f"Precision: {accuracy:.2f}% ({stats['match']}/{stats['total']})")
    else:
        print("No se encontraron filas de control en la extraccin.")
    print("="*40)
    
    return stats

if __name__ == "__main__":
    run_validation()
