import os
import cv2
import sys
import json
import numpy as np

def run_validation_debug():
    sys.path.append(os.getcwd())
    from main import process_image
    
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    print(f"[*] Iniciando VALIDACION de Page 3: {img_path}")
    sys.stdout.flush()
    
    # process_image RETORNA UN DICCIONARIO, no una tupla (ERROR PREVIO)
    full_result = process_image(img_path, page_num=3)
    
    # Extraer tabla de la lista de regiones
    rows = []
    for reg in full_result.get("regions", []):
        if reg.get("type") == "table":
            content = reg.get("content", {})
            header = content.get("header", [])
            data   = content.get("rows", [])
            if header or data:
                rows = [header] + data
            break
            
    if not rows:
        print("ERROR: No se detecto ninguna tabla en 'regions'.")
        print(f"Keys en result: {full_result.keys()}")
        return
        
    headers = rows[0]
    data_rows = rows[1:]
    col_map = {name.strip(): i for i, name in enumerate(headers)}
    idx_catalogo = col_map.get("CATALOGO", 0)
    
    print(f"Headers detectados ({len(headers)}): {headers[:15]}...")
    
    GROUND_TRUTH = {
        "1": {"Borde decorado (+10%)": "5", "TOTAL CERAMICA": "9"},
        "2": {"Borde decorado (+10%)": "4", "TOTAL CERAMICA": "12"},
        "3": {"LITICA TALLADA - Lasca": "3", "TOTAL CERAMICA": "3"},
        "51": {"Cat-variedad": "A Panzaleo Colonial", "TOTAL CERAMICA": "8"},
        "52": {"Cat-variedad": "Colonial", "TOTAL CERAMICA": "1"}
    }
    
    print(f"Analizando {len(data_rows)} filas de datos...")
    
    for row in data_rows:
        if len(row) <= idx_catalogo: continue
        cat_val = row[idx_catalogo].strip()
        if cat_val in GROUND_TRUTH:
            expected = GROUND_TRUTH[cat_val]
            print(f"-- Row {cat_val} --")
            for col_name, exp_val in expected.items():
                col_idx = col_map.get(col_name)
                got_val = "COL_NOT_IN_HEADERS"
                if col_idx is not None:
                     got_val = row[col_idx].strip() if col_idx < len(row) else "OUT_OF_BOUNDS"
                
                status = "PASS" if got_val == exp_val else "FAIL"
                print(f"  {col_name}: got='{got_val}', expected='{exp_val}' [{status}]")
    sys.stdout.flush()

if __name__ == "__main__":
    run_validation_debug()
