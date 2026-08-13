import os
import cv2
import sys
import json
import numpy as np

# ROOT
sys.path.append(os.getcwd())
from main import process_image, imwrite_unicode

GROUND_TRUTH = {
    "1": {"Borde decorado (+10%)": "5", "TOTAL CERAMICA": "9"},
    "2": {"Borde decorado (+10%)": "4", "TOTAL CERAMICA": "12"},
    "3": {"LITICA TALLADA - Lasca": "3", "TOTAL CERAMICA": "3"},
    "12": {"Cat-variedad": "Colonial", "TOTAL CERAMICA": "3"},
    "13": {"Cat-variedad": "Colonial Panzaleo", "TOTAL CERAMICA": "25"}
}

def run_validation_debug():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    print(f"[*] Iniciando VALIDACION de Page 3: {img_path}")
    
    result, steps = process_image(img_path, page_num=3)
    
    rows = result.get("table", {}).get("rows", [])
    if not rows:
        print("ERROR: No se detecto ninguna tabla.")
        return
        
    headers = rows[0]
    data_rows = rows[1:]
    
    col_map = {name.strip(): i for i, name in enumerate(headers)}
    idx_catalogo = col_map.get("CATALOGO", 0)
    
    print(f"Headers detectados: {headers[:10]}...")
    
    for row in data_rows:
        cat_val = row[idx_catalogo].strip()
        if cat_val in GROUND_TRUTH:
            expected = GROUND_TRUTH[cat_val]
            for col_name, exp_val in expected.items():
                col_idx = col_map.get(col_name)
                got_val = row[col_idx].strip() if col_idx is not None else "NOT_FOUND"
                
                status = "PASS" if got_val == exp_val else "FAIL"
                print(f"Row {cat_val} | {col_name}: got='{got_val}', expected='{exp_val}' [{status}]")

if __name__ == "__main__":
    run_validation_debug()
