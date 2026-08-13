import os
import sys
import json

# Path main
sys.path.append(os.getcwd())
from main import process_image

def test_full_page_3():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    print(f"Probando proceso completo en: {img_path}")
    
    result = process_image(img_path, 3)
    
    # Ver resultados de regiones
    for i, reg in enumerate(result.get("regions", [])):
        print(f"\nRegion {i} [{reg['type']}]:")
        if reg['type'] == 'table':
            content = reg.get("content", {})
            rows = content.get("rows", [])
            header = content.get("header", [])
            print(f"  Header: {header}")
            for r in rows[:5]: # Solo mostrar primeras 5 filas
                print(f"  Row: {r}")
        elif reg['type'] == 'text':
            print(f"  Text: {reg.get('content', '')}")

    # Guardar resultado en JSON para inspeccion
    with open("full_result_p3.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    test_full_page_3()
