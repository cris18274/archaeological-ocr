import os
import cv2
import sys
import json

sys.path.append(os.getcwd())
try:
    from main import process_image, reconstruct_table
except ImportError:
    # If main is broken, try manual import of components to test them
    pass

def test_p3_top_rows():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    
    # ROI: Top Header + First 5 Data Rows (Catalogo 1..5)
    # Based on grid: Header is at top, Catalogo 1 is around y=350..400
    # On 1654x2338 (Portrait): 
    # y=100..600 covers first few rows
    roi = img[100:800, 20:1620]
    
    print(f"Buscando tabla en ROI (Header + Top Rows)... {roi.shape}")
    # Usamos reconstruct_table directamente para ahorrar tiempo de detección de Layout
    result = reconstruct_table(roi, "p3_top_rows")
    
    rows = result.get("rows", [])
    print(f"Filas encontradas en el ROI: {len(rows)}")
    
    # GROUND TRUTH CHECK (Manual)
    for i, row in enumerate(rows):
        print(f"Row {i}: {row}")
        
    # Save the part of the table found
    with open("top_rows_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    test_p3_top_rows()
