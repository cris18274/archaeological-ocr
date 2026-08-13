import os
import cv2
import sys
import numpy as np

sys.path.append(os.getcwd())
from main import reconstruct_table

def test_boxes():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    
    # ROI: Just header + 1 row
    roi = img[100:600, 20:1620]
    
    # We'll use a mock 'recognize_cells_batch' to skip OCR and see boxes
    import main
    orig_ocr = main.recognize_cells_batch
    main.recognize_cells_batch = lambda img, boxes, rotate_deg=0: ["BOX"] * len(boxes)
    
    result = reconstruct_table(roi, "p3_boxes")
    
    main.recognize_cells_batch = orig_ocr # Restore
    
    rows = result.get("rows", [])
    print(f"Boxes found: {len(rows)} rows.")
    for i, row in enumerate(rows[:3]):
        print(f"Row {i} has {len(row)} cells.")

if __name__ == "__main__":
    test_boxes()
