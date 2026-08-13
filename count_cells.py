import os
import cv2
import numpy as np
import json

def imread_utf8(path):
    with open(path, "rb") as f:
        bytes = bytearray(f.read())
    nparray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(nparray, cv2.IMREAD_COLOR)

def find_cells_in_image(image_path):
    img = imread_utf8(image_path)
    if img is None: 
        print(f"No se pudo decodificar {image_path}")
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel_len_v = gray.shape[0] // 30
    kernel_len_h = gray.shape[1] // 30
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_v))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h, 1))
    
    v_lines = cv2.erode(thresh, v_kernel, iterations=3)
    v_lines = cv2.dilate(v_lines, v_kernel, iterations=3)
    h_lines = cv2.erode(thresh, h_kernel, iterations=3)
    h_lines = cv2.dilate(h_lines, h_kernel, iterations=3)
    
    table_mask = cv2.addWeighted(v_lines, 0.5, h_lines, 0.5, 0.0)
    _, table_mask = cv2.threshold(table_mask, 128, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20 and w < gray.shape[1]*0.9 and h < gray.shape[0]*0.9:
            cells.append((x, y, w, h))
            
    return cells

def main():
    pages = [
        r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\page_1.jpg",
        r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\page_2.jpg",
        r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\page_3.jpg"
    ]
    
    total_cells = 0
    page_stats = {}
    for p in pages:
        if os.path.exists(p):
            cells = find_cells_in_image(p)
            total_cells += len(cells)
            page_stats[os.path.basename(p)] = len(cells)
            print(f"{os.path.basename(p)} -> {len(cells)} cells detected")
        else:
            print(f"Not found: {p}")
            
    print(f"Total test cells detected: {total_cells}")
    
    with open("test_dataset_stats.json", "w") as f:
        json.dump({"total": total_cells, "pages": page_stats}, f)

if __name__ == '__main__':
    main()
