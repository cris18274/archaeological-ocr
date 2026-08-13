
import cv2
import os
import numpy as np

def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [imread] Error: {e}")
        return None

def detect_table_morphological_diag(img_path):
    img = imread_unicode(img_path)
    if img is None: return "File not found"
    
    # 1. Red channel for best grid contrast (Cyan/Blue grid lines)
    red = img[:,:,2] # Red channel (BGR -> index 2)
    
    # 2. Adaptive Thresholding for grid detection
    binary = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    cv2.imwrite("diag_1_red_binary.jpg", binary)

    # REGLA ocr.py: Escala con Ancho de Pagina
    W_ref = red.shape[1]
    
    # Líneas Verticales
    kl_v = max(W_ref // 120, 3)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kl_v))
    v_lines = cv2.erode(binary, v_kernel, iterations=3)
    v_lines = cv2.dilate(v_lines, v_kernel, iterations=3)
    cv2.imwrite("diag_2_v_lines.jpg", v_lines)

    # Líneas Horizontales
    kl_h = max(W_ref // 40, 3)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kl_h, 1))
    h_lines = cv2.erode(binary, h_kernel, iterations=3)
    h_lines = cv2.dilate(h_lines, h_kernel, iterations=3)
    cv2.imwrite("diag_3_h_lines.jpg", h_lines)

    # Grid
    grid = cv2.addWeighted(v_lines, 0.5, h_lines, 0.5, 0.0)
    grid_inv = cv2.bitwise_not(grid)
    grid_clean = cv2.erode(grid_inv, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    _, grid_final = cv2.threshold(grid_clean, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite("diag_4_grid.jpg", grid_final)

    contours, _ = cv2.findContours(grid_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    vis = img.copy()
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 400 and 10 < h < 250:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            count += 1
            
    cv2.imwrite("diag_5_result.jpg", vis)
    return f"Done! Found {count} boxes."

if __name__ == "__main__":
    p3_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\webapp\uploads\7112bb3b-7a4b-4b96-9b18-a94065ab2dc9_page_3.jpg"
    print(detect_table_morphological_diag(p3_path))
