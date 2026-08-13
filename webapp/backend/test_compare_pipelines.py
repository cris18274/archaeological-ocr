"""
Diagnóstico de diferencias entre ocr2.py (funciona) y main.py (falla).
Ambos procesan la misma imagen y se comparan paso a paso.
"""
import os, sys, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────── RUTAS ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMG_PATH    = os.path.normpath(os.path.join(BASE_DIR, "..", "uploads",
              "affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"))
OCR2_CSV    = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "ocr",
              "ocr2_results_p3", "results.csv"))

# ─────────── LEER IMAGEN (imdecode igual que ocr2.py) ──────────────────────
img = cv2.imdecode(np.fromfile(IMG_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
H, W = img.shape[:2]
print(f"[IMG] Tamaño de imagen: {W}x{H}")

# ─────────── REPLICAR EXACTAMENTE ocr2.py ──────────────────────────────────
print("\n=== PIPELINE ocr2.py (original) ===")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bin_img = cv2.bitwise_not(cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])

# Kernels basados en W total (imagen completa)
v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray.shape[1] // 120))
v_l = cv2.dilate(cv2.erode(bin_img, v_k, iterations=3), v_k, iterations=3)

h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1] // 40, 1))
h_l = cv2.dilate(cv2.erode(bin_img, h_k, iterations=3), h_k, iterations=3)

grid = cv2.threshold(
    cv2.erode(cv2.bitwise_not(cv2.addWeighted(v_l, 0.5, h_l, 0.5, 0)),
              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2),
    0, 255, cv2.THRESH_OTSU)[1]

cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boxes_ocr2 = [cv2.boundingRect(c) for c in cnts
              if cv2.contourArea(c) > 100 and 5 < cv2.boundingRect(c)[3] < 200]

print(f"  Kernel V: 1 x {gray.shape[1]//120}  (basado en W={gray.shape[1]})")
print(f"  Kernel H: {gray.shape[1]//40} x 1   (basado en W={gray.shape[1]})")
print(f"  Celdas detectadas por ocr2.py: {len(boxes_ocr2)}")

# ─────────── REPLICAR MAIN.py (región recortada por morfología) ────────────
print("\n=== PIPELINE main.py (deteccion morfologica -> recorte ROI) ===")

# 1. detect_table_morphological sobre canal rojo (como en main.py)
img_gray_red = img[:, :, 2]
img_bin_morph = cv2.adaptiveThreshold(img_gray_red, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

W_ref = img_gray_red.shape[1]
kl_v = max(W_ref // 120, 3)
kl_h = max(W_ref // 40, 3)

v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kl_v))
v_lines_m = cv2.dilate(cv2.erode(img_bin_morph, v_kernel, iterations=3), v_kernel, iterations=3)
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kl_h, 1))
h_lines_m = cv2.dilate(cv2.erode(img_bin_morph, h_kernel, iterations=3), h_kernel, iterations=3)

grid_m = cv2.addWeighted(v_lines_m, 0.5, h_lines_m, 0.5, 0.0)
grid_m = cv2.erode(cv2.bitwise_not(grid_m), cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)
_, grid_m = cv2.threshold(grid_m, 0, 255, cv2.THRESH_OTSU)

cnts_m, _ = cv2.findContours(grid_m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boxes_main = []
for cnt in cnts_m:
    x, y, w, h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > 400 and 10 < h < 250:
        boxes_main.append((x, y, w, h))

print(f"  Kernel V: 1 x {kl_v}  (basado en W_ref={W_ref})")
print(f"  Kernel H: {kl_h} x 1   (basado en W_ref={W_ref})")
print(f"  Celdas detectadas por main.py (morfología): {len(boxes_main)}")

# ─────────── Calcular BBOX de la región detectada ─────────────────────────
if boxes_main:
    x1 = min(b[0] for b in boxes_main)
    y1 = min(b[1] for b in boxes_main)
    x2 = max(b[0]+b[2] for b in boxes_main)
    y2 = max(b[1]+b[3] for b in boxes_main)
    roi_w = x2 - x1
    roi_h = y2 - y1
    print(f"\n  ROI región tabla: ({x1},{y1}) -> ({x2},{y2}) = {roi_w}x{roi_h} px")
    print(f"  ⚠️  La ROI representa {roi_w/W*100:.1f}% del ancho y {roi_h/H*100:.1f}% del alto")

    # ─────────── PROBLEMA CLAVE: Kernel en ROI vs Imagen Completa ──────────
    print(f"\n=== ANÁLISIS: Kernels cuando reconstruct_table procesa solo la ROI ===")
    # main.py USA H,W DE LA ROI (no de la imagen completa) pero con kl_v=max(W//120, 3)
    roi_struct = img[y1:y2, x1:x2]
    W_roi = roi_struct.shape[1]
    H_roi = roi_struct.shape[0]
    kl_v_roi = max(W_roi // 120, 3)
    kl_h_roi = max(W_roi // 40, 3)
    print(f"  ROI W={W_roi}, H={H_roi}")
    print(f"  Kernel V en ROI: 1 x {kl_v_roi}  (W_roi={W_roi} // 120 = {W_roi//120})")
    print(f"  Kernel H en ROI: {kl_h_roi} x 1   (W_roi={W_roi} // 40 = {W_roi//40})")
    print(f"  Kernel V en FULL: 1 x {W//120}   (W_full={W} // 120 = {W//120})")
    print(f"  Kernel H en FULL: {W//40} x 1    (W_full={W} // 40 = {W//40})")
    print(f"\n  *** DIFERENCIA CRÍTICA: Kernel H = {kl_h_roi} (ROI) vs {W//40} (FULL) ***")
    print(f"  *** Kernel V  = {kl_v_roi} (ROI) vs {W//120} (FULL) ***")

    # ─────────── APLICAR KERNELS CORRECTOS (W_full) A LA ROI ──────────────
    print(f"\n=== PRUEBA: Usando Kernels de FULL en ROI (correccion) ===")
    gray_roi = cv2.cvtColor(roi_struct, cv2.COLOR_BGR2GRAY)
    bin_roi = cv2.bitwise_not(cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])

    # Kernels con el ancho de la imagen COMPLETA (como ocr2.py)
    kl_v_fix = W // 120
    kl_h_fix = W // 40
    v_k_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kl_v_fix))
    v_l_fix = cv2.dilate(cv2.erode(bin_roi, v_k_fix, iterations=3), v_k_fix, iterations=3)
    h_k_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (kl_h_fix, 1))
    h_l_fix = cv2.dilate(cv2.erode(bin_roi, h_k_fix, iterations=3), h_k_fix, iterations=3)

    grid_fix = cv2.threshold(
        cv2.erode(cv2.bitwise_not(cv2.addWeighted(v_l_fix, 0.5, h_l_fix, 0.5, 0)),
                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2),
        0, 255, cv2.THRESH_OTSU)[1]

    cnts_fix, _ = cv2.findContours(grid_fix, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes_fix = [cv2.boundingRect(c) for c in cnts_fix
                 if cv2.contourArea(c) > 100 and 5 < cv2.boundingRect(c)[3] < 200]
    print(f"  Celdas con kernels corregidos (W_full) sobre ROI: {len(boxes_fix)}")

# ─────────── RESUMEN ────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║  RESUMEN DIAGNÓSTICO                                  ║
╠══════════════════════════════════════════════════════╣
║  ocr2.py: imagen completa + kernels W={W//120}/{W//40}     
║  Celdas: {len(boxes_ocr2):>5}                                     
╠══════════════════════════════════════════════════════╣
║  main.py: ROI recortada + kernels W_roi             
║  Celdas: {len(boxes_main):>5}                                     
╠══════════════════════════════════════════════════════╣
║  Corrección: ROI + kernels W_full                   
║  Celdas: {len(boxes_fix) if boxes_main else 0:>5}                                     
╚══════════════════════════════════════════════════════╝
""")
print("SOLUCIÓN: Pasar siempre el W de la imagen COMPLETA a reconstruct_table (full_width=W)")
print("         Y usar Otsu+Bitwise_not (ocr2.py) en lugar de AdaptiveThreshold+Canal Rojo (main.py)")
