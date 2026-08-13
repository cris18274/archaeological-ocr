import os
import cv2
import sys

# Agregar el path del backend para importar main
sys.path.append(os.getcwd())

from main import recognize_cells_batch, preprocess_cell

def test_recovery():
    img_path = r"../uploads/25f4b1e53b2890fd4012b37001b3002c_prueba1.JPG"
    if not os.path.exists(img_path):
        print(f"Error: Archivo no encontrado en {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: No se pudo cargar {img_path}")
        return
    
    h_orig, w_orig = img.shape[:2]
    print(f"Dimensiones de imagen: {w_orig}x{h_orig}")
    
    # Coordenadas reales basadas en search_area.png (892px)
    # x, y, w, h
    test_boxes = [
        [115, 105, 30, 30], # Area con un "1"
        [455, 116, 30, 30], # Area con un "3"
        [455, 158, 30, 25], # Area con un "15"
    ]
    
    print("Iniciando Verificación Híbrida (IA-Soft vs RAW)...")
    
    # Prueba con Preprocesamiento
    results_soft = recognize_cells_batch(img, test_boxes, allowlist="0123456789ABCDEFGHIJKabcdefghijk -_./")
    
    # ESTRATEGIA ULTRA-MICRO: Escalado 4x y Margen 0
    print("\nIniciando Prueba ULTRA-MICRO (Scale 4x, Margin 0)...")
    from main import easyocr_engine # Moved here to ensure it's imported before use in this block
    for i, (x, y, w, h) in enumerate(test_boxes):
        # Usamos coordenadas proporcionales si las originales eran para 4000px
        # Pero suponiendo que 413, 15 son para la de 892x728:
        crop = img[y:y+h, x:x+w]
        if crop.size == 0: continue
        
        # Super-Resolución local
        upscaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"debug_u_micro_{i}.png", upscaled)
        
        res = easyocr_engine.readtext(upscaled, detail=0)
        print(f"Celda {i} Ultra-Micro (EasyOCR):", res)
        
        from main import ocr_engine
        res_p = ocr_engine.ocr(upscaled, cls=False)
        print(f"Celda {i} Ultra-Micro (Paddle):", res_p)

    # Prueba RAW (bypass preprocess_cell)
    results_raw = []
    for (x, y, w, h) in test_boxes:
        crop = img[y:y+h, x:x+w]
        if crop.size == 0: continue
        res = easyocr_engine.readtext(crop, detail=0)
        results_raw.append(" ".join(res) if res else "")

    print("\nIniciando Prueba de Línea Base (Texto General)...")
    # En lugar de un ROI fijo, buscamos cualquier texto en la imagen para validar que el motor funciona
    res_base = easyocr_engine.readtext(img, detail=0)
    print("Texto detectado en página completa:", res_base[:10], "..." if len(res_base)>10 else "")
    
    # Intentamos un recorte más seguro para la zona inferior (donde suele estar CATALOGO)
    h, w = img.shape[:2]
    ref_crop = img[int(h*0.8):h, int(w*0.1):int(w*0.5)]
    if ref_crop.size > 0:
        cv2.imwrite("debug_baseline_area.png", ref_crop)
        res_area = easyocr_engine.readtext(ref_crop, detail=0)
        print("Texto en zona inferior (CATALOGO?):", res_area)

    print("\nIniciando Prueba Sin Filtros de Confianza...")
    for i, (x, y, w, h) in enumerate(test_boxes):
        crop = img[y:y+h, x:x+w]
        if crop.size == 0: continue
        res = easyocr_engine.readtext(crop, min_size=1, text_threshold=0.1)
        print(f"Celda {i} RAW (No Filters):", res)
        
        # Probar Paddle
        from main import ocr_engine
        res_p = ocr_engine.ocr(crop, cls=False)
        print(f"Celda {i} RAW (Paddle):", res_p)

    print("\n--- RESUMEN DE RESULTADOS ---")
    print(f"Soft (Ultra-Micro Batch): {results_soft}")
    
    if any(r.strip() for r in results_soft):
        print("\nSUCCESS: Se ha recuperado la detección de texto.")
    else:
        print("\nFAILURE: Las celdas siguen retornando vacío.")

if __name__ == "__main__":
    test_recovery()
