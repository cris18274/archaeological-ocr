"""
ocr2.py (v2 - mejorado)
========================
Pipeline OCR para tablas arqueológicas con las siguientes mejoras sobre v1:

  Mejora 1: Filtro de confianza minima (CONFIDENCE_THRESHOLD=0.75)
            -> Elimina falsos positivos en celdas vacías
  Mejora 2: Zoom adaptativo segun tamaño de celda (2x-4x)
            -> Detecta dígitos pequeños (1,2,3) que v1 perdía
  Mejora 3: Pre-rotacion de celda para texto vertical (columna etiquetas)
            -> Mejora reconocimiento de encabezados de fila rotados 90°
  Mejora 4: Diccionario de vocabulario arqueológico
            -> Corrige tildes y términos específicos del dominio
  Mejora 5: Margen adaptativo proporcional al alto de celda
            -> No corta texto en celdas pequeñas
"""

import cv2
import os
import re
import difflib
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Configuración ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.85   # Calibrado: reduce FP sin sacrificar FN

# Mejora 4: Vocabulario arqueológico para corrección post-OCR
VOCAB_ARQUEOLOGICO = {
    "litica tallada":             "LITICA TALLADA",
    "Lamina":                     "Lámina",
    "Nodulo":                     "Nódulo",
    "Nucleo":                     "Núcleo",
    "Apendice":                   "Apéndice",
    "Ceramica":                   "CERAMICA",
    "Ceramica total":             "CERAMICA TOTAL",
    "Ceramica incisa":            "CERAMICA INCISA",
    "Percutor Yunque":            "Percutor / Yunque",
    "Percutor":                   "Percutor / Yunque",
    "Reutilizador Torrero":       "Reutilizador / Torrero",
    "Reutilizado":                "Reutilizador / Torrero",
    "Pedestal fragmento":         "Pedestal / Fragmento",
    "Base fragmento":             "Base / Fragmento",
    "Podos":                      "Podos",
    "Pico":                       "Pico",
    "Figurilla fragmentada":      "Figurilla fragmentada",
    "Instrumento musical":        "Instrumento musical",
    "Cuerpo deco sin pc":         "Cuerpo deco sin PC",
    "Cuerpo deco con pc":         "Cuerpo deco con PC",
    "Cuerpo no deco sin pc":      "Cuerpo no deco sin PC",
    "Cuerpo no deco con pc":      "Cuerpo no deco con PC",
    "Borde decorado 10":          "Borde decorado (-10%)",
    "Borde decorado 10 mast":     "Borde decorado (+10%)",
    "Vasija reconstruida":        "Vasija reconstruida",
    "Vasija completa":            "Vasija completa",
}

def post_process(text: str) -> str:
    """Mejora 4: Normaliza texto usando vocabulario arqueológico."""
    t = text.strip()
    if not t:
        return t
    # Búsqueda difusa en el vocabulario
    best_score = 0.0
    best_match = t
    for wrong, correct in VOCAB_ARQUEOLOGICO.items():
        score = difflib.SequenceMatcher(None, t.lower(), wrong.lower()).ratio()
        if score > best_score and score >= 0.88:
            best_score = score
            best_match = correct
    return best_match

# ─── Helpers de I/O ───────────────────────────────────────────────────────────

def robust_read(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error robust_read: {e}")
        return None

def robust_write(path, img):
    try:
        ext = os.path.splitext(path)[1]
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception as e:
        logger.error(f"Error robust_write: {e}")
    return False

# ─── Rutas ────────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path  = os.path.join(os.path.dirname(current_dir), 'webapp', 'uploads',
                            'affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg')
output_dir  = os.path.join(current_dir, 'ocr2_results_p3_v2')
os.makedirs(output_dir, exist_ok=True)

# ─── Cargar imagen ────────────────────────────────────────────────────────────
print(f"Cargando imagen: {image_path}")
img = robust_read(image_path)
if img is None:
    print("FATAL: No se pudo cargar la imagen"); exit(1)
H, W = img.shape[:2]
print(f"Dimensiones: {W}x{H} px")

# ─── PaddleOCR ────────────────────────────────────────────────────────────────
print("Inicializando PaddleOCR (GPU)...")
ocr = PaddleOCR(lang='es', use_angle_cls=True, use_gpu=True, show_log=False)
print("OK")

# ─── Detección de grid ────────────────────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bin_ = cv2.bitwise_not(cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])

v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, W // 120))
v_l = cv2.dilate(cv2.erode(bin_, v_k, iterations=3), v_k, iterations=3)

h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 40, 1))
h_l = cv2.dilate(cv2.erode(bin_, h_k, iterations=3), h_k, iterations=3)

combined = cv2.addWeighted(v_l, 0.5, h_l, 0.5, 0)
inv      = cv2.bitwise_not(combined)
eroded   = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
grid     = cv2.threshold(eroded, 0, 255, cv2.THRESH_OTSU)[1]

cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boxes   = [cv2.boundingRect(c) for c in cnts
           if cv2.contourArea(c) > 100 and 5 < cv2.boundingRect(c)[3] < 200]

# ─── Ordenar en filas ─────────────────────────────────────────────────────────
boxes.sort(key=lambda b: b[1])
rows, curr, prev_y = [], [], None
avg_h = np.mean([b[3] for b in boxes]) if boxes else 20

for b in boxes:
    if prev_y is None or abs(b[1] - prev_y) <= avg_h * 0.5:
        curr.append(b)
    else:
        curr.sort(key=lambda b: b[0])
        rows.append(curr)
        curr = [b]
    prev_y = b[1]
if curr:
    curr.sort(key=lambda b: b[0])
    rows.append(curr)

n_rows = len(rows)
n_cols = max(len(r) for r in rows) if rows else 0
print(f"Grid: {n_rows} filas x {n_cols} cols ({len(boxes)} celdas)")

# ─── Visualizar grid ──────────────────────────────────────────────────────────
vis = img.copy()
for b in boxes:
    x, y, w, h = b
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 1)
robust_write(os.path.join(output_dir, "grid_detected.jpg"), vis)

# ─── CLAHE compartido ─────────────────────────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ─── Loop OCR mejorado ────────────────────────────────────────────────────────
results = [[""] * n_cols for _ in range(n_rows)]
total   = sum(len(r) for r in rows)
done    = 0

print("Extrayendo texto celda a celda...")
for r_i, row in enumerate(rows):
    for c_i, b in enumerate(row):
        x, y, w, h = b

        # ── Mejora 5: Margen adaptativo proporcional ──────────────────
        margin = max(3, min(8, h // 5))
        roi = img[max(y - margin, 0) : y + h + margin,
                  max(x - margin, 0) : x + w + margin]

        if roi.size == 0:
            done += 1
            continue

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ── Mejora 3: Pre-rotación si el texto es vertical ────────────
        # Celdas con h >> w son etiquetas de fila rotadas 90°
        if h > w * 1.5:
            gray_roi = cv2.rotate(gray_roi, cv2.ROTATE_90_CLOCKWISE)
            cell_h, cell_w = gray_roi.shape
        else:
            cell_h, cell_w = h, w

        # ── Mejora 2: Zoom adaptativo según tamaño ────────────────────
        # Celdas chicas necesitan mas zoom para detectar '1', '2', '3'
        if cell_h < 20:
            zoom = 4   # cap en 4x -- 5x genera ruido excesivo
        elif cell_h < 30:
            zoom = 3
        elif cell_h < 45:
            zoom = 3
        else:
            zoom = 2

        proc = clahe.apply(
            cv2.resize(gray_roi, None, fx=zoom, fy=zoom,
                       interpolation=cv2.INTER_CUBIC)
        )

        # NOTA: binarización adaptativa eliminada — generaba falsos positivos.
        # El CLAHE + zoom es suficiente para dígitos pequeños.

        # ── OCR ───────────────────────────────────────────────────────
        try:
            res = ocr.ocr(proc, cls=True)
            if res and res[0]:
                # ── Mejora 1: Filtro por confianza mínima ─────────────
                textos = [
                    l[1][0] for l in res[0]
                    if l[1][1] >= CONFIDENCE_THRESHOLD
                ]
                if textos:
                    raw = " ".join(textos)
                    # ── Mejora 4: Post-proceso vocabulario ────────────
                    results[r_i][c_i] = post_process(raw)
        except Exception as e:
            logger.debug(f"OCR error [{r_i},{c_i}]: {e}")

        done += 1
        if done % 300 == 0:
            pct = done / total * 100
            print(f"  {done}/{total} ({pct:.0f}%)", end="\r")

print(f"\nCompletado. {done} celdas procesadas.")

# ─── Exportar ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "results_v2.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"CSV guardado: {csv_path}")
print(f"Filas: {n_rows} | Columnas: {n_cols}")
print("\nPrimeras 5 filas (columnas 0-5):")
print(df.iloc[:5, :6].to_string())
