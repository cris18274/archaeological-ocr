"""
analyze_and_improve.py
======================
1) Carga el ground truth visual (ground_truth_page_3.json)
2) Corre el modelo OCR estándar sobre la imagen (pipeline ocr2.py)
3) Compara celda a celda y genera:
   - Reporte de errores por tipo
   - Sugerencias de mejoras al pipeline
   - Guarda análisis en analyze_results/
"""

import cv2, os, sys, json, time, difflib, re
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import logging

logging.basicConfig(level=logging.WARNING)

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG  = os.path.join(ROOT, 'webapp', 'uploads',
                    'affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg')
GT   = os.path.join(ROOT, 'webapp', 'backend', 'ground_truth_page_3.json')
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_results')
os.makedirs(OUT, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def robust_read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def detect_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_ = cv2.bitwise_not(
        cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    W = gray.shape[1]
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, W // 120))
    v_l = cv2.dilate(cv2.erode(bin_, v_k, iterations=3), v_k, iterations=3)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 40, 1))
    h_l = cv2.dilate(cv2.erode(bin_, h_k, iterations=3), h_k, iterations=3)
    combined = cv2.addWeighted(v_l, 0.5, h_l, 0.5, 0)
    inv = cv2.bitwise_not(combined)
    eroded = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
    grid = cv2.threshold(eroded, 0, 255, cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cnts
            if cv2.contourArea(c) > 100 and 5 < cv2.boundingRect(c)[3] < 200]

def sort_to_rows(boxes):
    boxes.sort(key=lambda x: x[1])
    rows, curr = [], []
    avg_h  = np.mean([b[3] for b in boxes]) if boxes else 20
    prev_y = None
    for b in boxes:
        if prev_y is None or abs(b[1]-prev_y) <= avg_h*0.5:
            curr.append(b)
        else:
            curr.sort(key=lambda x: x[0])
            rows.append(curr)
            curr = [b]
        prev_y = b[1]
    if curr:
        curr.sort(key=lambda x: x[0])
        rows.append(curr)
    return rows

def run_ocr(img, rows, ocr):
    num_cols = max(len(r) for r in rows) if rows else 0
    matrix   = [[""] * num_cols for _ in range(len(rows))]
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    total = sum(len(r) for r in rows); done = 0; t0 = time.time()
    for r_i, row in enumerate(rows):
        for c_i, b in enumerate(row):
            x, y, w, h = b
            roi = img[max(y-5,0):y+h+5, max(x-5,0):x+w+5]
            if roi.size > 0:
                g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                p = clahe.apply(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
                try:
                    res = ocr.ocr(p, cls=True)
                    if res and res[0]:
                        matrix[r_i][c_i] = " ".join(l[1][0] for l in res[0])
                except Exception:
                    pass
            done += 1
            if done % 300 == 0:
                pct = done/total*100; e = time.time()-t0
                print(f"  OCR {done}/{total} ({pct:.0f}%)  ETA:{(e/done)*(total-done):.0f}s", end="\r")
    print(f"\n  Completado en {time.time()-t0:.1f}s")
    return matrix

# ─── Clasificador de errores ──────────────────────────────────────────────────

def classify_error(gt_val, ocr_val):
    gt  = str(gt_val).strip()
    ocr = str(ocr_val).strip()
    if gt == ocr:
        return "EXACT_MATCH"
    if gt == "" and ocr != "":
        return "FALSE_POSITIVE"     # OCR inventó contenido
    if gt != "" and ocr == "":
        return "FALSE_NEGATIVE"     # OCR no detectó nada
    # Ambos tienen contenido pero difieren
    sim = difflib.SequenceMatcher(None, gt, ocr).ratio()
    # ¿Es error numérico?
    if re.match(r"^\d+$", gt) and re.match(r"^\d+$", ocr):
        return "DIGIT_CONFUSION"
    # ¿Es error solo de tilde/acento?
    if gt.lower().replace("á","a").replace("é","e").replace("í","i")\
                  .replace("ó","o").replace("ú","u") == \
       ocr.lower().replace("á","a").replace("é","e").replace("í","i")\
                   .replace("ó","o").replace("ú","u"):
        return "ACCENT_ERROR"
    if sim >= 0.85:
        return "NEAR_MATCH"         # Muy similar, error menor
    if sim >= 0.5:
        return "PARTIAL_MATCH"
    return "HALLUCINATION"          # Completamente diferente

# ─── Comparación GT vs OCR ────────────────────────────────────────────────────

def compare(gt_data, ocr_matrix):
    gt_rows = gt_data.get("rows", [])
    gt_header = gt_data.get("header", [])

    # Mapeo: índice en gt_rows -> fila OCR más cercana por etiqueta col-0
    results = []
    error_counts = {}

    # Construir lookup por label
    gt_list = []
    for i, row_obj in enumerate(gt_rows):
        data = row_obj.get("data", [])
        label = data[0] if data else ""
        gt_list.append((label, data, row_obj.get("_section",""), i))

    # Alinear con OCR
    print(f"  GT: {len(gt_list)} filas | OCR matrix: {len(ocr_matrix)} filas")

    for gt_i, (gt_label, gt_data_row, section, orig_i) in enumerate(gt_list):
        # Buscar la fila OCR que mejor coincide con el label
        best_ocr_row = None
        best_score   = -1
        for ocr_i, ocr_row in enumerate(ocr_matrix):
            ocr_label = ocr_row[0] if ocr_row else ""
            score = difflib.SequenceMatcher(None,
                        gt_label.lower(), ocr_label.lower()).ratio()
            if score > best_score:
                best_score  = score
                best_ocr_row = ocr_row
                best_ocr_i   = ocr_i

        row_errors = []
        # Comparar celda a celda (saltar col 0 - label)
        n = max(len(gt_data_row), len(best_ocr_row or []))
        for col_i in range(1, n):
            gv = gt_data_row[col_i] if col_i < len(gt_data_row) else ""
            ov = best_ocr_row[col_i] if (best_ocr_row and col_i < len(best_ocr_row)) else ""
            err = classify_error(gv, ov)
            error_counts[err] = error_counts.get(err, 0) + 1
            if err != "EXACT_MATCH":
                row_errors.append({
                    "col": col_i,
                    "gt": gv,
                    "ocr": ov,
                    "error_type": err,
                    "similarity": round(difflib.SequenceMatcher(None,str(gv),str(ov)).ratio(),3)
                })

        results.append({
            "gt_row" : gt_i,
            "label"  : gt_label,
            "section": section,
            "label_match_score": round(best_score, 3),
            "errors": row_errors
        })

    return results, error_counts

# ─── Sugerencias de mejora ────────────────────────────────────────────────────

def generate_improvement_plan(error_counts, all_errors):
    total_cells = sum(error_counts.values())
    exact       = error_counts.get("EXACT_MATCH", 0)
    accuracy    = round(exact / max(total_cells, 1) * 100, 1)

    fp = error_counts.get("FALSE_POSITIVE", 0)
    fn = error_counts.get("FALSE_NEGATIVE", 0)
    dc = error_counts.get("DIGIT_CONFUSION", 0)
    ae = error_counts.get("ACCENT_ERROR", 0)
    nm = error_counts.get("NEAR_MATCH", 0)
    pm = error_counts.get("PARTIAL_MATCH", 0)
    ha = error_counts.get("HALLUCINATION", 0)

    suggestions = []

    # 1. Falsos positivos -> umbral de confianza
    if fp > 10:
        suggestions.append({
            "priority": "ALTA",
            "tipo": "Falsos positivos en celdas vacías",
            "descripcion": f"El modelo escribe texto en {fp} celdas que deberían estar vacías.",
            "mejora": "Aplicar umbral mínimo de confianza (>0.75) en ocr_engine.ocr(). "
                       "Filtrar resultados con score bajo antes de asignarlos a la matriz.",
            "codigo": "if res and res[0]:\n"
                       "    textos = [l[1][0] for l in res[0] if l[1][1] > 0.75]\n"
                       "    matrix[r_i][c_i] = ' '.join(textos)"
        })

    # 2. Falsos negativos -> mejorar preprocesado
    if fn > 15:
        suggestions.append({
            "priority": "ALTA",
            "tipo": "Celdas con contenido no detectadas",
            "descripcion": f"El modelo no detecta nada en {fn} celdas que tienen contenido.",
            "mejora": "Aumentar el factor de zoom del roi (fx=3, fy=3) y aplicar binarización "
                       "adaptativa antes de pasar a PaddleOCR. Las celdas pequeñas con '1' o '2' "
                       "a veces no son detectadas por el motor de detección.",
            "codigo": "proc = cv2.adaptiveThreshold(\n"
                       "    clahe.apply(cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)),\n"
                       "    255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)"
        })

    # 3. Confusión dígitos
    if dc > 5:
        suggestions.append({
            "priority": "MEDIA",
            "tipo": "Confusión entre dígitos",
            "descripcion": f"{dc} celdas con errores 0→O, 1→l, 8→B, etc.",
            "mejora": "Post-proceso: en las celdas numéricas (contexto de tabla de conteo), "
                       "aplicar regex para normalizar caracteres similares.",
            "codigo": "def normalize_num(s):\n"
                       "    return re.sub(r'[OoIlS]{1}', lambda m: {'O':'0','o':'0','I':'1',"
                       "'l':'1','S':'5'}[m.group()], s) if re.match(r'^[0-9OIlSB]+$',s) else s"
        })

    # 4. Errores de tilde
    if ae > 3:
        suggestions.append({
            "priority": "BAJA",
            "tipo": "Errores de acentuación",
            "descripcion": f"{ae} palabras con tildes incorrectas (Lamina vs Lámina).",
            "mejora": "Diccionario de términos arqueológicos ya conocidos. Si el texto coincide "
                       "~90% con un término del vocabulario, reemplazar.",
            "codigo": "VOCAB = {\n"
                       "  'Lamina': 'Lámina', 'Nodulo': 'Nódulo', 'Apendice': 'Apéndice',\n"
                       "  'Periferia': 'Periferia', 'Nucleo': 'Núcleo'\n"
                       "}\n"
                       "for k,v in VOCAB.items():\n"
                       "    if difflib.SequenceMatcher(None,text,k).ratio()>0.9: text=v"
        })

    # 5. Etiquetas de fila (col 0) — texto rotado
    suggestions.append({
        "priority": "ALTA",
        "tipo": "Texto rotado en etiquetas de fila (columna 0)",
        "descripcion": "El texto de las etiquetas de fila está ROTADO 90° en la imagen. "
                        "PaddleOCR con use_angle_cls=True puede manejarlo, pero para documentos "
                        "fijos siempre rotados, mejor pre-rotar el ROI.",
        "mejora": "Detectar si la primera columna tiene celdas muy altas (h >> w) → rotar 90° "
                   "antes de OCR. Esto elimina la dependencia del clasificador de ángulo.",
        "codigo": "if h > w * 1.5:  # celda vertical\n"
                   "    roi_proc = cv2.rotate(roi_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)"
    })

    return {
        "accuracy_global": f"{accuracy}%",
        "celdas_analizadas": total_cells,
        "exactas": exact,
        "errores": {
            "FALSE_POSITIVE": fp,
            "FALSE_NEGATIVE": fn,
            "DIGIT_CONFUSION": dc,
            "ACCENT_ERROR": ae,
            "NEAR_MATCH": nm,
            "PARTIAL_MATCH": pm,
            "HALLUCINATION": ha
        },
        "plan_de_mejoras": suggestions
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  analyze_and_improve.py")
    print("=" * 65)

    # Cargar GT
    print("\n[1] Cargando ground truth...")
    with open(GT, encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_rows_count = len(gt_data.get("rows", []))
    print(f"    -> {gt_rows_count} filas en el GT")

    # Cargar imagen
    print("\n[2] Cargando imagen...")
    img = robust_read(IMG)
    if img is None:
        print("FATAL: imagen no encontrada"); sys.exit(1)
    H, W = img.shape[:2]
    print(f"    -> {W}x{H} px")

    # Grid detection
    print("\n[3] Detectando grid...")
    boxes = detect_grid(img)
    rows  = sort_to_rows(boxes)
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows) if rows else 0
    print(f"    -> {n_rows} filas x {n_cols} cols ({len(boxes)} celdas)")

    # Visualizar grid detectado
    vis = img.copy()
    for b in boxes:
        x,y,w,h = b
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,200,0), 1)
    grid_path = os.path.join(OUT, "grid_detected.jpg")
    is_ok, buf = cv2.imencode(".jpg", vis)
    if is_ok: buf.tofile(grid_path)
    print(f"    -> Visualización guardada: grid_detected.jpg")

    # OCR
    print("\n[4] Inicializando PaddleOCR (GPU)...")
    ocr = PaddleOCR(lang='es', use_angle_cls=True, use_gpu=True, show_log=False)
    print("    -> OK")

    print("\n[5] Corriendo OCR...")
    ocr_matrix = run_ocr(img, rows, ocr)

    # Guardar resultado OCR crudo
    df_ocr = pd.DataFrame(ocr_matrix)
    df_ocr.to_csv(os.path.join(OUT, "ocr_raw.csv"), index=False, encoding="utf-8-sig")

    # Comparar
    print("\n[6] Comparando GT vs OCR...")
    detail_results, error_counts = compare(gt_data, ocr_matrix)

    # Plan de mejoras
    plan = generate_improvement_plan(error_counts, detail_results)

    # ── Reporte en consola ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTADOS DE ANALISIS")
    print("=" * 65)
    print(f"  Precision global    : {plan['accuracy_global']}")
    print(f"  Celdas analizadas   : {plan['celdas_analizadas']}")
    print(f"  Exactas             : {plan['exactas']}")
    print("-" * 65)
    for k, v in plan["errores"].items():
        if v > 0:
            print(f"  {k:<30} : {v}")
    print("=" * 65)
    print("\n  PLAN DE MEJORAS:")
    for i, sug in enumerate(plan["plan_de_mejoras"], 1):
        safe_desc   = sug["descripcion"].encode("ascii","replace").decode()
        safe_mejora = sug["mejora"][:120].encode("ascii","replace").decode()
        print(f"\n  [{i}] [{sug['priority']}] {sug['tipo']}")
        print(f"      {safe_desc}")
        print(f"      MEJORA: {safe_mejora}...")

    # ── Por fila: peores etiquetas ────────────────────────────────────────
    print("\n  ETIQUETAS DE FILA (col 0) - alineacion GT vs OCR:")
    print(f"  {'GT Label':<35} {'Score':<8} {'Errores'}")
    print("  " + "-" * 65)
    for r in detail_results:
        n_err = len(r["errors"])
        score_str = f"{r['label_match_score']:.2f}"
        flag = "OK" if r["label_match_score"] >= 0.8 else "!!"
        print(f"  {r['label']:<35} {score_str:<8} {n_err} celdas incorrectas  {flag}")

    # ── Guardar JSON completo ─────────────────────────────────────────────
    full_report = {
        "summary": plan,
        "row_details": detail_results,
        "ocr_matrix_shape": [len(ocr_matrix), n_cols],
        "gt_rows": gt_rows_count
    }
    report_path = os.path.join(OUT, "analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\n  Reporte completo: {report_path}")

    # ── CSV de errores por celda ──────────────────────────────────────────
    error_rows = []
    for r in detail_results:
        for e in r["errors"]:
            error_rows.append({
                "fila_label": r["label"],
                "section":    r["section"],
                "col":        e["col"],
                "gt":         e["gt"],
                "ocr":        e["ocr"],
                "error_type": e["error_type"],
                "similarity": e["similarity"]
            })
    df_err = pd.DataFrame(error_rows)
    err_csv = os.path.join(OUT, "errors_detail.csv")
    df_err.to_csv(err_csv, index=False, encoding="utf-8-sig")
    print(f"  Detalle de errores: {err_csv}")
    print(f"\n{'='*65}")
    print("  Analisis completado!")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
