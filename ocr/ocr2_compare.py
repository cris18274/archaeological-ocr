"""
ocr2_compare.py
===============
Compara el rendimiento del modelo PaddleOCR estándar (GPU) contra el modelo
custom 'archaeo_rec_v1' sobre la Página 3 del documento arqueológico,
usando el mismo pipeline de grid-detection que ocr2.py.

Genera:
  - ocr2_results_compare/std_results.csv
  - ocr2_results_compare/custom_results.csv
  - ocr2_results_compare/comparison_report.txt
  - (si ground_truth_page_3.json existe) métricas de precisión
"""

import cv2
import os
import sys
import json
import time
import difflib
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import logging

logging.basicConfig(level=logging.WARNING)   # silenciar PaddlePaddle spam

# ─── Rutas ────────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)

IMAGE_PATH  = os.path.join(root_dir, 'webapp', 'uploads',
                            'affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg')
# Alternativa si la anterior no existe
IMAGE_ALT   = os.path.join(root_dir, 'webapp', 'uploads',
                            'c4bb9b56-0d84-4084-aa33-4144f4e1c019_p3.jpg')

CUSTOM_MODEL = r"C:\archaeo_model"
CUSTOM_CHARS  = r"C:\archaeo_model\ppocr_keys_v1.txt"

GT_PATH     = os.path.join(root_dir, 'webapp', 'backend', 'ground_truth_page_3.json')

OUT_DIR     = os.path.join(current_dir, 'ocr2_results_compare')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def robust_read(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] robust_read: {e}")
        return None

def detect_grid(img):
    """Mismo algoritmo que ocr2.py — kernels dinámicos sobre imagen completa."""
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
    eroded = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                       iterations=2)
    grid = cv2.threshold(eroded, 0, 255, cv2.THRESH_OTSU)[1]

    cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        cv2.boundingRect(c)
        for c in cnts
        if cv2.contourArea(c) > 100 and 5 < cv2.boundingRect(c)[3] < 200
    ]
    return boxes

def sort_to_rows(boxes):
    boxes.sort(key=lambda x: x[1])
    rows, curr = [], []
    avg_h  = np.mean([b[3] for b in boxes]) if boxes else 20
    prev_y = None
    for b in boxes:
        if prev_y is None or abs(b[1] - prev_y) <= avg_h * 0.5:
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

def run_ocr_on_grid(img, rows, ocr_engine, label="OCR"):
    """Extrae texto celda a celda usando el engine dado."""
    num_cols = max(len(r) for r in rows) if rows else 0
    matrix   = [["" for _ in range(num_cols)] for _ in range(len(rows))]
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    total = sum(len(r) for r in rows)
    done  = 0
    t0    = time.time()

    for r_i, row in enumerate(rows):
        for c_i, b in enumerate(row):
            x, y, w, h = b
            roi = img[max(y-5, 0):y+h+5, max(x-5, 0):x+w+5]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                proc = clahe.apply(
                    cv2.resize(gray, None, fx=2, fy=2,
                               interpolation=cv2.INTER_CUBIC))
                try:
                    res = ocr_engine.ocr(proc, cls=True)
                    if res and res[0]:
                        matrix[r_i][c_i] = " ".join(l[1][0] for l in res[0])
                except Exception:
                    pass
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - t0
                pct = done / total * 100
                eta = (elapsed / done) * (total - done)
                print(f"  [{label}] {done}/{total} ({pct:.0f}%)  "
                      f"ETA: {eta:.0f}s", end="\r")

    elapsed = time.time() - t0
    print(f"\n  [{label}] Completado en {elapsed:.1f}s")
    return matrix, elapsed

# ─── Métricas ─────────────────────────────────────────────────────────────────

def cell_similarity(a, b):
    """Similitud entre dos strings (0-1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, str(a).strip(), str(b).strip()).ratio()

def compare_matrices(std_mat, cus_mat, gt_mat=None):
    rows  = max(len(std_mat), len(cus_mat))
    cols  = max(
        max(len(r) for r in std_mat) if std_mat else 0,
        max(len(r) for r in cus_mat) if cus_mat else 0
    )

    total      = 0
    std_non_empty = 0; cus_non_empty = 0
    agree_cells   = 0
    std_gt_sim    = []; cus_gt_sim = []

    for r in range(rows):
        sr = std_mat[r] if r < len(std_mat) else []
        cr = cus_mat[r] if r < len(cus_mat) else []
        gr = gt_mat[r] if gt_mat and r < len(gt_mat) else []

        for c in range(cols):
            sv = sr[c] if c < len(sr) else ""
            cv = cr[c] if c < len(cr) else ""
            gv = gr[c] if c < len(gr) else ""

            total += 1
            if sv.strip(): std_non_empty += 1
            if cv.strip(): cus_non_empty += 1
            if sv.strip() == cv.strip(): agree_cells += 1

            if gt_mat:
                std_gt_sim.append(cell_similarity(sv, gv))
                cus_gt_sim.append(cell_similarity(cv, gv))

    report = {
        "total_cells"        : total,
        "std_filled_cells"   : std_non_empty,
        "custom_filled_cells": cus_non_empty,
        "cells_agreed"       : agree_cells,
        "agreement_pct"      : round(agree_cells / max(total, 1) * 100, 1),
    }
    if gt_mat:
        report["std_vs_gt_similarity"]    = round(np.mean(std_gt_sim) * 100, 1)
        report["custom_vs_gt_similarity"] = round(np.mean(cus_gt_sim) * 100, 1)

    return report

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Elegir imagen disponible
    img_path = IMAGE_PATH if os.path.exists(IMAGE_PATH) else IMAGE_ALT
    print(f"\n{'='*60}")
    print(f"  ocr2_compare.py — Pipeline de comparación OCR")
    print(f"{'='*60}")
    print(f"  Imagen  : {img_path}")
    print(f"  Custom  : {CUSTOM_MODEL}")
    print(f"  Salida  : {OUT_DIR}")
    print(f"{'='*60}\n")

    # Cargar imagen
    img = robust_read(img_path)
    if img is None:
        print("FATAL: No se pudo leer la imagen.")
        sys.exit(1)
    H, W = img.shape[:2]
    print(f"  Dimensiones: {W}x{H} px")

    # Detectar grid (una sola vez para ambos engines)
    print("\n[1] Detectando grid...")
    boxes = detect_grid(img)
    rows  = sort_to_rows(boxes)
    n_rows, n_cols = len(rows), max(len(r) for r in rows) if rows else 0
    print(f"    -> {n_rows} filas x {n_cols} columnas ({len(boxes)} celdas)")

    # ── ENGINE 1: PaddleOCR estándar (GPU) ───────────────────────────────
    print("\n[2] Inicializando PaddleOCR ESTÁNDAR (GPU)...")
    ocr_std = PaddleOCR(lang='es', use_angle_cls=True, use_gpu=True,
                        show_log=False)
    print("    -> OK")

    print("\n[3] Extrayendo texto con modelo ESTÁNDAR...")
    std_matrix, std_time = run_ocr_on_grid(img, rows, ocr_std, "STD")

    df_std = pd.DataFrame(std_matrix)
    df_std.to_csv(os.path.join(OUT_DIR, "std_results.csv"),
                  index=False, encoding="utf-8-sig")
    print(f"    -> CSV guardado: std_results.csv")

    # ── ENGINE 2: Modelo CUSTOM ──────────────────────────────────────────
    print("\n[4] Inicializando modelo CUSTOM (archaeo_rec_v1)...")
    if not os.path.isdir(CUSTOM_MODEL):
        print(f"    [WARN] Carpeta custom no encontrada: {CUSTOM_MODEL}")
        print("    -> Usando estándar como fallback para custom slot.")
        ocr_cus = ocr_std
    else:
        # Normalizar separadores de path para Windows
        cus_model_norm = os.path.normpath(CUSTOM_MODEL)
        char_dict_norm = os.path.normpath(CUSTOM_CHARS) if os.path.exists(CUSTOM_CHARS) else None
        try:
            paddle_kwargs = dict(
                rec_model_dir = cus_model_norm,
                use_angle_cls = True,
                lang          = 'es',
                use_gpu       = True,
                show_log      = False,
            )
            if char_dict_norm:
                paddle_kwargs["rec_char_dict_path"] = char_dict_norm
            ocr_cus = PaddleOCR(**paddle_kwargs)
            print("    -> Modelo custom cargado OK")
        except Exception as e:
            print(f"    [ERROR] No se pudo cargar modelo custom: {str(e)[:300]}")
            print("    -> Fallback: usando estandar")
            ocr_cus = ocr_std

    print("\n[5] Extrayendo texto con modelo CUSTOM...")
    cus_matrix, cus_time = run_ocr_on_grid(img, rows, ocr_cus, "CUSTOM")

    df_cus = pd.DataFrame(cus_matrix)
    df_cus.to_csv(os.path.join(OUT_DIR, "custom_results.csv"),
                  index=False, encoding="utf-8-sig")
    print(f"    -> CSV guardado: custom_results.csv")

    # ── Ground truth (opcional) ──────────────────────────────────────────
    gt_matrix = None
    if os.path.exists(GT_PATH):
        print(f"\n[6] Cargando ground truth desde: {GT_PATH}")
        try:
            gt_data = json.load(open(GT_PATH, encoding="utf-8"))
            # Espera estructura: {"rows": [[celda,...], ...], "header": [...]}
            gt_rows = gt_data.get("rows", [])
            gt_h    = gt_data.get("header", [])
            gt_matrix = [gt_h] + gt_rows if gt_h else gt_rows
            print(f"    -> {len(gt_matrix)} filas en ground truth")
        except Exception as e:
            print(f"    [WARN] No se pudo parsear GT: {e}")

    # ── Comparación y reporte ────────────────────────────────────────────
    print("\n[7] Generando reporte de comparación...")
    metrics = compare_matrices(std_matrix, cus_matrix, gt_matrix)

    # ── Diferencias fila por fila (primeras 15 filas, col 0) ─────────────
    diffs = []
    for r_i in range(min(len(std_matrix), len(cus_matrix), 40)):
        sv = std_matrix[r_i][0] if std_matrix[r_i] else ""
        cv = cus_matrix[r_i][0] if cus_matrix[r_i] else ""
        match = "OK" if sv.strip() == cv.strip() else "XX"
        diffs.append((r_i, sv, cv, match))

    # -- Imprimir reporte en consola ---------------------------------------
    sep = "-" * 68
    print("=" * 68)
    print("  REPORTE DE COMPARACION -- Pagina 3")
    print("=" * 68)
    print(f"  Grid detectado      : {n_rows} filas x {n_cols} cols = {len(boxes)} celdas")
    print(f"  Tiempo STD          : {std_time:.1f}s")
    print(f"  Tiempo CUSTOM       : {cus_time:.1f}s")
    print(sep)
    print(f"  Celdas totales      : {metrics['total_cells']}")
    print(f"  Celdas no vacias STD: {metrics['std_filled_cells']}")
    print(f"  Celdas no vacias CUS: {metrics['custom_filled_cells']}")
    print(f"  Celdas que coinciden: {metrics['cells_agreed']} ({metrics['agreement_pct']}%)")
    if gt_matrix:
        print("-" * 68)
        print(f"  Similitud STD vs GT : {metrics.get('std_vs_gt_similarity', 'N/A')}%")
        print(f"  Similitud CUS vs GT : {metrics.get('custom_vs_gt_similarity', 'N/A')}%")
    print("=" * 68)

    # Tabla comparativa columna 0 (etiquetas de fila)
    print(f"\n  {'ROW':<5} {'ESTANDAR (col 0)':<30} {'CUSTOM (col 0)':<30} OK?")
    print("  " + "-" * 68)
    for r_i, sv, cv, match in diffs:
        sv_s = sv[:28] + ".." if len(sv) > 28 else sv
        cv_s = cv[:28] + ".." if len(cv) > 28 else cv
        print(f"  {r_i:<5} {sv_s:<30} {cv_s:<30} {match}")

    # ── Guardar reporte de texto ──────────────────────────────────────────
    report_lines = [
        "REPORTE DE COMPARACIÓN OCR — Página 3",
        "=" * 68,
        f"Imagen  : {img_path}",
        f"Custom  : {CUSTOM_MODEL}",
        f"Grid    : {n_rows} filas × {n_cols} cols = {len(boxes)} celdas",
        f"Tiempo STD  : {std_time:.1f}s",
        f"Tiempo CUS  : {cus_time:.1f}s",
        "",
        f"Celdas totales       : {metrics['total_cells']}",
        f"Celdas no vacías STD : {metrics['std_filled_cells']}",
        f"Celdas no vacías CUS : {metrics['custom_filled_cells']}",
        f"Celdas que coinciden : {metrics['cells_agreed']} ({metrics['agreement_pct']}%)",
    ]
    if gt_matrix:
        report_lines += [
            f"Sim STD vs GT        : {metrics.get('std_vs_gt_similarity')}%",
            f"Sim CUS vs GT        : {metrics.get('custom_vs_gt_similarity')}%",
        ]
    report_lines += ["", "DETALLE — Columna 0 (etiquetas de fila):",
                     f"{'ROW':<5} {'ESTÁNDAR':<35} {'CUSTOM':<35} OK?"]
    for r_i, sv, cv, match in diffs:
        report_lines.append(f"{r_i:<5} {sv:<35} {cv:<35} {match}")

    report_path = os.path.join(OUT_DIR, "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Reporte guardado en: {report_path}")

    # ── DataFrame de diferencias lado a lado ─────────────────────────────
    max_rows = max(len(std_matrix), len(cus_matrix))
    max_cols = max(
        max(len(r) for r in std_matrix) if std_matrix else 0,
        max(len(r) for r in cus_matrix) if cus_matrix else 0,
    )
    diff_records = []
    for r_i in range(max_rows):
        sr = std_matrix[r_i] if r_i < len(std_matrix) else []
        cr = cus_matrix[r_i] if r_i < len(cus_matrix) else []
        for c_i in range(max_cols):
            sv = sr[c_i] if c_i < len(sr) else ""
            cv = cr[c_i] if c_i < len(cr) else ""
            sim = cell_similarity(sv, cv)
            diff_records.append({
                "row": r_i, "col": c_i,
                "std": sv, "custom": cv,
                "similarity": round(sim, 3),
                "match": sv.strip() == cv.strip()
            })
    df_diff = pd.DataFrame(diff_records)
    df_diff.to_csv(os.path.join(OUT_DIR, "diff_table.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"  Tabla de diferencias: diff_table.csv")
    print(f"\n{'='*68}")
    print("  ¡Comparación completada!")
    print(f"{'='*68}\n")

if __name__ == "__main__":
    main()
