import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from paddleocr import PaddleOCR
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rutas de archivos
IMAGE_PATH = r'C:\Users\ASUS\Downloads\final-funding\prueba.png'
OUTPUT_DIR = os.path.dirname(IMAGE_PATH)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'tabla_ceramica_formateada.csv')
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, 'tabla_ceramica_formateada.xlsx')
DEBUG_DIR = os.path.join(OUTPUT_DIR, 'debug_images')
DEBUG_OCR_DIR = os.path.join(DEBUG_DIR, 'ocr_cells')

# Crear carpetas para depuración
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(DEBUG_OCR_DIR, exist_ok=True)

# Configuración para limitar longitud de texto
MAX_TEXT_LENGTH = 1000

# 1. Verificar configuración del entorno
logger.info("Verificando configuración del entorno...")
try:
    import paddle
    logger.info(f"PaddlePaddle versión: {paddle.__version__}")
    logger.info(f"CUDA disponible: {paddle.device.is_compiled_with_cuda()}")
except ImportError:
    logger.error("PaddlePaddle no está instalado. Se requiere para PaddleOCR.")
    raise
logger.info(f"TensorFlow versión: {tf.__version__}")

# 2. Cargar imagen
logger.info("Cargando imagen...")
img = cv2.imread(IMAGE_PATH)
if img is None:
    logger.error(f"No se pudo cargar la imagen desde {IMAGE_PATH}")
    raise FileNotFoundError(f"No se pudo cargar la imagen desde {IMAGE_PATH}")
altura_imagen, anchura_imagen = img.shape[:2]
cv2.imwrite(os.path.join(DEBUG_DIR, '01_original.jpg'), img)

# 3. Función de detección de tabla MEJORADA
def table_detection(img_path, debug_dir, debug_ocr_dir):
    logger.info("Detectando tabla...")
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"No se pudo leer la imagen desde {img_path}")
        raise FileNotFoundError(f"No se pudo leer la imagen desde {img_path}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(debug_dir, '02_gray.jpg'), img_gray)

    # Binarización con Otsu
    (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)
    cv2.imwrite(os.path.join(debug_dir, '03_binary.jpg'), img_bin)

    # Detección de líneas verticales
    kernel_length_v = (np.array(img_gray).shape[1]) // 120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)
    cv2.imwrite(os.path.join(debug_dir, '04_vertical_lines.jpg'), vertical_lines_img)

    # Detección de líneas horizontales
    kernel_length_h = (np.array(img_gray).shape[1]) // 40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)
    cv2.imwrite(os.path.join(debug_dir, '05_horizontal_lines.jpg'), horizontal_lines_img)

    # Combinar líneas para formar la cuadrícula
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(debug_dir, '06_table_segment.jpg'), table_segment)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos
    cell_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100 and cv2.boundingRect(cnt)[3] > 5 and cv2.boundingRect(cnt)[3] < 200]
    logger.info(f"Se detectaron {len(cell_contours)} contornos válidos")

    # FUNCIÓN MEJORADA DE ORDENAMIENTO
    def sort_contours_improved(cnts):
        """
        Ordena los contornos en una estructura de tabla 2D mejorada
        """
        if not cnts:
            return []
        
        # Obtener bounding boxes
        boxes = [cv2.boundingRect(cnt) for cnt in cnts]
        
        # Agrupar por filas usando clustering basado en posición Y
        boxes_with_contours = list(zip(boxes, cnts))
        boxes_with_contours.sort(key=lambda x: x[0][1])  # Ordenar por Y
        
        rows = []
        current_row = []
        
        # Calcular threshold dinámico para agrupar filas
        y_positions = [box[1] for box in boxes]
        heights = [box[3] for box in boxes]
        avg_height = np.mean(heights) if heights else 20
        row_threshold = avg_height * 0.5  # 50% de la altura promedio
        
        logger.info(f"Altura promedio de celdas: {avg_height}, threshold para filas: {row_threshold}")
        
        previous_y = None
        for (x, y, w, h), cnt in boxes_with_contours:
            if previous_y is None or abs(y - previous_y) <= row_threshold:
                # Misma fila
                current_row.append(((x, y, w, h), cnt))
            else:
                # Nueva fila
                if current_row:
                    # Ordenar la fila actual por posición X
                    current_row.sort(key=lambda x: x[0][0])
                    rows.append([item[1] for item in current_row])  # Solo contornos
                current_row = [((x, y, w, h), cnt)]
                
            previous_y = y
        
        # Agregar la última fila
        if current_row:
            current_row.sort(key=lambda x: x[0][0])
            rows.append([item[1] for item in current_row])
        
        # Log para debug
        logger.info(f"Se detectaron {len(rows)} filas")
        for i, row in enumerate(rows):
            logger.info(f"Fila {i}: {len(row)} celdas")
            for j, cnt in enumerate(row):
                x, y, w, h = cv2.boundingRect(cnt)
                logger.info(f"  Celda [{i},{j}]: x={x}, y={y}, w={w}, h={h}")
        
        return rows

    sorted_rows = sort_contours_improved(cell_contours)
    
    # Visualizar contornos con mejor etiquetado
    contour_img = img.copy()
    for i, row in enumerate(sorted_rows):
        for j, cnt in enumerate(row):
            x, y, w, h = cv2.boundingRect(cnt)
            # Colores diferentes para cada fila
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 6]
            cv2.rectangle(contour_img, (x, y), (x+w, y+h), color, 2)
            # Etiqueta más clara
            label = f"[{i},{j}]"
            cv2.putText(contour_img, label, (x+5, y+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Fondo negro
            cv2.putText(contour_img, label, (x+5, y+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Texto blanco
    
    cv2.imwrite(os.path.join(debug_dir, '07_cell_contours_improved.jpg'), contour_img)
    
    return sorted_rows, img

# 4. Inicializar PaddleOCR
logger.info("Inicializando PaddleOCR...")
ocr = PaddleOCR(lang='es', use_angle_cls=True, use_gpu=False)

# 5. Revisar orientación del texto
logger.info("Analizando orientación del texto en la imagen completa...")
output = ocr.ocr(IMAGE_PATH)[0]
rotated = False

if output:
    angles = [line[1][2] for line in output if len(line[1]) > 2]
    if angles:
        avg_angle = np.mean(angles)
        logger.info(f"Ángulo promedio de texto detectado: {avg_angle} grados")
        if abs(avg_angle) > 5:
            logger.info(f"Rotando imagen {avg_angle} grados para alinear texto...")
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(os.path.join(DEBUG_DIR, '08_rotated.jpg'), img)
            rotated = True

# 6. Ejecutar detección de tabla
if rotated:
    sorted_rows, img = table_detection(os.path.join(DEBUG_DIR, '08_rotated.jpg'), DEBUG_DIR, DEBUG_OCR_DIR)
else:
    sorted_rows, img = table_detection(IMAGE_PATH, DEBUG_DIR, DEBUG_OCR_DIR)

# 7. Extraer y reconocer texto con PaddleOCR
logger.info("Extrayendo texto de las celdas con PaddleOCR...")

def preprocess_cell(cell_img):
    scale_factor = 2
    cell_img = cv2.resize(cell_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    if len(cell_img.shape) == 3:
        cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        cell_gray = cell_img
    cell_gray = cv2.GaussianBlur(cell_gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cell_enhanced = clahe.apply(cell_gray)
    return cell_enhanced

def clean_ocr_text(text):
    if not text or pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    if len(re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜñÑ.,-]', '', text)) < len(text) * 0.1:
        return ""
    return text[:MAX_TEXT_LENGTH]

def recognize_cell_text(cell_img, row_idx, col_idx):
    cell_id = f"{row_idx}_{col_idx}"
    processed_cell = preprocess_cell(cell_img)
    cell_path = os.path.join(DEBUG_OCR_DIR, f'cell_{cell_id}_processed.jpg')
    cv2.imwrite(cell_path, processed_cell)
    
    try:
        result = ocr.ocr(cell_path)[0]
        if result:
            texts = [line[1][0] for line in result]
            confidence = [line[1][1] for line in result]
            angles = [line[1][2] for line in result if len(line[1]) > 2]
            
            if angles and not rotated:
                avg_angle = np.mean(angles)
                if abs(avg_angle) > 5:
                    logger.debug(f"Celda [{row_idx},{col_idx}] con texto rotado: {avg_angle} grados")
                    (h, w) = cell_img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    rotated_cell = cv2.warpAffine(cell_img, M, (w, h))
                    rotated_path = os.path.join(DEBUG_OCR_DIR, f'cell_{cell_id}_rotated.jpg')
                    cv2.imwrite(rotated_path, rotated_cell)
                    result = ocr.ocr(rotated_path)[0]
                    if result:
                        texts = [line[1][0] for line in result]
            
            text = " ".join(texts)
            text = clean_ocr_text(text)
            logger.debug(f"Celda [{row_idx},{col_idx}]: Texto = '{text}'")
            
            # Guardar información de debug
            with open(os.path.join(DEBUG_OCR_DIR, f'cell_{cell_id}_text.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Posición: [{row_idx},{col_idx}]\n")
                f.write(f"Texto reconocido: {text}\n")
                f.write(f"Confianza promedio: {np.mean(confidence) if confidence else 0:.2f}\n")
                if angles:
                    f.write(f"Ángulos: {angles}\n")
            
            return text
        else:
            logger.debug(f"Celda [{row_idx},{col_idx}]: No se detectó texto")
            return ""
    except Exception as e:
        logger.error(f"Error OCR en celda [{row_idx},{col_idx}]: {e}")
        return ""

# RECONSTRUCCIÓN MEJORADA DE LA TABLA
logger.info("Procesando celdas y construyendo tabla...")

# Calcular dimensiones de la tabla
num_rows = len(sorted_rows)
num_cols = max(len(row) for row in sorted_rows) if sorted_rows else 0
logger.info(f"Dimensiones de la tabla: {num_rows} filas, {num_cols} columnas")

# Inicializar matriz de resultados
table_matrix = [["" for _ in range(num_cols)] for _ in range(num_rows)]

# Procesar cada celda directamente en su posición correcta
for row_idx, row in enumerate(sorted_rows):
    for col_idx, cnt in enumerate(row):
        x, y, w, h = cv2.boundingRect(cnt)
        logger.debug(f"Procesando celda [{row_idx},{col_idx}]: x={x}, y={y}, w={w}, h={h}")
        
        # Extraer ROI con margen
        margin = 5
        roi = img[max(y-margin, 0):y+h+margin, max(x-margin, 0):x+w+margin]
        
        if roi.size == 0:
            cell_text = ""
            logger.debug(f"Celda [{row_idx},{col_idx}]: ROI vacía")
        else:
            # Guardar imagen original de la celda
            cell_path = os.path.join(DEBUG_OCR_DIR, f'cell_{row_idx}_{col_idx}_original.jpg')
            cv2.imwrite(cell_path, roi)
            
            # Reconocer texto
            cell_text = recognize_cell_text(roi, row_idx, col_idx)
        
        # Asignar directamente a la matriz
        table_matrix[row_idx][col_idx] = cell_text
        logger.debug(f"Asignado texto '{cell_text}' a posición [{row_idx},{col_idx}]")

# Crear visualización final con texto reconocido
logger.info("Creando visualización final...")
final_vis = img.copy()
for row_idx, row in enumerate(sorted_rows):
    for col_idx, cnt in enumerate(row):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Color basado en si hay texto o no
        color = (0, 255, 0) if table_matrix[row_idx][col_idx] else (0, 0, 255)
        cv2.rectangle(final_vis, (x, y), (x+w, y+h), color, 2)
        
        # Mostrar texto reconocido (primeros 15 caracteres)
        text = table_matrix[row_idx][col_idx]
        display_text = text[:15] + "..." if len(text) > 15 else text
        if not display_text:
            display_text = "EMPTY"
        
        # Etiqueta con posición y texto
        label = f"[{row_idx},{col_idx}]: {display_text}"
        
        # Texto con fondo para mejor legibilidad
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(final_vis, (x, y-text_height-5), (x+text_width, y), (255, 255, 255), -1)
        cv2.putText(final_vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

cv2.imwrite(os.path.join(DEBUG_DIR, '09_final_table_result.jpg'), final_vis)

# Log del contenido de la matriz
logger.info("Contenido de la tabla reconstruida:")
for i, row in enumerate(table_matrix):
    logger.info(f"Fila {i}: {row}")

# 8. Crear DataFrame final
logger.info("Creando DataFrame final...")
df = pd.DataFrame(table_matrix)

# Asignar nombres de columnas
df.columns = [f"Col_{i}" for i in range(len(df.columns))]

# Limpiar datos
def clean_non_alphanumeric(text):
    if not text or pd.isna(text):
        return ""
    if isinstance(text, str):
        if re.fullmatch(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜñÑ\s.,-]+', text):
            return ""
        return text.strip()[:MAX_TEXT_LENGTH]
    return text

# Aplicar limpieza
for i in range(len(df)):
    for j in range(len(df.columns)):
        df.iloc[i, j] = clean_non_alphanumeric(df.iloc[i, j])

logger.info("DataFrame final:")
logger.info(df.to_string())

# 9. Guardar resultados
logger.info("Guardando resultados...")
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
df.to_excel(OUTPUT_EXCEL, index=False)

# Guardar matriz como respaldo
pd.DataFrame(table_matrix).to_csv(os.path.join(OUTPUT_DIR, 'table_matrix_backup.csv'), index=False, encoding='utf-8')

logger.info("✅ Proceso completado exitosamente!")
logger.info("✅ Archivos generados:")
logger.info(f"- CSV: {OUTPUT_CSV}")
logger.info(f"- Excel: {OUTPUT_EXCEL}")
logger.info(f"- Respaldo de matriz: {os.path.join(OUTPUT_DIR, 'table_matrix_backup.csv')}")
logger.info(f"- Imágenes de depuración: {DEBUG_DIR}")
logger.info(f"- Celdas OCR individuales: {DEBUG_OCR_DIR}")

logger.info("\n" + "="*50)
logger.info("VISTA PREVIA DE LA TABLA EXTRAÍDA:")
logger.info("="*50)
print(df.to_string())
logger.info("="*50)




