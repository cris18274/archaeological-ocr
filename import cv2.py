import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import pandas as pd
import re
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import tempfile
import logging

# ======================================
# CONFIGURACIÓN DE LOGGING Y OCR
# ======================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ocr = PaddleOCR(lang='es', use_textline_orientation=True)

# ======================================
# FUNCIONES DE PROCESAMIENTO
# ======================================

def limpiar_texto(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\x20-\x7EáéíóúÁÉÍÓÚüÜñÑ.,:%/°º\-()\[\]0-9a-zA-Z ]+', '', text)
    return text.strip()

def detectar_tabla(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 15)

    H, W = gray.shape[:2]
    v_kernel_len = max(10, H // 40)
    h_kernel_len = max(10, W // 40)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))

    vertical = cv2.dilate(cv2.erode(bin_img, vertical_kernel), vertical_kernel)
    horizontal = cv2.dilate(cv2.erode(bin_img, horizontal_kernel), horizontal_kernel)

    grid = cv2.addWeighted(vertical, 0.5, horizontal, 0.5, 0)
    grid = cv2.erode(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, grid = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = [cnt for cnt in contours if cv2.contourArea(cnt) > (H*W*0.00002)]
    boxes = [cv2.boundingRect(c) for c in cells]
    centers = [(x+w/2, y+h/2, w, h, c) for (x,y,w,h), c in zip(boxes, cells)]

    med_h = np.median([h for _,_,_,h,_ in centers]) if centers else 20
    row_thr = med_h * 0.6
    centers.sort(key=lambda t: t[1])
    rows, cur, prev_y = [], [], None

    for xc,yc,w,h,cnt in centers:
        if prev_y is None or abs(yc-prev_y)<=row_thr:
            cur.append((xc,yc,w,h,cnt))
        else:
            cur.sort(key=lambda t:t[0])
            rows.append([t[4] for t in cur])
            cur=[(xc,yc,w,h,cnt)]
        prev_y=yc
    if cur:
        cur.sort(key=lambda t:t[0])
        rows.append([t[4] for t in cur])

    return rows

def ocr_celda(cell_img):
    if len(cell_img.shape)==2:
        rgb = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(rgb, cls=True)
    result = result[0] if result else []
    texts = [r[1][0] for r in result] if result else []
    return limpiar_texto(" ".join(texts))

def procesar_imagen(img_path, output_dir):
    logger.info(f"Procesando: {img_path}")
    img = cv2.imread(img_path)
    rows = detectar_tabla(img)
    if not rows:
        raise ValueError("No se detectaron celdas en la imagen.")

    num_rows = len(rows)
    num_cols = max(len(r) for r in rows)
    tabla = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for i, row in enumerate(rows):
        for j, cnt in enumerate(row):
            x, y, w, h = cv2.boundingRect(cnt)
            roi = img[y:y+h, x:x+w]
            texto = ocr_celda(roi)
            tabla[i][j] = texto

    df = pd.DataFrame(tabla)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    csv = os.path.join(output_dir, f"{base}_ocr.csv")
    xlsx = os.path.join(output_dir, f"{base}_ocr.xlsx")
    df.to_csv(csv, index=False, encoding='utf-8')
    df.to_excel(xlsx, index=False)

    todas_palabras = " ".join(df.fillna("").astype(str).values.flatten()).split()
    palabra_larga = max(todas_palabras, key=len) if todas_palabras else ""
    celdas_llenas = np.sum(df.applymap(lambda x: bool(str(x).strip())).values)

    return df, palabra_larga, celdas_llenas


# ======================================
# INTERFAZ TKINTER
# ======================================

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluador OCR - Tablas")
        self.root.geometry("900x600")
        self.file_path = None

        self.label_title = ttk.Label(root, text="Evaluación de Tablas con OCR (PaddleOCR)", font=("Segoe UI", 14))
        self.label_title.pack(pady=10)

        # Botón de selección
        self.btn_select = ttk.Button(root, text="Seleccionar archivo (PDF o Imagen)", command=self.select_file)
        self.btn_select.pack(pady=10)

        # Panel de imagen
        self.canvas = tk.Canvas(root, width=600, height=400, bg="gray90")
        self.canvas.pack(pady=10)

        # Resultados
        self.label_info = ttk.Label(root, text="Resultados aparecerán aquí...", font=("Segoe UI", 10))
        self.label_info.pack(pady=10)

        # Barra de progreso
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=5)

    def select_file(self):
        filetypes = [("Archivos", "*.png *.jpg *.jpeg *.pdf")]
        filepath = filedialog.askopenfilename(title="Selecciona un archivo", filetypes=filetypes)
        if not filepath:
            return
        self.file_path = filepath
        self.display_preview(filepath)
        self.run_ocr()

    def display_preview(self, filepath):
        try:
            if filepath.lower().endswith(".pdf"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    poppler_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poppler", "poppler-24.08.0", "Library", "bin")
                    imgs = convert_from_path(filepath, dpi=200, first_page=1, last_page=1, poppler_path=poppler_dir)
                    img_path = os.path.join(temp_dir, "preview.jpg")
                    imgs[0].save(img_path, "JPEG")
                    img = Image.open(img_path)
            else:
                img = Image.open(filepath)

            img.thumbnail((600, 400))
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(300, 200, image=self.img_tk)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo mostrar la imagen: {e}")

    def run_ocr(self):
        if not self.file_path:
            messagebox.showwarning("Atención", "Primero selecciona un archivo.")
            return

        output_dir = os.path.join(os.path.dirname(self.file_path), "salidas_ocr")
        os.makedirs(output_dir, exist_ok=True)
        self.progress.start(10)
        self.label_info.config(text="Procesando OCR, espera unos segundos...")

        self.root.after(100, self.process_file, self.file_path, output_dir)

    def process_file(self, filepath, output_dir):
        try:
            if filepath.lower().endswith(".pdf"):
                poppler_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poppler", "poppler-24.08.0", "Library", "bin")
                pages = convert_from_path(filepath, dpi=250, poppler_path=poppler_dir)
                for i, page in enumerate(pages):
                    temp_img = os.path.join(output_dir, f"page_{i+1}.jpg")
                    page.save(temp_img, "JPEG")
                    df, palabra_larga, celdas_llenas = procesar_imagen(temp_img, output_dir)
                    self.show_results(df, palabra_larga, celdas_llenas, i+1)
            else:
                df, palabra_larga, celdas_llenas = procesar_imagen(filepath, output_dir)
                self.show_results(df, palabra_larga, celdas_llenas)
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")
        finally:
            self.progress.stop()

    def show_results(self, df, palabra_larga, celdas_llenas, page=None):
        filas, cols = df.shape
        text = f"✅ OCR completado.\nFilas detectadas: {filas}\nColumnas: {cols}\nCeldas llenas: {celdas_llenas}\nPalabra más larga: {palabra_larga}"
        if page:
            text = f"Página {page}:\n" + text
        self.label_info.config(text=text)
        logger.info(text)


# ======================================
# EJECUCIÓN PRINCIPAL
# ======================================

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
