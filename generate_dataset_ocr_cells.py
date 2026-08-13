# generate_dataset_ocr_cells.py
import os, random, math, json
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

OUT_DIR = r"dataset"
TRAIN_N = 8000
VAL_N = 2000
IMG_W, IMG_H = 320, 48   # shape por defecto para rec (3,48,320)
BG_COLORS = [(250,250,250), (245,247,250), (252,252,240)]
FONTS = [
    # agrega rutas a TTF/OTF de tu sistema si quieres
    None  # None -> fuente por defecto PIL (robusta cross-platform)
]
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + \
        list("abcdefghijklmnopqrstuvwxyz") + \
        list("0123456789") + \
        list(" -()%/°º[].,")
TOKENS_NUM = [str(i) for i in range(0, 201)]
TOKENS_SYM = ["(-10%)", "(+10%)", "(0%)", "-", "—", "— —", "N", "n", "P", "p", "O", "o"]
TOKENS_WORD = ["Vasija", "Base", "Pedestal", "Asa", "Pico", "Cuerpo", "Borde", "Deco", "Impreso", "Fragm.", "Indef."]
SEED = 1337
random.seed(SEED)

def rand_font(size):
    path = random.choice(FONTS)
    try:
        return ImageFont.truetype(path if path else "arial.ttf", size)
    except:
        return ImageFont.load_default()

def make_cell_like_bg(w, h):
    # fondo ligeramente coloreado + líneas finas simulando cuadrícula
    img = Image.new("RGB", (w, h), random.choice(BG_COLORS))
    d = ImageDraw.Draw(img)
    # líneas horizontales/verticales muy finas
    if random.random() < 0.9:
        col = (200, 200, 200)
        # vertical
        for x in range(0, w, random.randint(12, 18)):
            d.line([(x,0),(x,h)], fill=col, width=1)
        # horizontal
        for y in range(0, h, random.randint(12, 16)):
            d.line([(0,y),(w,y)], fill=col, width=1)
    # sombreado leve
    if random.random() < 0.4:
        img = ImageOps.colorize(ImageOps.grayscale(img), black="#f7f7f7", white="#ffffff")
    return img

def synth_text():
    mode = random.choices(
        ["num","sym","char","word","shortline"],
        weights=[35,20,20,15,10], k=1
    )[0]
    if mode == "num":
        return random.choice(TOKENS_NUM)
    if mode == "sym":
        return random.choice(TOKENS_SYM)
    if mode == "char":
        L = random.randint(1, 3)
        return "".join(random.choice(CHARS) for _ in range(L))
    if mode == "word":
        return random.choice(TOKENS_WORD)
    if mode == "shortline":
        L = random.randint(2, 8)
        return "".join(random.choice(CHARS) for _ in range(L))
    return "A"

def render_sample(text, w=IMG_W, h=IMG_H):
    img = make_cell_like_bg(w, h)
    d = ImageDraw.Draw(img)

    # tamaño de fuente pequeño a medio
    fs = random.randint(14, 28)
    font = rand_font(fs)

    # color de texto (oscuro, pero variable)
    tc = random.choice([(0,0,0),(20,20,20),(30,30,30),(10,10,10)])

    # medir y centrar
    tw, th = d.textbbox((0,0), text, font=font)[2:]
    x = (w - tw) // 2 + random.randint(-4, 4)
    y = (h - th) // 2 + random.randint(-2, 2)

    # opcional: subrayados o marcas débiles
    if random.random() < 0.07:
        d.line([(5,h-5),(w-5,h-5)], fill=(180,180,180), width=1)

    d.text((x, y), text, fill=tc, font=font)

    # augmentaciones suaves
    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    if random.random() < 0.2:
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255))
    if random.random() < 0.25:
        # ruido leve
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, 8, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img

def make_split(n, split_name):
    img_dir = os.path.join(OUT_DIR, split_name)
    os.makedirs(img_dir, exist_ok=True)
    lab_path = os.path.join(OUT_DIR, f"{split_name}_label.txt")
    with open(lab_path, "w", encoding="utf-8") as f:
        for i in range(n):
            text = synth_text()
            img = render_sample(text)
            name = f"{split_name}_{i:06d}.jpg"
            img.save(os.path.join(img_dir, name), quality=95)
            f.write(f"{os.path.join(img_dir, name)}\t{text}\n")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_split(TRAIN_N, "train")
    make_split(VAL_N, "val")
    print("OK. Dataset en ./dataset (train/val + label.txt)")

if __name__ == "__main__":
    main()
