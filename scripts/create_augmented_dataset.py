import os
import cv2
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import uuid
import re

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "training")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train_data", "imgs")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "val_data", "imgs")

TRAIN_LIST = os.path.join(DATASET_DIR, "train_list.txt")
VAL_LIST = os.path.join(DATASET_DIR, "val_list.txt")

AUGMENTATIONS_PER_IMAGE = 50
SYNTHETIC_NUMBERS_TARGET = 3000

# ==========================================
# SETUP DIRS
# ==========================================
for d in [TRAIN_IMG_DIR, VAL_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# AUGMENTATION PIPELINE
# ==========================================
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=5, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
    A.MotionBlur(blur_limit=5, p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.ISONoise(p=0.3),
])

def clean_label_from_filename(filename):
    """
    Cleans tags like '(+10%)', '(-10%)' from existing filenames
    and extracts the core textual label.
    """
    name = os.path.splitext(filename)[0]
    name = re.sub(r'\(.*?\)', '', name)  # Remove things in parens
    return name.strip()

def process_user_images():
    """
    Reads user's original images, augments them, and returns label list.
    """
    labels = []
    print(f"Reading user images from {INPUT_DIR}...")
    img_paths = glob.glob(os.path.join(INPUT_DIR, "*.*"))
    
    if not img_paths:
        print("WARNING: No images found in training folder.")
        return labels

    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    for img_path in img_paths:
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in valid_exts:
            continue
            
        filename = os.path.basename(img_path)
        base_label = clean_label_from_filename(filename)
        
        # Open image correctly handling unicode paths in windows
        with open(img_path, "rb") as f:
            chunk = f.read()
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
            
        if img is None:
            continue

        for i in range(AUGMENTATIONS_PER_IMAGE):
            augmented = transform(image=img)['image']
            
            # 80/20 train/val split
            is_val = random.random() < 0.2
            target_dir = VAL_IMG_DIR if is_val else TRAIN_IMG_DIR
            prefix = "val_data" if is_val else "train_data"
            
            out_filename = f"aug_{uuid.uuid4().hex[:8]}.jpg"
            out_path = os.path.join(target_dir, out_filename)
            rel_path = f"{prefix}/imgs/{out_filename}"
            
            cv2.imencode('.jpg', augmented)[1].tofile(out_path)
            labels.append((rel_path, base_label))
            
    print(f"Generated {len(labels)} augmented images from user examples.")
    return labels

def generate_synthetic_numbers():
    """
    Generates images of numbers (integers, floats, large numbers) with random standard fonts.
    """
    print(f"Generating {SYNTHETIC_NUMBERS_TARGET} synthetic number images...")
    labels = []
    
    # fonts available on most Windows systems
    font_names = ["arial.ttf", "calibri.ttf", "times.ttf", "tahoma.ttf", "consola.ttf"]
    fonts = []
    for fn in font_names:
        try:
            fonts.append(ImageFont.truetype(fn, size=random.randint(24, 48)))
        except IOError:
            pass
            
    if not fonts:
        print("Warning: Could not load Windows fonts. Using default.")
        fonts.append(ImageFont.load_default())

    for _ in range(SYNTHETIC_NUMBERS_TARGET):
        rnd = random.random()
        if rnd < 0.4:
            # integer
            num_str = str(random.randint(0, 9999))
        elif rnd < 0.7:
            # float
            num_str = f"{random.uniform(0, 100):.2f}"
            if random.random() < 0.5: num_str = num_str.replace(".", ",") # comma decimals
        else:
            # complex numbers with separators
            num_str = f"{random.randint(1, 999):,}".replace(",", ".") # e.g. 1.500
            
        # Draw on pillow
        font = random.choice(fonts)
        # get size
        bbox = font.getbbox(num_str)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Add padding
        pad_x, pad_y = random.randint(5, 15), random.randint(5, 15)
        img_w, img_h = w + pad_x * 2, h + pad_y * 2
        if img_w <= 0 or img_h <= 0: continue
        
        # Random background color (light)
        bg_color = (random.randint(230,255), random.randint(230,255), random.randint(230,255))
        # Random text color (dark)
        txt_color = (random.randint(0,50), random.randint(0,50), random.randint(0,50))
        
        pil_img = Image.new('RGB', (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(pil_img)
        draw.text((pad_x, pad_y - bbox[1]), num_str, font=font, fill=txt_color)
        
        # Convert to cv2 and apply augmentations
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        augmented = transform(image=cv_img)['image']
        
        # 80/20 train/val split
        is_val = random.random() < 0.2
        target_dir = VAL_IMG_DIR if is_val else TRAIN_IMG_DIR
        prefix = "val_data" if is_val else "train_data"
        
        out_filename = f"synth_{uuid.uuid4().hex[:8]}.jpg"
        out_path = os.path.join(target_dir, out_filename)
        rel_path = f"{prefix}/imgs/{out_filename}"
        
        cv2.imencode('.jpg', augmented)[1].tofile(out_path)
        labels.append((rel_path, num_str))
        
    print(f"Generated {len(labels)} synthetic number images.")
    return labels

def main():
    all_labels = []
    all_labels.extend(process_user_images())
    all_labels.extend(generate_synthetic_numbers())
    
    # Shuffle
    random.shuffle(all_labels)
    
    # Write lists
    train_lines = []
    val_lines = []
    
    for rel_path, label in all_labels:
        line = f"{rel_path}\t{label}\n"
        if "train_data" in rel_path:
            train_lines.append(line)
        else:
            val_lines.append(line)
            
    with open(TRAIN_LIST, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
        
    with open(VAL_LIST, "w", encoding="utf-8") as f:
        f.writelines(val_lines)
        
    print(f"Done! Dataset saved in {DATASET_DIR}")
    print(f"Train samples: {len(train_lines)}")
    print(f"Val samples: {len(val_lines)}")

if __name__ == "__main__":
    main()
