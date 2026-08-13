import os
import cv2
import sys
import easyocr

def benchmark():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    # ROI: Catalog 1 value "5"
    # x=~235, y=~395, w=~45, h=~35
    crop = img[395:430, 235:280]
    
    print("Init EasyOCR...")
    reader = easyocr.Reader(['es','en'], gpu=True)
    
    print("Reading crop (raw)...")
    res = reader.readtext(crop, detail=0)
    print("Raw: ", res)
    
    print("Reading crop (upscaled 3x)...")
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res_up = reader.readtext(up, detail=0)
    print("Upscaled: ", res_up)

if __name__ == "__main__":
    benchmark()
