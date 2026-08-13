import os
import cv2
import sys
from paddleocr import PaddleOCR

def benchmark_paddle():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    # ROI: Catalog 1 value "5"
    crop = img[395:430, 235:280]
    
    print("Init PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='es', use_gpu=False)
    
    print("Reading crop (raw)...")
    res = ocr.ocr(crop, cls=False)
    print("Raw: ", res)
    
    print("Reading crop (upscaled 3x)...")
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res_up = ocr.ocr(up, cls=False)
    print("Upscaled: ", res_up)

if __name__ == "__main__":
    benchmark_paddle()
