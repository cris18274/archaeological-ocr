import sys
from paddleocr import PaddleOCR
import numpy as np
import cv2

def test_cls():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    print("Init PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='es', use_gpu=False)
    
    # Crop a vertical word area (roughly where PANZALEO is)
    # Let's just use the whole image first to see the general orientation result
    # For cls-only:
    print("Running CLS on full image...")
    res = ocr.ocr(img, det=False, rec=False, cls=True)
    print("CLS result: ", res)

if __name__ == "__main__":
    test_cls()
