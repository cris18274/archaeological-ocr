import sys
from paddleocr import PaddleOCR
import numpy as np

def test_full():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    print("Init PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='es', use_gpu=False)
    
    print(f"Reading full image: {img_path}")
    result = ocr.ocr(img_path, cls=True)
    
    if result and result[0]:
        for i, line in enumerate(result[0]):
            print(f"Line {i}: {line[1]}")
            if i > 5: break
    else:
        print("No result")

if __name__ == "__main__":
    test_full()
