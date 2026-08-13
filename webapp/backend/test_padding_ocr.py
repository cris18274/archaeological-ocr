import os
import cv2
import sys
import easyocr

def test_padding():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    # ROI: Catalog 1 value "5"
    crop = img[395:430, 235:280]
    
    reader = easyocr.Reader(['es','en'], gpu=True)
    
    # ADD PADDING (15px)
    PAD = 15
    padded = cv2.copyMakeBorder(crop, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.imwrite("debug_padded_roi.png", padded)
    
    print("Reading padded crop (raw)...")
    res = reader.readtext(padded, detail=0)
    print("Padded: ", res)
    
    print("Reading padded + upscaled 3x...")
    up = cv2.resize(padded, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res_up = reader.readtext(up, detail=0)
    print("Padded + Upscaled: ", res_up)

if __name__ == "__main__":
    test_padding()
