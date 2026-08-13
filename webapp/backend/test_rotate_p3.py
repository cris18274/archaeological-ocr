import os
import cv2
import sys

sys.path.append(os.getcwd())
from main import correct_vertical_text_90

def test_auto_rotate():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    print(f"Original: {img.shape}")
    
    rotated, was_rotated = correct_vertical_text_90(img)
    print(f"Was rotated: {was_rotated}")
    if was_rotated:
        print(f"New shape: {rotated.shape}")
        cv2.imwrite("final_rotated_p3.png", rotated)
    else:
        print("No rotation detected.")
        cv2.imwrite("final_rotated_p3.png", img)

if __name__ == "__main__":
    test_auto_rotate()
