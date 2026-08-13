import cv2
import os

def overlay_grid():
    img_path = r"../uploads/affc44d0-9985-4b7c-bb6f-ea1ce9cf7d85_p3.jpg"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Grid every 200 pixels
    for x in range(0, w, 200):
        cv2.line(img, (x, 0), (x, h), (255, 0, 0), 2)
        cv2.putText(img, str(x), (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    for y in range(0, h, 200):
        cv2.line(img, (0, y), (w, y), (0, 0, 255), 2)
        cv2.putText(img, str(y), (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    cv2.imwrite("p3_grid.png", img)
    print(f"Grid saved to p3_grid.png ({w}x{h})")

if __name__ == "__main__":
    overlay_grid()
