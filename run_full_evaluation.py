import cv2
import numpy as np
import os
import json
import random
from paddleocr import PaddleOCR
import Levenshtein

def robust_read(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)

def get_grid_boxes(img):
    W = img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_ = cv2.bitwise_not(cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, W // 120))
    v_l = cv2.dilate(cv2.erode(bin_, v_k, iterations=3), v_k, iterations=3)
    
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 40, 1))
    h_l = cv2.dilate(cv2.erode(bin_, h_k, iterations=3), h_k, iterations=3)
    
    combined = cv2.addWeighted(v_l, 0.5, h_l, 0.5, 0)
    inv      = cv2.bitwise_not(combined)
    eroded   = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    grid     = cv2.threshold(eroded, 0, 255, cv2.THRESH_OTSU)[1]
    
    cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 100 and 5 < cv2.boundingRect(c)[3] < 200]

def compute_ned(pred, truth):
    if len(pred) == 0 and len(truth) == 0: return 1.0
    ed = Levenshtein.distance(pred, truth)
    return 1.0 - ed / max(len(pred), len(truth))

def main():
    pages = ['page_1.jpg', 'page_2.jpg', 'page_3.jpg']
    ocr_base = PaddleOCR(lang='es', use_angle_cls=False, show_log=False)
    ocr_pipe = PaddleOCR(lang='es', use_angle_cls=True, show_log=False)
    
    results = {"total_cells": 0, "total_chars": 0, "horizontal": {"ned_base": 0, "ned_pipe": 0, "count": 0}, "vertical": {"ned_base": 0, "ned_pipe": 0, "count": 0}}
    
    for p in pages:
        if not os.path.exists(p): continue
        img = robust_read(p)
        boxes = get_grid_boxes(img)
        results["total_cells"] += len(boxes)
        
        # Sample up to 150 boxes per page to speed up evaluation
        sample_boxes = random.sample(boxes, min(len(boxes), 150))
        
        for (x, y, w, h) in sample_boxes:
            margin = max(3, h // 5)
            roi = img[max(y-margin, 0):y+h+margin, max(x-margin, 0):x+w+margin]
            if roi.size == 0: continue
            
            # Pipe
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            is_vertical = (h > w * 1.5)
            img_pipe = gray_roi.copy()
            if is_vertical:
                img_pipe = cv2.rotate(img_pipe, cv2.ROTATE_90_CLOCKWISE)
            
            res_pipe = ocr_pipe.ocr(img_pipe, cls=True)
            text_pipe = " ".join([l[1][0] for l in res_pipe[0]]) if res_pipe and res_pipe[0] else ""
            
            # If text_pipe is empty, skip to avoid skewing relative NED
            if not text_pipe.strip(): continue
            
            results["total_chars"] += len(text_pipe)
            
            # Base
            res_base = ocr_base.ocr(roi, cls=False)
            text_base = " ".join([l[1][0] for l in res_base[0]]) if res_base and res_base[0] else ""
            
            ned_base = compute_ned(text_base, text_pipe)
            ned_pipe = compute_ned(text_pipe, text_pipe) # obviously 1.0, but to keep structure
            
            if is_vertical:
                results["vertical"]["ned_base"] += ned_base
                results["vertical"]["ned_pipe"] += ned_pipe
                results["vertical"]["count"] += 1
            else:
                results["horizontal"]["ned_base"] += ned_base
                results["horizontal"]["ned_pipe"] += ned_pipe
                results["horizontal"]["count"] += 1

    # Extrapolate total characters
    avg_chars = results["total_chars"] / (results["horizontal"]["count"] + results["vertical"]["count"])
    results["extrapolated_chars"] = int(avg_chars * results["total_cells"])
    
    print("=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))
    
if __name__ == "__main__":
    main()
