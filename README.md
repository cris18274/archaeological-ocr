# Structure-Aware OCR for Degraded Archaeological Tables

This repository contains the official implementation of a hybrid Computer Vision and Deep Learning pipeline designed specifically for the digitization of degraded historical and archaeological tabular records.

The pipeline explicitly decouples spatial layout reconstruction from semantic recognition, utilizing morphological cell extraction and an aspect-ratio heuristic for deterministic orientation correction, successfully processing perpendicular table headers without cross-column bleeding.

## Features
- **Morphological Cell Segmentation:** Extracts physical table bounding boxes using dynamic horizontal and vertical mathematical morphology kernels, overcoming broken/faded ink lines.
- **Orientation Correction:** Automatically detects perpendicular vertical text ($R = \frac{h}{w} > 1.5$) and deterministically applies a $+90^\circ$ rotation prior to inference.
- **Cell-wise OCR Inference:** Uses PaddleOCR (CRNN + ResNet34-vd) to predict text isolated per cell, preventing spatial hallucination across adjacent columns.
- **Provenance-Aware Extraction:** Maintains original image bounding boxes mapped directly to the extracted semantic text for downstream human-in-the-loop validation.

## Repository Contents
- `ocr/ocr2.py`: The core Optical Character Recognition and morphological processing pipeline.
- `run_full_evaluation.py`: Script to evaluate the CER, Accuracy, and NED across the tabular datasets.
- `generate_paper_figures.py`: Script leveraging OpenCV and Matplotlib to visually extract segmentation plots and accuracy ablation charts.
- `extracted_paper/`: Contains the LaTeX source (`sn-article.tex`) of the doctoral-level manuscript detailing the methodology, algorithms, and comprehensive evaluation metrics.

## Requirements
To run this pipeline, you will need the following dependencies:
- Python 3.8+
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`
- `paddlepaddle` (or `paddlepaddle-gpu` for CUDA acceleration)
- `paddleocr`

You can install the core dependencies via:
```bash
pip install opencv-python numpy matplotlib paddleocr paddlepaddle
```

## Usage
1. **Running the Pipeline:**
   Provide a scanned document (e.g., `page_1.jpg`) to the OCR pipeline to reconstruct the grid and extract text.
   ```bash
   python ocr/ocr2.py
   ```
2. **Generating Evaluation Figures:**
   If you wish to visualize the morphological steps and orientation correction:
   ```bash
   python generate_paper_figures.py
   ```
   The figures will be saved in `extracted_paper/figures/`.

## Author
[Cristian Ibadango](https://github.com/cris18274)

## License
MIT License
