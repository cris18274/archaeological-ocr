import os
import sys

rec_model_dir = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\custom_models\archaeo_rec_v1\inference"
print(f"Cargando modelo custom desde: {rec_model_dir}")

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        rec_model_dir=rec_model_dir,
        use_angle_cls=True,
        lang='es',
        use_gpu=False,
        show_log=True
    )
    print("¡Modelo cargado exitosamente en memoria con los pesos customizados!")
except Exception as e:
    print("Error al inicializar PaddleOCR con pesos custom:")
    import traceback
    traceback.print_exc()
