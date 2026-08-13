import traceback
import os
import sys

# Disabilitar checks de red para los modelos
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

print("Iniciando prueba de carga de motor...")

try:
    import paddle.base.libpaddle as libpaddle
    if not hasattr(libpaddle.AnalysisConfig, 'set_optimization_level'):
        print("Aplicando parche a AnalysisConfig.set_optimization_level")
        libpaddle.AnalysisConfig.set_optimization_level = lambda self, x: None
    
    print("Importando PPStructureV3...")
    from paddleocr import PPStructureV3
    
    print("Inicializando PPStructureV3 (lang='es') en CPU para depuración...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    engine = PPStructureV3(lang='es')
    
    print("¡Motor cargado con éxito!")
    sys.exit(0)
except Exception:
    print("--- ERROR DURANTE LA CARGA ---")
    traceback.print_exc()
    sys.exit(1)
