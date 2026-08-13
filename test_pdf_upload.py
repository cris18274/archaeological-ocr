import requests
import os

url = "http://localhost:8001/upload"
file_path = r"D:\Proyecto de Investigación_Cristian Ibadango\ocr-version\CATALOGO_EXPEDIENTE ARQUEOLÓGICO DEL MATERIAL CULTURAL YACHAY EP (2).pdf"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

print(f"Subiendo archivo: {file_path}")
files = {"file": open(file_path, "rb")}
try:
    # Tiempo de espera mucho más largo para 10 páginas en CPU
    response = requests.post(url, files=files, timeout=1200)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Request ID: {data['request_id']}")
        for page in data['pages']:
            print(f"Página {page['page']}: {len(page['tables'])} tablas detectadas")
            if 'error' in page:
                print(f"  Error en página: {page['error']}")
    else:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
