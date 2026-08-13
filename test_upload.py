import requests
import os

url = "http://localhost:8001/upload"
file_path = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\page_3.jpg"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

files = {"file": open(file_path, "rb")}
try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
