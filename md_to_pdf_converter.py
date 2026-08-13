import markdown
import os
import subprocess
import time

def convert_md_to_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 1.6; margin: 2cm; color: #333; }}
h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 20px; margin-bottom: 20px; page-break-inside: avoid; }}
th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
th {{ background-color: #f8f9fa; color: #333; font-weight: bold; }}
code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; font-family: monospace; font-size: 0.9em; }}
pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; border: 1px solid #eaecf0; }}
blockquote {{ border-left: 4px solid #007bff; padding-left: 15px; color: #666; font-style: italic; background-color: #f8f9fa; padding: 10px; }}
.alert {{ border-left: 4px solid #17a2b8; padding: 10px 15px; margin-bottom: 20px; background-color: #e2e3e5; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    
    # Escribir HTML temporal
    html_path = md_path.replace('.md', '.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Renderizando PDF a partir de {md_path} mediante Edge...")
    
    edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    if not os.path.exists(edge_path):
        print('Edge no encontrado en', edge_path)
        return

    cmd = [
        edge_path,
        '--headless',
        '--disable-gpu',
        '--run-all-compositor-stages-before-draw',
        f'--print-to-pdf={pdf_path}',
        html_path
    ]
    subprocess.run(cmd, check=True)
    time.sleep(2) # Dar tiempo para asegurar que se libera el archivo

    # Limpiar HTML
    try:
        os.remove(html_path)
    except:
        pass
    print(f"Creado con exito: {pdf_path}")

if __name__ == '__main__':
    base_dir = r"C:\Users\Estudiantes\.gemini\antigravity\brain\255e635a-7540-448b-b073-9fda326cc86e"
    out_dir = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version"
    
    informe_md = os.path.join(base_dir, "informe_ejecutivo.md")
    modelos_md = os.path.join(base_dir, "modelos_bd.md")
    
    informe_pdf = os.path.join(out_dir, "informe_ejecutivo.pdf")
    modelos_pdf = os.path.join(out_dir, "modelos_bd.pdf")
    
    convert_md_to_pdf(informe_md, informe_pdf)
    convert_md_to_pdf(modelos_md, modelos_pdf)
