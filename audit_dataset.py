import os
import json

def audit_split(file_path):
    if not os.path.exists(file_path):
        return {"cells": 0, "chars": 0}
    
    cells = 0
    chars = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cells += 1
                try:
                    # En algunos casos los labels en PaddleOCR pueden tener un JSON escapado o solo el texto
                    label_str = parts[1]
                    chars += len(label_str)
                except:
                    pass
    return {"cells": cells, "chars": chars}

def main():
    base_dir = r"d:\Proyecto de Investigación_Cristian Ibadango\ocr-version\ocr-version\dataset"
    train_file = os.path.join(base_dir, "train_list.txt")
    val_file = os.path.join(base_dir, "val_list.txt")
    
    train_stats = audit_split(train_file)
    val_stats = audit_split(val_file)
    
    total_cells = train_stats["cells"] + val_stats["cells"]
    total_chars = train_stats["chars"] + val_stats["chars"]
    
    print(f"--- Dataset Audit ---")
    print(f"Train - Cells: {train_stats['cells']}, Characters: {train_stats['chars']}")
    print(f"Val   - Cells: {val_stats['cells']}, Characters: {val_stats['chars']}")
    print(f"Total - Cells: {total_cells}, Characters: {total_chars}")
    
    # Escribiremos los resultados a un JSON temporal para que el agente pueda leerlo luego
    out_dict = {
        "train": train_stats,
        "val": val_stats,
        "total": {"cells": total_cells, "chars": total_chars}
    }
    with open("dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(out_dict, f)

if __name__ == "__main__":
    main()
