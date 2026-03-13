import json
import sys

def extract_code(ipynb_file):
    try:
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        code = []
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'code':
                source = "".join(cell.get('source', []))
                code.append(source)
        return "\n\n# --- CELL ---\n".join(code)
    except Exception as e:
        return str(e)

print(f"--- {sys.argv[1]} ---")
print(extract_code(sys.argv[1]))