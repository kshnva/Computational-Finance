import nbformat

# Load notebook
with open("2-2.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Extract all code cells
code_cells = [cell['source'] for cell in nb.cells if cell['cell_type'] == 'code']

# Optionally print or save them
for i, code in enumerate(code_cells):
    print(f"# Cell {i+1}\n{code}\n")
