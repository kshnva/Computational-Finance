import nbformat

# Load the notebook
with open("1-1.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Extract and print all outputs
for cell in nb.cells:
    if cell.cell_type == "code":
        for output in cell.get("outputs", []):
            if "text" in output:
                print(output["text"])
            elif "data" in output and "text/plain" in output["data"]:
                print(output["data"]["text/plain"])