"""Convert src/notebook.py (percent format) into an .ipynb."""
import re, sys, nbformat

src = open(sys.argv[1]).read()
nb = nbformat.v4.new_notebook()
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in header:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        nb.cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(body))
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, sys.argv[2])
