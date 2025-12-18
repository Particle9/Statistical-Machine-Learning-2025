c = get_config()
c.PDFExporter.latex_command = ['/Library/tex/texbin/pdflatex','{filename}', '-quiet']