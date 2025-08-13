import PyPDF2

with open("11.010-1-1.de.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("✅ PDF text saved to output.txt")
