import pytesseract
from PIL import Image
# LOCAL
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\klara.orban\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

img = Image.open(r"C:\Users\klara.orban\OneDrive - Reformierte Kirchen Bern Jura Solothurn\Dokumente\Programming\bjsChatBot\pytesseract_try.png")
text = pytesseract.image_to_string(img)

print(text)
