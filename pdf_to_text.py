import os
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# Paths are read from .env; hardcoded values are kept as fallbacks so the
# script still works without a .env entry if the defaults match the machine.
POPPLER_PATH   = os.getenv(
    "POPPLER_PATH",
    r"C:\poppler\poppler-26.02.0\Library\bin"
)
TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.makedirs("Data_text", exist_ok=True)

for filename in os.listdir("Data_pdfs"):
    if not filename.endswith(".pdf"):
        continue
    out_file = os.path.join("Data_text", filename.replace(".pdf", ".txt"))
    if os.path.exists(out_file):
        continue
    pages = convert_from_path(os.path.join("Data_pdfs", filename), poppler_path=POPPLER_PATH)
    text = "".join(pytesseract.image_to_string(page) for page in pages)
    with open(out_file, "w") as f:
        f.write(text)