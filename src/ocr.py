import pytesseract
import cv2

def ocr_lines(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
    return [line.strip() for line in text.splitlines() if line.strip()]
