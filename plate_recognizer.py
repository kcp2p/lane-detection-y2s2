"""
License plate recognition using OCR.
"""

import sys
from PIL import Image
import pytesseract

class LicensePlateRecognizer:
    """Class to handle OCR and post-processing for license plates."""
    
    def __init__(self):
        """Initialize the recognizer."""
        try:
            pytesseract.get_tesseract_version()
        except:
            print("ERROR: Tesseract OCR is not properly installed or configured.")
            print("Please install Tesseract OCR and make sure it's properly configured with pytesseract.")
            print("Installation guide: https://github.com/tesseract-ocr/tesseract")
            sys.exit(1)
    
    def recognize(self, image):
        """Run OCR on the processed license plate image.
        
        Args:
            image: Processed binary image of the plate
            
        Returns:
            str: Recognized text
        """
        pil_image = Image.fromarray(image)
        
        text = pytesseract.image_to_string(
            pil_image,
            lang='eng',
            config='--psm 6 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        
        return self.post_process(text)
    
    def post_process(self, text):
        """Clean up OCR results by removing punctuation.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
                 '#', '*', '+', '\\', '•', '~', '@', '£',
                 '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′',
                 'Â', '█', '½', '…',
                 '"', '★', '"', '–', '●', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
                 '¥', '▓', '—', '‹', '─',
                 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', ''', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '¾',
                 'Ã', '⋅', ''', '∞',
                 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'Ø',
                 '¹', '≤', '‡', '√', '«', ' ', '\n', '\t', '\r']
        
        for punct in puncts:
            if punct in text:
                text = text.replace(punct, '')
        
        return text
