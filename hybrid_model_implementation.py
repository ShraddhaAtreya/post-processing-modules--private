"""
Hybrid Kannada OCR Implementation
=================================
- Handles PDF to Image conversion
- Runs OCR using EasyOCR (Custom Model) or Google Cloud Vision
- 🚀 GPU ENABLED for EasyOCR
"""

import os
import logging
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import easyocr
import torch

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HybridOCR")

class HybridKannadaOCR:
    def __init__(self, model_path=None, class_mapping_path=None, credentials_path=None):
        """
        Initialize OCR Engine with GPU Support.
        """
        self.use_gpu = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if self.use_gpu else "CPU"
        
        logger.info(f"🚀 Initializing OCR Engine on: {device_name}")
        
        # Initialize EasyOCR with GPU=True
        # 'kn' = Kannada, 'en' = English
        logger.info("Loading EasyOCR model...")
        self.reader = easyocr.Reader(['kn', 'en'], gpu=self.use_gpu)
        
        # Placeholder for Google Vision (if you use it later)
        self.credentials_path = credentials_path

    def process_pdf(self, pdf_path):
        """
        Convert PDF to images and extract text.
        """
        logger.info(f"📄 Processing PDF: {pdf_path}")
        
        try:
            # 1. Convert PDF to Images
            # Note: requires Poppler installed in system PATH
            images = convert_from_path(pdf_path)
            logger.info(f"Converted PDF to {len(images)} images.")
            
            full_text = []
            
            # 2. Process Each Page
            for i, image in enumerate(images):
                logger.info(f"🔍 Scanning page {i+1}/{len(images)}...")
                
                # Convert PIL image to numpy array (EasyOCR expects this)
                image_np = np.array(image)
                
                # Run OCR
                # detail=0 returns just the list of text strings
                page_result = self.reader.readtext(image_np, detail=0, paragraph=True)
                
                # Join text for the page
                page_text = " ".join(page_result)
                full_text.append(page_text)
                
            combined_text = "\n\n".join(full_text)
            
            return {
                "raw_text": combined_text,
                "corrected_text": combined_text, # Will be fixed by pipeline later
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ OCR Failed: {e}")
            # Check for common Poppler error
            if "poppler" in str(e).lower():
                logger.error("⚠️ It looks like Poppler is not installed or not in PATH.")
                return {"raw_text": "Error: Poppler is not installed. Please install Poppler for PDF conversion.", "status": "error"}
            
            return {"raw_text": "", "status": "error"}

# For testing this file alone
if __name__ == "__main__":
    ocr = HybridKannadaOCR()
    print("OCR Engine Ready.")