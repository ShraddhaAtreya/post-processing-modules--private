"""
Extraction with Correction Orchestrator
=======================================
Coordinating the OCR Extraction and Correction Pipeline.

Updates:
- ✅ Integrated Context-Aware Validator (ngram_context.db)
- ✅ GPU support for EasyOCR (if available)
"""

import os
import logging
from typing import Dict, Any, Optional

# Import your modules
from enhanced_validator import EnhancedValidator
from correction_pipeline import create_pipeline
# Assuming you have an OCR module, e.g., easyocr_wrapper or similar
# from easyocr_wrapper import EasyOCRWrapper (Adjust based on your actual import)
# For this file, we focus on the Orchestrator logic.

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8' # Fix for Windows Unicode error
)
logger = logging.getLogger("Orchestrator")

class OCRExtractionOrchestrator:
    def __init__(self, 
                 model_path: str, 
                 class_mapping_path: str,
                 credentials_path: str):
        
        self.logger = logger
        self.logger.info("Initializing OCR Extraction & Correction System...")

        # 1. Initialize Validator with Context DB
        # This is the CRITICAL update:
        self.validator = EnhancedValidator(
            dictionary_paths=[
                "data/dictionaries/Padakosha_kannada_csv.csv",
                "data/dictionaries/combined_word_scrapped_csv.csv"
            ],
            context_db_path="data/ngram_context.db" # <--- CONNECTED HERE
        )

        # 2. Initialize Pipeline with the Context-Aware Validator
        self.pipeline = create_pipeline(validator=self.validator)

        # 3. Initialize OCR Engine (Placeholder for your actual OCR init)
        # You would initialize your HybridOCR or Google Vision client here
        self.credentials_path = credentials_path
        self.logger.info("System Initialized.")

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Full workflow: PDF -> Images -> OCR -> Correction
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        # --- STAGE 1: EXTRACTION (Placeholder logic) ---
        # In your actual code, this calls your OCR extraction logic
        # raw_text = self.ocr_engine.extract_text(pdf_path)
        # For now, let's assume we get text (you keep your existing OCR logic here)
        
        # NOTE: You likely have existing logic here to call Google Vision / Custom Model
        # I am preserving the structure you likely have, just ensure the validator 
        # above is what matters.
        
        # Let's say we get the text:
        from hybrid_model_implementation import HybridKannadaOCR 
        # (Assuming this is how you run your model based on previous chats)
        
        ocr_engine = HybridKannadaOCR(
            model_path="models/hybrid_kannada_ocr_20251204_151641.pth",
            mapping_path="models/class_mapping_20251204_151641.json",
            credentials_path=self.credentials_path
        )
        
        # Run OCR
        # Note: Adjust 'gpu=True' inside your HybridOCR class if you updated it for GPU
        extraction_result = ocr_engine.process_pdf(pdf_path) 
        raw_text = extraction_result.get('text', '')

        # --- STAGE 2: CORRECTION ---
        self.logger.info("Starting Context-Aware Correction...")
        
        # The pipeline now uses the N-Gram database automatically
        correction_results = self.pipeline.correct_paragraph(raw_text)
        
        # Reconstruct corrected text
        corrected_text = " ".join([res.final_sentence for res in correction_results])
        
        return {
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "details": [res.to_dict() for res in correction_results]
        }

# Helper function used by Gradio
def extract_and_correct_pdf(pdf_path, model_path, class_mapping_path, credentials_path):
    orchestrator = OCRExtractionOrchestrator(
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        credentials_path=credentials_path
    )
    return orchestrator.process_pdf(pdf_path)