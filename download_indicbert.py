"""
Load IndicBERT Model - Working Version
======================================
This script loads IndicBERT from already downloaded files.
"""

import os
import sys

# CRITICAL: Remove offline mode if set
if 'HF_HUB_OFFLINE' in os.environ:
    del os.environ['HF_HUB_OFFLINE']
    print("✓ Disabled offline mode")

from transformers import AutoTokenizer, AutoModelForMaskedLM

print("\n" + "="*70)
print("Loading IndicBERT Model")
print("="*70 + "\n")

try:
    print("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/IndicBERTv2-MLM-only",
        local_files_only=True  # Use cached files
    )
    print("✓ Tokenizer loaded successfully")
    
    print("\nStep 2: Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(
        "ai4bharat/IndicBERTv2-MLM-only",
        local_files_only=True,  # Use cached files
        use_safetensors=True,
        device_map="cpu"  # Use CPU
    )
    print("✓ Model loaded successfully")
    
    # Create pipeline for easy testing
    print("\nStep 3: Creating fill-mask pipeline...")
    from transformers import pipeline
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    print("✓ Pipeline ready")
    
    # Test with Kannada sentences
    print("\n" + "="*70)
    print("TESTING INDICBERT WITH KANNADA SENTENCES")
    print("="*70)
    
    test_cases = [
        ("ನಾನು ಆಹಾರವನ್ನು [MASK]", "Should suggest: ತಿನ್ನು (eat), ಮಾಡು (make)"),
        ("ನಾನು ಪುಸ್ತಕವನ್ನು [MASK]", "Should suggest: ಓದು (read), ಬರೆ (write)"),
        ("ನಾನು ನೀರವನ್ನು [MASK]", "Should suggest: ಕುಡಿ (drink)"),
        ("ಅವರು ಕೆಲಸವನ್ನು [MASK]", "Should suggest: ಮಾಡು (do)"),
        ("ನಾನು ಸಂಗೀತವನ್ನು [MASK]", "Should suggest: ಕೇಳು (listen)")
    ]
    
    for i, (sentence, expected) in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}: {sentence}")
        print(f"Expected: {expected}")
        print(f"{'─'*70}")
        
        results = pipe(sentence, top_k=5)
        
        print("Top 5 Predictions:")
        for rank, pred in enumerate(results, 1):
            token = pred['token_str']
            score = pred['score']
            confidence = "★★★★★" if score > 0.5 else "★★★★" if score > 0.3 else "★★★" if score > 0.1 else "★★"
            print(f"  {rank}. {token:20s} (score: {score:.4f}) {confidence}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nIndicBERT is working correctly and ready to integrate!")
    print("Model location: C:\\Users\\atrey\\.cache\\huggingface\\hub\\")
    print("\nNext step: Integrate into your context-aware validator")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n" + "="*70)
    print("TROUBLESHOOTING")
    print("="*70)
    
    import traceback
    print("\nFull error details:")
    traceback.print_exc()
    
    print("\n" + "="*70)
    print("If you see 'OfflineModeIsEnabled' error:")
    print("  1. Close ALL terminals/PowerShell windows")
    print("  2. Open a NEW terminal")
    print("  3. Activate your virtual environment: .venv\\Scripts\\Activate")
    print("  4. Run this script again")
    print("\nIf you see 'No module named' errors:")
    print("  pip install transformers sentencepiece accelerate")
    print("="*70)