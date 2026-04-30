"""
Download and Test mT5 Model for Kannada
========================================

This script downloads Google's pretrained mT5 model and tests it
for Kannada text correction.

Model: mt5-small (300M parameters, ~1.2GB)
Source: Hugging Face - google/mt5-small

Author: MTech Thesis Project
Date: January 2026
"""

import os
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Model selection
MODEL_NAME = "google/mt5-small"  # Change to "google/mt5-base" for larger model
CACHE_DIR = "models/mt5"  # Local cache directory

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

print("="*70)
print("mT5 MODEL DOWNLOAD AND TEST")
print("="*70)
print(f"Model:      {MODEL_NAME}")
print(f"Cache Dir:  {CACHE_DIR}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")


# ============================================================
# STEP 1: DOWNLOAD TOKENIZER
# ============================================================

print("STEP 1: Downloading Tokenizer...")
print("-"*70)

try:
    tokenizer = MT5Tokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        local_files_only=False  # Download if not present
    )
    print("✅ Tokenizer downloaded successfully!")
    print(f"   Vocab size: {tokenizer.vocab_size:,}")
except Exception as e:
    print(f"❌ Error downloading tokenizer: {e}")
    exit(1)


# ============================================================
# STEP 2: DOWNLOAD MODEL
# ============================================================

print("\n" + "="*70)
print("STEP 2: Downloading Model...")
print("-"*70)
print("⏳ This may take 5-10 minutes depending on your internet speed...")

try:
    model = MT5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        local_files_only=False,  # Download if not present
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    print("✅ Model downloaded successfully!")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: ~{total_params * 4 / (1024**3):.2f} GB")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    exit(1)


# ============================================================
# STEP 3: TEST MODEL WITH KANNADA TEXT
# ============================================================

print("\n" + "="*70)
print("STEP 3: Testing with Kannada Text")
print("="*70)

def test_mt5_correction(error_text: str, expected: str = None):
    """
    Test mT5 for text correction.
    
    Args:
        error_text: Text with error
        expected: Expected correction (optional)
    """
    print(f"\n📝 Input:    {error_text}")
    
    # Prepare input for mT5
    # Format: "correct: <text with error>"
    input_text = f"correct: {error_text}"
    
    # Tokenize
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).input_ids
    
    # Generate correction
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=5,           # Beam search for better quality
            num_return_sequences=3, # Get top 3 suggestions
            early_stopping=True
        )
    
    # Decode outputs
    suggestions = []
    for i, output in enumerate(outputs):
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        suggestions.append(decoded)
        print(f"   Suggestion {i+1}: {decoded}")
    
    if expected:
        if expected in suggestions:
            print(f"   ✅ Expected '{expected}' found in suggestions!")
        else:
            print(f"   ⚠️  Expected '{expected}' not in top 3")
    
    return suggestions


# ============================================================
# TEST CASES
# ============================================================

print("\n" + "-"*70)
print("Test 1: Semantic Error (Food + Reading)")
print("-"*70)
test_mt5_correction(
    "ನಾನು ಆಹಾರವನ್ನು ಓದುತ್ತಿದ್ದೇನೆ",
    expected="ನಾನು ಆಹಾರವನ್ನು ತಿನ್ನುತ್ತಿದ್ದೇನೆ"
)

print("\n" + "-"*70)
print("Test 2: Semantic Error (Book + Eating)")
print("-"*70)
test_mt5_correction(
    "ನಾನು ಪುಸ್ತಕವನ್ನು ತಿನ್ನುತ್ತಿದ್ದೇನೆ",
    expected="ನಾನು ಪುಸ್ತಕವನ್ನು ಓದುತ್ತಿದ್ದೇನೆ"
)

print("\n" + "-"*70)
print("Test 3: Semantic Error (Water + Writing)")
print("-"*70)
test_mt5_correction(
    "ನಾನು ನೀರನ್ನು ಬರೆಯುತ್ತಿದ್ದೇನೆ",
    expected="ನಾನು ನೀರನ್ನು ಕುಡಿಯುತ್ತಿದ್ದೇನೆ"
)

print("\n" + "-"*70)
print("Test 4: Spelling Error")
print("-"*70)
test_mt5_correction(
    "ಕನ್ನಡಾ ಭಾಷೆ",
    expected="ಕನ್ನಡ ಭಾಷೆ"
)


# ============================================================
# STEP 4: SAVE MODEL INFO
# ============================================================

print("\n" + "="*70)
print("STEP 4: Saving Model Information")
print("="*70)

model_info = {
    "model_name": MODEL_NAME,
    "cache_dir": CACHE_DIR,
    "download_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "parameters": sum(p.numel() for p in model.parameters()),
    "vocab_size": tokenizer.vocab_size,
    "status": "ready"
}

# Save to file
info_file = os.path.join(CACHE_DIR, "model_info.txt")
with open(info_file, 'w', encoding='utf-8') as f:
    for key, value in model_info.items():
        f.write(f"{key}: {value}\n")

print(f"✅ Model info saved to: {info_file}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("DOWNLOAD COMPLETE!")
print("="*70)
print(f"✅ mT5 model successfully downloaded and tested")
print(f"✅ Model cached at: {os.path.abspath(CACHE_DIR)}")
print(f"✅ Model is ready to use!")
print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. The model is now cached locally")
print("2. Future loads will be instant (no re-download)")
print("3. Use the model in your validator:")
print()
print("   from transformers import MT5ForConditionalGeneration, MT5Tokenizer")
print(f"   model = MT5ForConditionalGeneration.from_pretrained('{MODEL_NAME}')")
print(f"   tokenizer = MT5Tokenizer.from_pretrained('{MODEL_NAME}')")
print()
print("="*70)


# ============================================================
# VERIFICATION FUNCTION
# ============================================================

def verify_model_cached():
    """Verify that model is properly cached."""
    print("\n" + "="*70)
    print("VERIFICATION: Checking Cached Files")
    print("="*70)
    
    # Check if cache directory exists
    if not os.path.exists(CACHE_DIR):
        print("❌ Cache directory not found")
        return False
    
    # Check for model files
    cache_contents = os.listdir(CACHE_DIR)
    
    if len(cache_contents) == 0:
        print("❌ Cache directory is empty")
        return False
    
    print(f"✅ Found {len(cache_contents)} cached items:")
    for item in cache_contents[:5]:  # Show first 5 items
        print(f"   • {item}")
    
    if len(cache_contents) > 5:
        print(f"   ... and {len(cache_contents) - 5} more files")
    
    return True

# Run verification
verify_model_cached()


# ============================================================
# USAGE EXAMPLE
# ============================================================

print("\n" + "="*70)
print("USAGE EXAMPLE FOR YOUR VALIDATOR")
print("="*70)

example_code = '''
# In your validator code:

from transformers import MT5ForConditionalGeneration, MT5Tokenizer

class MT5Corrector:
    def __init__(self):
        # Load model (instant if already cached)
        self.model = MT5ForConditionalGeneration.from_pretrained(
            "google/mt5-small",
            cache_dir="models/mt5"
        )
        self.tokenizer = MT5Tokenizer.from_pretrained(
            "google/mt5-small",
            cache_dir="models/mt5"
        )
    
    def correct(self, text):
        # Prepare input
        input_text = f"correct: {text}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        
        # Generate correction
        outputs = self.model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            num_return_sequences=3
        )
        
        # Decode
        suggestions = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return suggestions

# Usage:
corrector = MT5Corrector()
suggestions = corrector.correct("ನಾನು ಆಹಾರವನ್ನು ಓದುತ್ತಿದ್ದೇನೆ")
print(suggestions)  # ['ನಾನು ಆಹಾರವನ್ನು ತಿನ್ನುತ್ತಿದ್ದೇನೆ', ...]
'''

print(example_code)

print("\n" + "="*70)
print("All done! Model is ready for your thesis work!")
print("="*70)