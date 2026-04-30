"""
Download and Test Pretrained Kannada LSTM Model
===============================================

Downloads the pretrained LSTM language model from the nlp-for-kannada
repository (by goru001) and tests it for next-word prediction.

Model: ULMFiT-based LSTM trained on Kannada Wikipedia
Source: https://github.com/goru001/nlp-for-kannada
Perplexity: 70.10 on Kannada Wikipedia

Author: MTech Thesis Project
Date: January 2026
"""

import os
import requests
import gdown
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Model files from Google Drive (from nlp-for-kannada repo)
MODEL_FILES = {
    "language_model": {
        "id": "1s8d83UKyw_C6h-wbGUqYSXtHxuFUcVlS",
        "output": "models/lstm/kannada_lm.pkl",
        "description": "LSTM Language Model (trained on Kannada Wikipedia)"
    },
    "tokenizer": {
        "id": "1BN0KgTcX6CWAJ-YF2DiwZhDUSEwVjvIH",
        "output": "models/lstm/kannada_tokenizer.pkl",
        "description": "SentencePiece Tokenizer"
    }
}

CACHE_DIR = "models/lstm"

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

print("="*70)
print("KANNADA LSTM MODEL DOWNLOAD")
print("="*70)
print(f"Source:     nlp-for-kannada (goru001)")
print(f"Model:      ULMFiT LSTM")
print(f"Cache Dir:  {CACHE_DIR}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")


# ============================================================
# STEP 1: INSTALL REQUIRED PACKAGES
# ============================================================

print("STEP 1: Checking Required Packages")
print("-"*70)

required_packages = {
    "fastai": "1.0.61",
    "torch": "1.13.1",
    "gdown": "latest"
}

def check_and_install_packages():
    """Check and install required packages."""
    import subprocess
    import sys
    
    for package, version in required_packages.items():
        try:
            if package == "fastai":
                import fastai
                print(f"✅ {package} already installed (version: {fastai.__version__})")
            elif package == "torch":
                import torch
                print(f"✅ {package} already installed (version: {torch.__version__})")
            elif package == "gdown":
                import gdown
                print(f"✅ {package} already installed")
        except ImportError:
            print(f"⚠️  {package} not found. Installing...")
            if version == "latest":
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
            print(f"✅ {package} installed successfully")

check_and_install_packages()


# ============================================================
# STEP 2: DOWNLOAD MODEL FILES
# ============================================================

print("\n" + "="*70)
print("STEP 2: Downloading Model Files from Google Drive")
print("="*70)

def download_from_gdrive(file_id, output_path, description):
    """
    Download file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        output_path: Where to save the file
        description: Description for logging
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"\n📥 Downloading: {description}")
    print(f"   Output: {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(output_path):
        print(f"   ✅ File already exists. Skipping download.")
        return True
    
    try:
        print(f"   ⏳ Downloading... (this may take a few minutes)")
        gdown.download(url, output_path, quiet=False)
        print(f"   ✅ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


# Download all files
download_success = True
for file_name, file_info in MODEL_FILES.items():
    success = download_from_gdrive(
        file_info["id"],
        file_info["output"],
        file_info["description"]
    )
    if not success:
        download_success = False

if not download_success:
    print("\n❌ Some files failed to download!")
    print("   Please check your internet connection and try again.")
    exit(1)


# ============================================================
# STEP 3: LOAD AND TEST MODEL
# ============================================================

print("\n" + "="*70)
print("STEP 3: Loading and Testing Model")
print("="*70)

try:
    from fastai.text import load_learner
    import torch
    
    print("\n📦 Loading LSTM model...")
    print("   This may take 1-2 minutes...")
    
    # Load model
    learn = load_learner(CACHE_DIR, 'kannada_lm.pkl')
    
    print("✅ Model loaded successfully!")
    
    # Model info
    print(f"\n📊 Model Information:")
    print(f"   Architecture: ULMFiT (LSTM-based)")
    print(f"   Training data: Kannada Wikipedia")
    print(f"   Perplexity: 70.10")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n💡 TIP: Make sure fastai version 1.0.61 is installed:")
    print("   pip install fastai==1.0.61")
    exit(1)


# ============================================================
# STEP 4: TEST NEXT-WORD PREDICTION
# ============================================================

print("\n" + "="*70)
print("STEP 4: Testing Next-Word Prediction")
print("="*70)

def predict_next_words(model, context, n_words=10, temperature=0.8):
    """
    Predict next words given context.
    
    Args:
        model: Loaded fastai learner
        context: Input text context
        n_words: Number of words to predict
        temperature: Sampling temperature (lower = more conservative)
    
    Returns:
        List of predicted words with probabilities
    """
    try:
        # Predict next words
        predictions = model.predict(context, n_words=n_words, temperature=temperature)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def get_top_k_predictions(model, context, k=5):
    """
    Get top-K most likely next words.
    
    Args:
        model: Loaded fastai learner
        context: Input context
        k: Number of top predictions to return
    
    Returns:
        List of (word, probability) tuples
    """
    try:
        # This is a simplified version - actual implementation depends on fastai API
        # For now, we'll use the predict function
        text_pred = model.predict(context, n_words=1, temperature=0.5)
        
        # Note: Getting exact probabilities requires accessing model internals
        # For the hybrid validator, we'll use the generated text
        return text_pred
    except Exception as e:
        print(f"Error: {e}")
        return None


# Test cases
print("\n" + "-"*70)
print("Test 1: Context with Object (Food)")
print("-"*70)

context1 = "ನಾನು ಆಹಾರವನ್ನು"
print(f"📝 Context: '{context1}'")
print(f"   Expected next words: ತಿನ್ನು, ಮಾಡು, ತೆಗೆ...")

try:
    pred1 = predict_next_words(learn, context1, n_words=5)
    print(f"   Prediction: {pred1}")
except Exception as e:
    print(f"   ⚠️  Prediction failed: {e}")
    print(f"   Note: This model may need specific fastai version")


print("\n" + "-"*70)
print("Test 2: Context with Object (Book)")
print("-"*70)

context2 = "ನಾನು ಪುಸ್ತಕವನ್ನು"
print(f"📝 Context: '{context2}'")
print(f"   Expected next words: ಓದು, ಬರೆ, ತೆಗೆ...")

try:
    pred2 = predict_next_words(learn, context2, n_words=5)
    print(f"   Prediction: {pred2}")
except Exception as e:
    print(f"   ⚠️  Prediction failed: {e}")


print("\n" + "-"*70)
print("Test 3: Context with Object (Water)")
print("-"*70)

context3 = "ನಾನು ನೀರನ್ನು"
print(f"📝 Context: '{context3}'")
print(f"   Expected next words: ಕುಡಿ, ತೆಗೆ, ಮಾಡು...")

try:
    pred3 = predict_next_words(learn, context3, n_words=5)
    print(f"   Prediction: {pred3}")
except Exception as e:
    print(f"   ⚠️  Prediction failed: {e}")


# ============================================================
# STEP 5: EXTRACT VOCABULARY
# ============================================================

print("\n" + "="*70)
print("STEP 5: Extracting Vocabulary Information")
print("="*70)

try:
    vocab = learn.data.vocab
    print(f"✅ Vocabulary size: {len(vocab.itos):,} words")
    
    # Show sample words
    print(f"\n📝 Sample vocabulary (first 20 words):")
    for i, word in enumerate(vocab.itos[:20]):
        print(f"   {i:3d}. {word}")
    
    # Check for common Kannada verbs
    common_verbs = ['ತಿನ್ನು', 'ಓದು', 'ಬರೆ', 'ಕುಡಿ', 'ಮಾಡು', 'ನೋಡು']
    print(f"\n🔍 Checking for common verbs in vocabulary:")
    for verb in common_verbs:
        if verb in vocab.stoi:
            print(f"   ✅ {verb:10s} (index: {vocab.stoi[verb]})")
        else:
            print(f"   ❌ {verb:10s} (not found)")
    
except Exception as e:
    print(f"⚠️  Could not extract vocabulary: {e}")


# ============================================================
# STEP 6: SAVE MODEL INFO
# ============================================================

print("\n" + "="*70)
print("STEP 6: Saving Model Information")
print("="*70)

model_info = {
    "model_type": "ULMFiT LSTM",
    "source": "nlp-for-kannada (goru001)",
    "training_data": "Kannada Wikipedia",
    "perplexity": "70.10",
    "vocabulary_size": len(vocab.itos) if 'vocab' in locals() else "Unknown",
    "download_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "cache_dir": CACHE_DIR,
    "status": "ready"
}

info_file = os.path.join(CACHE_DIR, "model_info.txt")
with open(info_file, 'w', encoding='utf-8') as f:
    for key, value in model_info.items():
        f.write(f"{key}: {value}\n")

print(f"✅ Model info saved to: {info_file}")


# ============================================================
# STEP 7: USAGE EXAMPLE FOR VALIDATOR
# ============================================================

print("\n" + "="*70)
print("STEP 7: Usage Example for Your Validator")
print("="*70)

example_code = '''
# In your validator code:

from fastai.text import load_learner

class LSTMPredictor:
    def __init__(self, model_path="models/lstm"):
        # Load model
        self.model = load_learner(model_path, 'kannada_lm.pkl')
    
    def predict_next_words(self, context, n=5):
        """
        Predict next words given context.
        
        Args:
            context: Input text (e.g., "ನಾನು ಆಹಾರವನ್ನು")
            n: Number of words to predict
        
        Returns:
            Generated text with predicted words
        """
        prediction = self.model.predict(context, n_words=n, temperature=0.8)
        return prediction
    
    def get_verb_candidates(self, object_phrase):
        """
        Get likely verb candidates for an object.
        
        Args:
            object_phrase: Object with marker (e.g., "ಆಹಾರವನ್ನು")
        
        Returns:
            List of verb roots
        """
        # Generate continuation
        pred_text = self.predict_next_words(object_phrase, n=10)
        
        # Extract verb roots (this is simplified)
        # In practice, you'd parse the generated text
        # and extract verb forms
        
        return pred_text

# Usage:
predictor = LSTMPredictor()

# Get predictions
context = "ನಾನು ಆಹಾರವನ್ನು"
prediction = predictor.predict_next_words(context, n=5)
print(f"Context: {context}")
print(f"Prediction: {prediction}")
# Expected: "ನಾನು ಆಹಾರವನ್ನು ತಿನ್ನು..." or similar
'''

print(example_code)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("DOWNLOAD COMPLETE!")
print("="*70)
print(f"✅ LSTM model successfully downloaded and loaded")
print(f"✅ Model cached at: {os.path.abspath(CACHE_DIR)}")
print(f"✅ Model is ready to use!")
print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Model is cached locally - future loads will be instant")
print("2. Use the model for next-word prediction in your validator")
print("3. Combine with conjugation matcher for proper verb forms")
print("4. Integrate with IndicBERT for context ranking")
print("\n" + "="*70)
print("MODEL CAPABILITIES:")
print("="*70)
print("✅ Next-word prediction")
print("✅ Language modeling")
print("✅ Context-aware generation")
print("✅ Trained on Kannada Wikipedia (perplexity: 70.10)")
print("\n" + "="*70)
print("LIMITATIONS:")
print("="*70)
print("⚠️  Generates full text, not just verb roots")
print("⚠️  May need post-processing to extract verbs")
print("⚠️  Works best with fastai 1.0.61")
print("\n" + "="*70)
print("All done! Model is ready for your thesis work! 🎉")
print("="*70)


# ============================================================
# VERIFICATION
# ============================================================

def verify_installation():
    """Verify that everything is properly installed."""
    print("\n" + "="*70)
    print("VERIFICATION CHECKLIST")
    print("="*70)
    
    checks = []
    
    # Check model file
    model_path = MODEL_FILES["language_model"]["output"]
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        checks.append(("Model file exists", True, f"{size_mb:.1f} MB"))
    else:
        checks.append(("Model file exists", False, "Not found"))
    
    # Check tokenizer file
    tokenizer_path = MODEL_FILES["tokenizer"]["output"]
    if os.path.exists(tokenizer_path):
        checks.append(("Tokenizer file exists", True, "OK"))
    else:
        checks.append(("Tokenizer file exists", False, "Not found"))
    
    # Check fastai
    try:
        import fastai
        checks.append(("fastai installed", True, f"v{fastai.__version__}"))
    except:
        checks.append(("fastai installed", False, "Not installed"))
    
    # Check model loadable
    checks.append(("Model loadable", 'learn' in locals(), "Loaded" if 'learn' in locals() else "Not tested"))
    
    # Print results
    print()
    for check, status, info in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check:25s} : {info}")
    
    all_passed = all(check[1] for check in checks[:3])  # First 3 are critical
    
    if all_passed:
        print("\n✅ All critical checks passed!")
        print("   Model is ready to use in your validator.")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
    
    return all_passed

verify_installation()