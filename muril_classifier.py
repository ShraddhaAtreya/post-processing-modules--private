# =============================================================
# train_muril.py
# Run this file to train the MuRIL Layer 3 classifier
#
# Usage:
#   python train_muril.py
#
# What it does:
#   1. Reads your Kannada corpus from data/corpus/kn.txt
#   2. Auto-labels sentences as divine / human
#   3. Downloads MuRIL model (~900MB, first run only)
#   4. Trains logistic regression classifier on embeddings
#   5. Saves trained classifier to models/
# =============================================================

import os
import sys

# -------------------------------------------------------
# STEP 0 — verify corpus file exists
# -------------------------------------------------------

CORPUS_PATH = r"F:\INTERFACE_GRADIO\data\corpus\kn.txt"

if not os.path.exists(CORPUS_PATH):
    print(f"ERROR: Corpus file not found at:")
    print(f"  {CORPUS_PATH}")
    print()
    print("Please check the path and try again.")
    sys.exit(1)

print(f"Corpus found: {CORPUS_PATH}")

# -------------------------------------------------------
# STEP 1 — load corpus sentences
# -------------------------------------------------------

print("Loading corpus...")

with open(CORPUS_PATH, encoding="utf-8") as f:
    sentences = f.read().splitlines()

# Remove empty lines
sentences = [s.strip() for s in sentences if s.strip()]

print(f"Loaded {len(sentences):,} sentences from corpus")

# -------------------------------------------------------
# STEP 2 — generate training data
# -------------------------------------------------------

from muril_classifier import generate_training_data

print()
print("Generating labeled training data...")
print("(Only sentences with clear divine/human signals are kept)")
print()

data = generate_training_data(
    corpus_sentences=sentences,
    output_path="models/honorific_training_data.json",
    min_confidence=0.80   # only high-confidence labels kept
)

print(f"Training data saved to: models/honorific_training_data.json")
print()

# -------------------------------------------------------
# STEP 3 — check we have enough data
# -------------------------------------------------------

if len(data) < 20:
    print(f"WARNING: Only {len(data)} training examples generated.")
    print("This is too few to train a reliable classifier.")
    print()
    print("Options:")
    print("  1. Lower min_confidence to 0.60 to include more sentences")
    print("  2. Add more corpus text to kn.txt")
    print()
    print("Retrying with lower confidence threshold (0.60)...")
    data = generate_training_data(
        corpus_sentences=sentences,
        output_path="models/honorific_training_data.json",
        min_confidence=0.60
    )
    print()

if len(data) < 10:
    print(f"ERROR: Still only {len(data)} examples. Cannot train.")
    print("Your corpus may not contain sentences with divine/human context words.")
    print("Consider adding religious or everyday Kannada text to your corpus.")
    sys.exit(1)

# -------------------------------------------------------
# STEP 4 — train the classifier
# -------------------------------------------------------

from muril_classifier import MuRILClassifier

print("Starting MuRIL classifier training...")
print("Note: First run downloads ~900MB MuRIL model (cached after that)")
print()

clf = MuRILClassifier()

try:
    accuracy = clf.train(
        training_data_path="models/honorific_training_data.json",
        test_size=0.2,
        save_path="models/muril_honorific_classifier.pkl"
    )
    print()
    print("=" * 50)
    print(f"Training complete!")
    print(f"Accuracy : {accuracy * 100:.1f}%")
    print(f"Saved to : models/muril_honorific_classifier.pkl")
    print("=" * 50)
    print()
    print("Layer 3 is now active.")
    print("Re-run test_honorific.py to see improved results.")

except Exception as e:
    print(f"Training failed: {e}")
    print()
    print("Common causes:")
    print("  - No internet connection for first MuRIL download")
    print("  - Not enough RAM (MuRIL needs ~4GB)")
    print("  - transformers not installed: pip install transformers sentencepiece")
    sys.exit(1)