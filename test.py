"""
Minimal Validation Test
Debug exactly what's happening in validation
"""

import csv

# Load dictionary
print("Loading dictionary...")
dictionary_words = set()

with open("data/dictionaries/Padakosha_kannada_csv.csv", 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    next(csv_reader, None)
    next(csv_reader, None)
    
    for row in csv_reader:
        if len(row) >= 2:
            word = row[1].strip()
            word = word.replace('¹', '').replace('²', '').replace('³', '')
            word = word.replace('⁴', '').replace('⁵', '').replace('⁶', '')
            word = word.replace('⁷', '').replace('⁸', '').replace('⁹', '')
            word = word.replace('⁰', '').strip()
            
            if word and len(word) >= 1:
                if any('\u0C80' <= c <= '\u0CFF' for c in word):
                    dictionary_words.add(word)

print(f"✅ Loaded {len(dictionary_words):,} words\n")

# Test validation logic
def strip_vibhakti(word):
    """Strip vibhakti suffixes"""
    if len(word) < 3:
        return word
    
    vibhakti_suffixes = [
        'ಗಳಿಂದ', 'ಗಳಲ್ಲಿ', 'ಗಳನ್ನು', 'ಗಳಿಗೆ',
        'ಯವರು', 'ಯವರ',
        'ಯಲ್ಲಿ', 'ಯಿಂದ', 'ಯನ್ನು',
        'ಅಲ್ಲಿ', 'ಇಂದ', 'ಅನ್ನು',
        'ಗಳು', 'ಗಳ',
        'ವರು', 'ವರ',
        'ತ್ತಾನೆ', 'ತ್ತಾಳೆ', 'ತ್ತಾರೆ',
        'ಲ್ಲಿ', 'ನ್ನು', 'ಗೆ', 'ನು', 'ಯ', 'ರ', 'ವ'
    ]
    
    for suffix in vibhakti_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)]
    
    return word

def validate_word(word):
    """Validate a word - EXACT LOGIC from enhanced_validator"""
    print(f"\n🔍 Validating: {word}")
    print("-" * 60)
    
    # Step 1: Check exact match
    is_valid = word in dictionary_words
    print(f"  Step 1 - Exact match check: {is_valid}")
    
    # Step 2: If not found, try stripping
    if not is_valid:
        stripped = strip_vibhakti(word)
        print(f"  Step 2 - Stripped word: '{stripped}' (original: '{word}')")
        print(f"  Step 2 - Are they different? {stripped != word}")
        
        if stripped != word:
            in_dict = stripped in dictionary_words
            print(f"  Step 2 - Is stripped in dict? {in_dict}")
            
            if in_dict:
                is_valid = True
                print(f"  Step 2 - Setting is_valid = True")
    
    print(f"  FINAL RESULT: {is_valid}")
    print("-" * 60)
    
    return is_valid

# Test the three words
test_words = ["ಸೂರ್ಯನು", "ಬೆಳ್ಳಗೆ", "ಬರುತ್ತಾನೆ"]

print("="*60)
print("VALIDATION TEST")
print("="*60)

for word in test_words:
    result = validate_word(word)
    print(f"\n{'✅' if result else '❌'} {word}: {'VALID' if result else 'INVALID'}\n")

print("="*60)