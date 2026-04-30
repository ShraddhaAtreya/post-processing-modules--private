"""
Debug Script - Test Collocation Matching
=========================================

Tests why ನೀರ + ಕುಡಿ matching isn't working.
"""

import pandas as pd
import csv

# Load collocation database
print("="*70)
print("DEBUGGING COLLOCATION MATCHING")
print("="*70)

db_path = "data/collocation/collocation_database.csv"
df = pd.read_csv(db_path, encoding='utf-8')

print(f"\nTotal pairs in database: {len(df):,}")

# Test 1: Check if ನೀರ exists
print("\n" + "─"*70)
print("TEST 1: Check ನೀರ in database")
print("─"*70)

neer_rows = df[df['object'] == 'ನೀರ']
print(f"Found {len(neer_rows)} verbs for 'ನೀರ':")
print(neer_rows.sort_values('score', ascending=False).head(10))

# Test 2: Test object marker stripping
print("\n" + "─"*70)
print("TEST 2: Object Marker Stripping")
print("─"*70)

test_words = ['ನೀರನ್ನು', 'ನೀರವನ್ನು', 'ಆಹಾರವನ್ನು', 'ಪುಸ್ತಕವನ್ನು']

def strip_object_marker(word):
    """Strip object markers."""
    markers = ['ನ್ನು', 'ವನ್ನು', 'ಅನ್ನು']
    for marker in markers:
        if word.endswith(marker):
            return word[:-len(marker)]
    return word

for word in test_words:
    stripped = strip_object_marker(word)
    in_db = stripped in df['object'].values
    print(f"  {word:20s} → {stripped:15s} {'✓ IN DB' if in_db else '✗ NOT IN DB'}")

# Test 3: Test verb root extraction
print("\n" + "─"*70)
print("TEST 3: Verb Root Extraction")
print("─"*70)

verb_roots = {
    'ಮಾಡು', 'ಬರೆ', 'ಓದು', 'ತಿನ್ನು', 'ಕುಡಿ', 'ಹೋಗು', 'ಬಾ', 'ಕೊಡು'
}

def extract_verb_root(word):
    """Extract verb root from conjugated verb."""
    for root in verb_roots:
        if root in word:
            return root
    return None

test_verbs = [
    'ಓದುತ್ತಿದ್ದೇನೆ',
    'ತಿನ್ನುತ್ತಿದ್ದೇನೆ', 
    'ಬರೆಯುತ್ತಿದ್ದೇನೆ',
    'ಕುಡಿಯುತ್ತಿದ್ದೇನೆ',
    'ಮಾಡುತ್ತಿದ್ದೇನೆ'
]

for verb in test_verbs:
    root = extract_verb_root(verb)
    in_db = root in df['verb'].values if root else False
    print(f"  {verb:25s} → {root if root else 'None':10s} {'✓ IN DB' if in_db else '✗ NOT IN DB'}")

# Test 4: Full matching test
print("\n" + "─"*70)
print("TEST 4: Full Object-Verb Matching")
print("─"*70)

test_cases = [
    ('ನೀರನ್ನು', 'ಬರೆಯುತ್ತಿದ್ದೇನೆ', 'ಬರೆ'),
    ('ನೀರನ್ನು', 'ಕುಡಿಯುತ್ತಿದ್ದೇನೆ', 'ಕುಡಿ'),
    ('ಆಹಾರವನ್ನು', 'ಓದುತ್ತಿದ್ದೇನೆ', 'ಓದು'),
    ('ಆಹಾರವನ್ನು', 'ತಿನ್ನುತ್ತಿದ್ದೇನೆ', 'ತಿನ್ನು'),
]

for obj_word, verb_word, expected_root in test_cases:
    obj_stripped = strip_object_marker(obj_word)
    verb_root = extract_verb_root(verb_word)
    
    # Check if pair exists
    pair_exists = False
    score = 0.0
    
    if obj_stripped in df['object'].values and verb_root:
        matches = df[(df['object'] == obj_stripped) & (df['verb'] == verb_root)]
        if len(matches) > 0:
            pair_exists = True
            score = matches.iloc[0]['score']
    
    print(f"\n  Object: {obj_word:20s} → {obj_stripped}")
    print(f"  Verb:   {verb_word:25s} → {verb_root if verb_root else 'None'}")
    print(f"  Pair:   {obj_stripped} + {verb_root if verb_root else '?'}")
    print(f"  Status: {'✓ FOUND' if pair_exists else '✗ NOT FOUND'} {f'(score: {score:.4f})' if pair_exists else ''}")

# Test 5: Why might ನೀರ + ಬರೆ have high score?
print("\n" + "─"*70)
print("TEST 5: Check Why Wrong Verbs Might Score High")
print("─"*70)

# Check ನೀರ + ಬರೆ
neer_bare = df[(df['object'] == 'ನೀರ') & (df['verb'] == 'ಬರೆ')]
if len(neer_bare) > 0:
    print(f"\n⚠️  ನೀರ + ಬರೆ EXISTS in database!")
    print(f"    Score: {neer_bare.iloc[0]['score']:.4f}")
    print(f"    This is why validator might not flag it as semantic error")
else:
    print(f"\n✓ ನೀರ + ಬರೆ does NOT exist in database (good)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nIf you see:")
print("  • ✓ ನೀರ + ಕುಡಿ FOUND with score > 0.3")
print("  • ✗ ನೀರ + ಬರೆ NOT FOUND or score < 0.05")
print("  • All object/verb extractions working")
print("\nThen the validator SHOULD work correctly!")
print("="*70)