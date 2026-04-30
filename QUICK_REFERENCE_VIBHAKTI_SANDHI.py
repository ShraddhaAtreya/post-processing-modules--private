"""
QUICK REFERENCE: Vibhakti & Sandhi Integration
===============================================

Fast answers to common questions and tasks.
"""

# ==============================================================================
# QUICK START (5 MINUTES)
# ==============================================================================

"""
1. RUN THE INTERFACE (easiest way to see it working)

    python gradio_interface_integrated.py
    
    Then open: http://localhost:7860
    
    Paste Kannada text → Click "Analyze & Correct" → See results with vibhakti details


2. USE IN YOUR CODE

    from vibhakti_sandhi_pipeline import VibhaktiSandhiProcessor
    
    processor = VibhaktiSandhiProcessor()
    result = processor.process_sentence("ರಾಮನನ್ನು ಓದುತ್ತಿದೆ")
    
    print(result.corrected_sentence)


3. RUN TESTS

    python test_vibhakti_sandhi_integration.py
    
    Expected output: 6/6 tests passed


4. SEE EXAMPLES

    python integration_examples.py
    
    Shows 7 working examples of all features
"""


# ==============================================================================
# VIBHAKTI CASES (What vibhakti means)
# ==============================================================================

VIBHAKTI_GUIDE = """
CASE MARKER (Vibhakti) | SUFFIX | USAGE | EXAMPLE
─────────────────────────────────────────────────────────────────────────────

ಪ್ರಥಮಾ (Nominative)
  No suffix        | Subject, agent
  ರಾಮನು ಓದುತ್ತಿದೆ    | "Rama reads" (rama = subject)

ದ್ವಿತೀಯಾ (Accusative) 
  -ನ್ನು, -ವನ್ನು      | Direct object (thing acted upon)
  ರಾಮನು ಪುಸ್ತಕವನ್ನು ಓದುತ್ತಿದೆ | "Rama reads BOOK" (book = object)

ಚತುರ್ಥೀ (Dative)
  -ಕ್ಕೆ, -ಗೆ, -ಿಗೆ   | Indirect object, recipient, direction
  ರಾಮನು ಮಕ್ಕಳಿಗೆ ಪುಸ್ತಕ ಕೊಡುತ್ತಿದೆ | "Rama gives book to CHILDREN"

ತೃತೀಯಾ (Instrumental)
  -ಇಂದ, -ಿಂದ, -ಯಿಂದ  | Tool, means, cause
  ರಾಮನು ಮೊಂಡಿಂದ ಬೀಜ ಬಾಯಿಸುತ್ತಿದೆ | "Rama waters with STICK"

ಷಷ್ಠೀ (Genitive)
  -ದ, -ರ, -ನ, -ಯ   | Possession, ownership
  ರಾಮನ ಪುಸ್ತಕ       | "Rama's BOOK" (rama = possessor)

ಸಪ್ತಮೀ (Locative)
  -ಿ, -ಲ್ಲಿ, -ಅಲ್ಲಿ    | Location, place
  ಮನೆಯಲ್ಲಿ ಓದುತ್ತಿದೆ   | "Reading in HOUSE" (house = location)

ಸಂಬೋಧನೆ (Vocative)
  (usually no suffix or -ಓ) | Direct address
  ರಾಮ! ಆಗಿಬಾರು        | "Rama! Come here" (rama = addressed person)
"""


# ==============================================================================
# SANDHI RULES (How words join)
# ==============================================================================

SANDHI_GUIDE = """
SANDHI RULE | WHEN APPLIED | TRANSFORMATION | EXAMPLE
─────────────────────────────────────────────────────────────────────────────

Yakara Sandhi
  When root ends with vowel (ಆ, ಇ, ಉ, etc.)
  → Add 'ಯ' before certain suffixes
  ಪುಸ್ತಕ + ಆನು → ಪುಸ್ತಕ್ಯಾನು


Vakaragama (consonant insertion)
  When root ends with consonant
  → Add vowel (ಅ, ಆ) before suffix
  ಹೆಸರ + ಗಳು → ಹೆಸರುಗಳು


Lopa (sound dropping)
  When vowels meet
  → Drop certain sounds
  ಕ್ರಿಸ್ತುವ + ಉ → ಕ್ರಿಸ್ತೂ (ವ drops)


Ādeśa (sound substitution)
  When incompatible sounds meet
  → Replace with compatible sound
  ತ್ + ಥ → ಠ್ (replacement)
"""


# ==============================================================================
# COMMON ERRORS & FIXES
# ==============================================================================

ERROR_FIX_GUIDE = """
ERROR PATTERN | CAUSE | FIX | CONFIDENCE
──────────────────────────────────────────────────────────────────────────────

ರಾಮನನ್ನು ಹೋದನು
  rama[ACC] went
  → Wrong vibhakti
  → ರಾಮನು ಹೋದನು (use nominative)
  → HIGH - subject never takes accusative

ಪುಸ್ತಕ ಓದುತ್ತಿದೆ
  book read
  → Missing vibhakti on object
  → ಪುಸ್ತಕವನ್ನು ಓದುತ್ತಿದೆ (add accusative)
  → MEDIUM - context dependent

ನಾನು ಅವನಿಗೆ ಕೊಟ್ಟೆ
  I him[DAT] gave
  → Might need accusative instead
  → Could be: ನಾನು ಅವನನ್ನು ಕೊಟ್ಟೆ
  → LOW - needs more context

ಮನೆ ಹೋದೆ
  house went
  → Might be missing locative
  → ಮನೆಗೆ ಹೋದೆ (to house)
  → MEDIUM - verb determines correctness
"""


# ==============================================================================
# API REFERENCE (Key functions)
# ==============================================================================

API_REFERENCE = """
MODULE: vibhakti_sandhi_pipeline

VibhaktiChecker
──────────────────────────────────────────────────────────────────────────────
  analyze_word(word: str) → VibhaktiAnalysis
    Detect vibhakti in word
    Returns: detected_vibhakti, root_word, case_type

  validate_vibhakti(word, role, context) → VibhaktiAnalysis
    Check if vibhakti matches grammatical role
    Returns: action, confidence, corrected_vibhakti

  strip_vibhakti(word) → (root, vibhakti)
    Remove vibhakti from word
    Returns: root_word, vibhakti_suffix

  reattach_vibhakti(root, vibhakti) → word
    Add vibhakti back with sandhi rules
    Returns: final_word_form


SandhiHandler
──────────────────────────────────────────────────────────────────────────────
  apply_sandhi(root, suffix) → (action, final_form, rule_name)
    Join root and suffix with sandhi
    Returns: SandhiAction, final_word, rule_applied

  split_word_candidates(word) → [components] or None
    Try to split incorrectly joined word
    Returns: list of components or None


VibhaktiSandhiProcessor
──────────────────────────────────────────────────────────────────────────────
  process_word(word, role, context) → WordCorrection
    Complete analysis of single word
    Returns: WordCorrection with vibhakti & sandhi details

  process_sentence(text, roles, context) → SentenceCorrection
    Complete analysis of entire sentence
    Returns: SentenceCorrection with all word details


ENUMS
──────────────────────────────────────────────────────────────────────────────
  VibhaktiAction: RETAINED, CORRECTED, REATTACHED, STRIPPED, INCOMPATIBLE
  SandhiAction: SPLIT, APPLIED, NONE
  ConfidenceLevel: HIGH, MEDIUM, LOW, AMBIGUOUS
"""


# ==============================================================================
# COMMON TASKS
# ==============================================================================

COMMON_TASKS = """
TASK 1: Detect vibhakti in a word
────────────────────────────────────────────────────────────────────────────
  from vibhakti_sandhi_pipeline import VibhaktiChecker
  
  checker = VibhaktiChecker()
  analysis = checker.analyze_word("ರಾಮನನ್ನು")
  
  print(analysis.detected_vibhakti)    # "ನ್ನು"
  print(analysis.root_word)             # "ರಾಮ"
  print(analysis.vibhakti_case)        # "ದ್ವಿತೀಯಾ (Accusative)"


TASK 2: Check if vibhakti is correct for a role
────────────────────────────────────────────────────────────────────────────
  validation = checker.validate_vibhakti("ರಾಮನನ್ನು", "subject", "unknown")
  
  print(validation.action)              # "STRIPPED"
  print(validation.confidence.value)   # "high"
  print(validation.action_reason)      # "Subject should be nominative..."


TASK 3: Apply sandhi when attaching a suffix
────────────────────────────────────────────────────────────────────────────
  from vibhakti_sandhi_pipeline import SandhiHandler
  
  handler = SandhiHandler()
  action, final_form, rule = handler.apply_sandhi("ಪುಸ್ತಕ", "ವನ್ನು")
  
  print(final_form)     # "ಪುಸ್ತಕವನ್ನು"
  print(rule)           # "vakaragama"


TASK 4: Correct an entire sentence
────────────────────────────────────────────────────────────────────────────
  from vibhakti_sandhi_pipeline import VibhaktiSandhiProcessor
  
  processor = VibhaktiSandhiProcessor()
  
  result = processor.process_sentence(
      "ರಾಮನನ್ನು ಓದುತ್ತಿದೆ",
      word_roles={"ರಾಮನನ್ನು": "subject"},
      context_type="action"
  )
  
  print(result.corrected_sentence)  # "ರಾಮನು ಓದುತ್ತಿದೆ"


TASK 5: Get detailed analysis for each word
────────────────────────────────────────────────────────────────────────────
  result = processor.process_sentence("ರಾಮನನ್ನು ಓದುತ್ತಿದೆ")
  
  for word_corr in result.word_corrections:
      print(f"Original: {word_corr.original}")
      print(f"Corrected: {word_corr.corrected}")
      
      if word_corr.vibhakti_analysis:
          va = word_corr.vibhakti_analysis
          print(f"  Vibhakti action: {va.action.value}")
          print(f"  Confidence: {va.confidence.value}")


TASK 6: Use with your own validator
────────────────────────────────────────────────────────────────────────────
  from validation import KannadaWordValidator
  from vibhakti_sandhi_pipeline import VibhaktiSandhiProcessor
  
  validator = KannadaWordValidator()
  processor = VibhaktiSandhiProcessor(validator)
  
  result = processor.process_sentence("ರಾಮನು ಪುಸ್ತಕ ಓದುತ್ತಿದೆ")


TASK 7: Export results to JSON
────────────────────────────────────────────────────────────────────────────
  import json
  
  result = processor.process_sentence("ರಾಮನು ಓದುತ್ತಿದೆ")
  
  # Convert to dict
  result_dict = result.to_dict()
  
  # Serialize to JSON
  json_str = json.dumps(result_dict, ensure_ascii=False)
  
  # Save or send
  with open("result.json", "w", encoding="utf-8") as f:
      f.write(json_str)
"""


# ==============================================================================
# CONFIDENCE GUIDE
# ==============================================================================

CONFIDENCE_GUIDE = """
CONFIDENCE LEVEL | THRESHOLD | WHAT TO DO | EXAMPLE
──────────────────────────────────────────────────────────────────────────────

HIGH (>80%)
  ✓ Auto-correct immediately
  ✓ No user review needed
  → ರಾಮನನ್ನು (subject with ACC)
     Should be ರಾಮನು (subject should be NOM)

MEDIUM (50-80%)
  ⚠️ Present as suggestion
  ⚠️ User should review
  → ಪುಸ್ತಕ (object without vibhakti)
     Might need ಪುಸ್ತಕವನ್ನು (if direct object)

LOW (<50%)
  ❌ Flag for manual review
  ❌ Show all possible options
  → ನಾನು ಅವನಿಗೆ (could be dative or accusative)
     Need full context to decide

AMBIGUOUS
  ❌ Multiple valid options exist
  ❌ Show alternatives with explanations
  → ಇದು ಚೆಲ್ಲಾರ
     Could be different meanings depending on intonation
"""


# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

TROUBLESHOOTING = """
PROBLEM | SOLUTION | VERIFY
──────────────────────────────────────────────────────────────────────────────

"ModuleNotFoundError: vibhakti_sandhi_pipeline"
  → vibhakti_sandhi_pipeline.py not in path
  → Add to same directory as your script
  python -c "from vibhakti_sandhi_pipeline import VibhaktiChecker; print('OK')"

"ImportError: sandi_farmation"
  → sandi_farmation.py not found
  → System falls back to simple concatenation (still works)
  → Optional: ensure sandi_farmation.py is in Python path

Low confidence for all words
  → Context might be ambiguous
  → Try providing complete sentences
  → Use explicit word_roles parameter
  
Vibhakti not detected
  → Suffix might not be in vibhakti_suffixes list
  → Check: from vibhakti_sandhi_pipeline import vibhakti_suffixes
  → Add missing suffix: vibhakti_suffixes.append('ನೀ')

UI not showing correctly
  → Ensure Gradio is installed: pip install gradio
  → Check browser console for errors
  → Try different browser (Chrome/Firefox)
  → Check Gradio version: python -c "import gradio; print(gradio.__version__)"

Sandhi not applied
  → Might be no sandhi needed for that combination
  → Check SandhiHandler logs
  → Verify sandi_farmation.py is available
"""


# ==============================================================================
# FILE LOCATIONS
# ==============================================================================

FILE_GUIDE = """
Core Module:
  d:\\Interface_gradio\\vibhakti_sandhi_pipeline.py
    Main classes: VibhaktiChecker, SandhiHandler, VibhaktiSandhiProcessor

Enhanced Interface:
  d:\\Interface_gradio\\gradio_interface_integrated.py
    Run this to launch the web UI

Tests:
  d:\\Interface_gradio\\test_vibhakti_sandhi_integration.py
    6 comprehensive test modules

Examples:
  d:\\Interface_gradio\\integration_examples.py
    7 working examples of all features

Documentation:
  d:\\Interface_gradio\\README_VIBHAKTI_SANDHI.md
    User guide and quick reference
  
  d:\\Interface_gradio\\VIBHAKTI_SANDHI_INTEGRATION_GUIDE.md
    Technical documentation
  
  d:\\Interface_gradio\\DELIVERY_SUMMARY_VIBHAKTI_SANDHI.md
    Delivery details and summary

Dependencies (already in project):
  d:\\Interface_gradio\\context_based_suggestion_para.py
    Existing vibhakti and sandhi logic
  
  d:\\Interface_gradio\\sandi_farmation.py
    Existing sandhi rules
  
  d:\\Interface_gradio\\validation.py
    Word validation
  
  d:\\Interface_gradio\\correction_pipeline.py
    Main correction pipeline
"""


# ==============================================================================
# KEYBOARD SHORTCUTS & TIPS
# ==============================================================================

TIPS = """
TIPS FOR BEST RESULTS
──────────────────────────────────────────────────────────────────────────────

1. Provide complete sentences
   ✓ "ರಾಮನು ಪುಸ್ತಕವನ್ನು ಓದುತ್ತಿದೆ" (complete)
   ✗ "ರಾಮನು ಓದುತ್ತಿದೆ" (incomplete - missing object)

2. Include verbs
   ✓ Verbs help determine roles and vibhakti
   ✗ Fragments without verbs are harder to analyze

3. Review suggestions, not auto-corrections
   ✓ Always review MEDIUM confidence corrections
   ✓ Medium suggestions often depend on context

4. Use the UI for exploration
   ✓ See detailed explanations for each word
   ✓ Toggle vibhakti/sandhi details on/off
   ✓ Review confidence levels

5. Check examples
   ✓ Run integration_examples.py to see how features work
   ✓ Modify examples to test your own sentences

KEYBOARD SHORTCUTS (in Gradio UI)
──────────────────────────────────────────────────────────────────────────────
  Enter in text field → Auto-focus to next field
  Shift+Enter         → New line in text area
  Ctrl+A              → Select all text
"""


# ==============================================================================
# PRINT GUIDE
# ==============================================================================

def print_all_guides():
    """Print all quick reference guides"""
    print("="*80)
    print("QUICK REFERENCE GUIDE: Vibhakti & Sandhi Integration")
    print("="*80)
    
    print("\n" + VIBHAKTI_GUIDE)
    print("\n" + SANDHI_GUIDE)
    print("\n" + ERROR_FIX_GUIDE)
    print("\n" + API_REFERENCE)
    print("\n" + COMMON_TASKS)
    print("\n" + CONFIDENCE_GUIDE)
    print("\n" + TROUBLESHOOTING)
    print("\n" + FILE_GUIDE)
    print("\n" + TIPS)


if __name__ == "__main__":
    print_all_guides()
