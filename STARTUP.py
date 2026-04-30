#!/usr/bin/env python3
"""
STARTUP GUIDE - Vibhakti & Sandhi Integration
==============================================

This script provides interactive guidance for getting started with the
vibhakti and sandhi integration.

Run this to see what's available and where to start.
"""

import os
import sys


def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def print_section(text):
    print(f"\n  ➜ {text}")
    print("     " + "-"*75)


def option(num, description, command=None):
    if command:
        print(f"    [{num}] {description}")
        print(f"        Command: {command}")
    else:
        print(f"    [{num}] {description}")


def main():
    print_header("🔤 VIBHAKTI & SANDHI INTEGRATION - STARTUP GUIDE")
    
    print("""
This integration adds intelligent Kannada case marker (vibhakti) and 
phonetic rule (sandhi) handling to your OCR correction pipeline.

✓ Detects what case marker is on each word
✓ Validates whether it's appropriate for the role
✓ Corrects it with proper phonetic rules
✓ Explains everything in simple terms
✓ Never forces corrections when uncertain
    """)
    
    print_section("WHAT WOULD YOU LIKE TO DO?")
    
    print("""
FIRST TIME? START HERE:
    """)
    option(1, "See it working immediately (Gradio interface)",
           "python gradio_interface_integrated.py")
    option(2, "Read the complete index (where to find everything)",
           "cat INDEX_VIBHAKTI_SANDHI.md")
    option(3, "Print quick reference guide",
           "python QUICK_REFERENCE_VIBHAKTI_SANDHI.py")
    
    print("""
WANT TO UNDERSTAND:
    """)
    option(4, "Read user guide (quick start + examples)",
           "cat README_VIBHAKTI_SANDHI.md")
    option(5, "Read technical guide (architecture + details)",
           "cat VIBHAKTI_SANDHI_INTEGRATION_GUIDE.md")
    option(6, "See what was delivered",
           "cat DELIVERY_SUMMARY_VIBHAKTI_SANDHI.md")
    
    print("""
WANT TO TEST & VERIFY:
    """)
    option(7, "Run all tests (verify everything works)",
           "python test_vibhakti_sandhi_integration.py")
    option(8, "Run working examples (7 scenarios)",
           "python integration_examples.py")
    option(9, "Try it in Python (minimal example)")
    
    print("""
READY TO USE IN CODE:
    """)
    option(10, "See integration examples",
            "cat integration_examples.py")
    option(11, "Get help with common tasks",
             "python QUICK_REFERENCE_VIBHAKTI_SANDHI.py")
    option(12, "Read the source code",
            "cat vibhakti_sandhi_pipeline.py")
    
    print("""
NEED HELP:
    """)
    option(13, "Check troubleshooting guide")
    option(14, "Print all guides to screen",
            "python QUICK_REFERENCE_VIBHAKTI_SANDHI.py")
    option(15, "Exit")
    
    print("\n  Select option (1-15): ", end="")
    
    try:
        choice = input().strip()
        
        if choice == "1":
            print("""
    Starting Gradio interface...
    
    After it starts, open your browser to: http://localhost:7860
    
    You'll see:
    - Text input area for Kannada text
    - "Analyze & Correct" button
    - Detailed analysis with vibhakti and sandhi information
    - Example sentences to try
    
    Press Ctrl+C to stop the server
    """)
            os.system("python gradio_interface_integrated.py")
        
        elif choice == "2":
            print("\n    Opening INDEX_VIBHAKTI_SANDHI.md...\n")
            os.system("cat INDEX_VIBHAKTI_SANDHI.md | less" if os.name == "posix" else "type INDEX_VIBHAKTI_SANDHI.md")
        
        elif choice == "3":
            print("\n    Generating quick reference...\n")
            os.system("python QUICK_REFERENCE_VIBHAKTI_SANDHI.py")
        
        elif choice == "4":
            print("\n    Opening README_VIBHAKTI_SANDHI.md...\n")
            os.system("cat README_VIBHAKTI_SANDHI.md | less" if os.name == "posix" else "type README_VIBHAKTI_SANDHI.md")
        
        elif choice == "5":
            print("\n    Opening technical guide...\n")
            os.system("cat VIBHAKTI_SANDHI_INTEGRATION_GUIDE.md | less" if os.name == "posix" else "type VIBHAKTI_SANDHI_INTEGRATION_GUIDE.md")
        
        elif choice == "6":
            print("\n    Opening delivery summary...\n")
            os.system("cat DELIVERY_SUMMARY_VIBHAKTI_SANDHI.md | less" if os.name == "posix" else "type DELIVERY_SUMMARY_VIBHAKTI_SANDHI.md")
        
        elif choice == "7":
            print("\n    Running test suite...\n")
            os.system("python test_vibhakti_sandhi_integration.py")
        
        elif choice == "8":
            print("\n    Running examples...\n")
            os.system("python integration_examples.py")
        
        elif choice == "9":
            print_header("MINIMAL PYTHON EXAMPLE")
            print("""
This is the simplest way to use the system:

    from vibhakti_sandhi_pipeline import VibhaktiSandhiProcessor
    
    # Create processor
    processor = VibhaktiSandhiProcessor()
    
    # Correct a sentence
    result = processor.process_sentence("ರಾಮನನ್ನು ಪುಸ್ತಕ ಓದುತ್ತಿದೆ")
    
    # Show corrected version
    print(f"Corrected: {result.corrected_sentence}")
    # Output: ರಾಮನು ಪುಸ್ತಕ ಓದುತ್ತಿದೆ
    
    # See details for each word
    for word_corr in result.word_corrections:
        if word_corr.corrected != word_corr.original:
            print(f"{word_corr.original} → {word_corr.corrected}")

Run this? (y/n): """)
            if input().strip().lower() == "y":
                print("\nStarting Python interactive shell...\n")
                code = """
from vibhakti_sandhi_pipeline import VibhaktiSandhiProcessor

processor = VibhaktiSandhiProcessor()
result = processor.process_sentence("ರಾಮನನ್ನು ಪುಸ್ತಕ ಓದುತ್ತಿದೆ")

print(f"Original: ರಾಮನನ್ನು ಪುಸ್ತಕ ಓದುತ್ತಿದೆ")
print(f"Corrected: {result.corrected_sentence}")

for word_corr in result.word_corrections:
    if word_corr.corrected != word_corr.original:
        print(f"  {word_corr.original} → {word_corr.corrected}")
        if word_corr.vibhakti_analysis:
            print(f"    Action: {word_corr.vibhakti_analysis.action.value}")
"""
                exec(code)
        
        elif choice == "10":
            print("\n    Opening integration examples...\n")
            os.system("cat integration_examples.py | less" if os.name == "posix" else "type integration_examples.py")
        
        elif choice == "11":
            print("\n    Generating quick reference...\n")
            os.system("python QUICK_REFERENCE_VIBHAKTI_SANDHI.py | less" if os.name == "posix" else "python QUICK_REFERENCE_VIBHAKTI_SANDHI.py")
        
        elif choice == "12":
            print("\n    Opening source code...\n")
            os.system("cat vibhakti_sandhi_pipeline.py | less" if os.name == "posix" else "type vibhakti_sandhi_pipeline.py")
        
        elif choice == "13":
            print_section("TROUBLESHOOTING")
            print("""
COMMON ISSUES:

1. "ModuleNotFoundError: vibhakti_sandhi_pipeline"
   → Make sure all files are in the same directory
   → Run from the d:\\Interface_gradio\\ directory

2. "ImportError: sandi_farmation"
   → The system works without it (falls back to simple concatenation)
   → If available, it's used for better phonetic rules

3. "Gradio not installed"
   → pip install gradio

4. Low confidence for all corrections
   → Context might be ambiguous
   → Try providing complete sentences with verbs

5. Vibhakti not detected
   → Check if the suffix is in the list
   → Expand vibhakti_suffixes if needed

For more troubleshooting, see:
  - README_VIBHAKTI_SANDHI.md (Troubleshooting section)
  - QUICK_REFERENCE_VIBHAKTI_SANDHI.py (run it to see detailed reference)
            """)
        
        elif choice == "14":
            print("\n    Printing all guides...\n")
            os.system("python QUICK_REFERENCE_VIBHAKTI_SANDHI.py")
        
        elif choice == "15":
            print("\n  Goodbye!\n")
            return
        
        else:
            print("\n  Invalid option. Please try again.\n")
            main()
    
    except KeyboardInterrupt:
        print("\n\n  Goodbye!\n")
    except Exception as e:
        print(f"\n  Error: {e}\n")
        main()


if __name__ == "__main__":
    main()
