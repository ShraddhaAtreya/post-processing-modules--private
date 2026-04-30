"""
INTEGRATION EXAMPLE: Using Vibhakti & Sandhi Pipeline
======================================================

This script demonstrates how to use the integrated vibhakti and sandhi
pipeline in your correction workflow.
"""

import json
from typing import Dict, List, Optional


class IntegrationExample:
    """
    Comprehensive example of using the integrated pipeline.
    """
    
    @staticmethod
    def example_1_basic_word_correction():
        """
        Example 1: Correct a single word with vibhakti analysis
        """
        print("\n" + "="*70)
        print("EXAMPLE 1: Basic Word Correction with Vibhakti Analysis")
        print("="*70)
        
        from vibhakti_sandhi_pipeline import VibhaktiChecker
        
        checker = VibhaktiChecker()
        
        # Word with wrong vibhakti
        word = "ರಾಮನನ್ನು"  # Rama with accusative (wrong for subject)
        role = "subject"
        
        print(f"\n📝 Correcting word: {word}")
        print(f"   Role in sentence: {role}")
        
        # Analyze
        analysis = checker.validate_vibhakti(word, role, "unknown")
        
        print(f"\n   Analysis:")
        print(f"   └─ Detected vibhakti: {analysis.detected_vibhakti}")
        print(f"   └─ Root word: {analysis.root_word}")
        print(f"   └─ Action: {analysis.action.value}")
        print(f"   └─ Reason: {analysis.action_reason}")
        print(f"   └─ Confidence: {analysis.confidence.value}")
        
        # If correction needed
        if analysis.corrected_vibhakti is not None:
            corrected_word = analysis.root_word + (analysis.corrected_vibhakti or "")
            print(f"\n   ✓ Correction: {word} → {corrected_word}")
        else:
            print(f"\n   ✓ Correction: Strip '{analysis.detected_vibhakti}' (no suffix for nominative)")
    
    
    @staticmethod
    def example_2_sandhi_application():
        """
        Example 2: Apply sandhi rules when attaching suffixes
        """
        print("\n" + "="*70)
        print("EXAMPLE 2: Sandhi Application During Suffix Attachment")
        print("="*70)
        
        from vibhakti_sandhi_pipeline import SandhiHandler
        
        handler = SandhiHandler()
        
        test_cases = [
            ("ಪುಸ್ತಕ", "ವನ್ನು", "Noun + accusative"),
            ("ರಾಮ", "ುತ್ತಿದೆ", "Verb + present continuous"),
        ]
        
        for root, suffix, description in test_cases:
            print(f"\n🔗 {description}")
            print(f"   Root: {root}")
            print(f"   Suffix: {suffix}")
            
            action, final_form, rule = handler.apply_sandhi(root, suffix)
            
            print(f"   Result: {final_form}")
            print(f"   Sandhi applied: {rule if rule else 'None (simple concatenation)'}")
    
    
    @staticmethod
    def example_3_sentence_processing():
        """
        Example 3: Process a complete sentence with all stages
        """
        print("\n" + "="*70)
        print("EXAMPLE 3: Complete Sentence Processing")
        print("="*70)
        
        from vibhakti_sandhi_pipeline import VibhaktiSandhiProcessor
        
        processor = VibhaktiSandhiProcessor()
        
        # Original sentence with errors
        sentence = "ರಾಮನನ್ನು ಪುಸ್ತಕ ಓದುತ್ತಿದೆ"  # Wrong: rama is subject, not object
        
        print(f"\n📖 Original sentence: {sentence}")
        
        # Define word roles
        word_roles = {
            "ರಾಮನನ್ನು": "subject",       # Subject should be nominative
            "ಪುಸ್ತಕ": "direct_object",  # Object should be accusative
            "ಓದುತ್ತಿದೆ": "verb"
        }
        
        # Process
        result = processor.process_sentence(
            sentence,
            word_roles=word_roles,
            context_type="action"
        )
        
        print(f"✓ Corrected sentence: {result.corrected_sentence}")
        
        # Show details for each word
        print(f"\n📝 Word-by-word analysis:")
        for i, wc in enumerate(result.word_corrections, 1):
            print(f"\n   Word {i}: {wc.original}")
            
            if wc.original != wc.corrected:
                print(f"   └─ Corrected to: {wc.corrected}")
            
            if wc.vibhakti_analysis:
                va = wc.vibhakti_analysis
                print(f"   └─ Vibhakti action: {va.action.value}")
                print(f"   └─ Confidence: {va.confidence.value}")
    
    
    @staticmethod
    def example_4_ambiguous_context():
        """
        Example 4: Handle ambiguous cases (suggestions, not corrections)
        """
        print("\n" + "="*70)
        print("EXAMPLE 4: Handling Ambiguous Cases")
        print("="*70)
        
        from vibhakti_sandhi_pipeline import VibhaktiChecker, ConfidenceLevel
        
        checker = VibhaktiChecker()
        
        # Sentence fragment that could have multiple interpretations
        word = "ಅವನಿಗೆ"  # Dative: could mean "to him" or "for him"
        
        print(f"\n⚠️  Word with ambiguous interpretation: {word}")
        print(f"   Could mean: 'to him' (dative) or 'for him' (dative/purpose)")
        
        # Analyze with different roles
        roles = ["indirect_object", "beneficiary"]
        
        for role in roles:
            analysis = checker.validate_vibhakti(word, role, "unknown")
            
            print(f"\n   If role is '{role}':")
            print(f"   └─ Confidence: {analysis.confidence.value}")
            print(f"   └─ Action: {analysis.action.value}")
            
            # Only auto-correct if HIGH confidence
            if analysis.confidence == ConfidenceLevel.HIGH:
                print(f"   └─ ✓ Auto-correct (high confidence)")
            elif analysis.confidence == ConfidenceLevel.MEDIUM:
                print(f"   └─ ⚠  Suggest for review (medium confidence)")
            else:
                print(f"   └─ ❌ Flag for manual review (low confidence)")
    
    
    @staticmethod
    def example_5_ui_integration():
        """
        Example 5: Format results for UI display
        """
        print("\n" + "="*70)
        print("EXAMPLE 5: Formatting for UI Display")
        print("="*70)
        
        from vibhakti_sandhi_pipeline import (
            VibhaktiSandhiProcessor,
            format_word_correction_display
        )
        
        processor = VibhaktiSandhiProcessor()
        
        # Single word
        word = "ರಾಮನನ್ನು"
        correction = processor.process_word(word, "subject", "unknown")
        
        print(f"\n📱 Formatting for UI: {word}")
        
        # Format for display
        display = format_word_correction_display(correction)
        
        print(f"\n   Display format (ready for UI):")
        for key, value in display.items():
            if value and isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     └─ {k}: {v}")
            elif value:
                print(f"   {key}: {value}")
    
    
    @staticmethod
    def example_6_batch_processing():
        """
        Example 6: Process multiple sentences in batch
        """
        print("\n" + "="*70)
        print("EXAMPLE 6: Batch Processing Multiple Sentences")
        print("="*70)
        
        from vibhakti_sandhi_pipeline import VibhaktiSandhiProcessor
        
        processor = VibhaktiSandhiProcessor()
        
        sentences = [
            "ರಾಮನು ಪುಸ್ತಕ ಓದುತ್ತಿದೆ",
            "ನಾನು ಅವನನ್ನು ಪೇಟೆಯಿಂದ ತಗೆದುಕೊಂಡೆ",
            "ಮನೆಯಲ್ಲಿ ತಾಯಿ ಅಡುಕೆ ಮಾಡುತ್ತಿದೆ",
        ]
        
        print(f"\n📚 Processing {len(sentences)} sentences:")
        
        results = []
        for i, sentence in enumerate(sentences, 1):
            result = processor.process_sentence(sentence, context_type="action")
            results.append(result)
            
            print(f"\n   {i}. {sentence}")
            print(f"      → {result.corrected_sentence}")
            
            # Count changes
            changes = sum(1 for wc in result.word_corrections if wc.corrected != wc.original)
            print(f"      Changes: {changes} word(s)")
        
        # Summary
        total_words = sum(len(r.word_corrections) for r in results)
        total_changes = sum(
            sum(1 for wc in r.word_corrections if wc.corrected != wc.original)
            for r in results
        )
        
        print(f"\n📊 Summary:")
        print(f"   Total words processed: {total_words}")
        print(f"   Total corrections: {total_changes}")
        print(f"   Correction rate: {total_changes/total_words*100:.1f}%")
    
    
    @staticmethod
    def example_7_error_handling():
        """
        Example 7: Error handling and edge cases
        """
        print("\n" + "="*70)
        print("EXAMPLE 7: Error Handling & Edge Cases")
        print("="*70)
        
        from vibhakti_sandhi_pipeline import VibhaktiChecker, VibhaktiSandhiProcessor
        
        checker = VibhaktiChecker()
        processor = VibhaktiSandhiProcessor()
        
        test_cases = [
            ("", "Empty word"),
            ("ರಾಮ", "Word without vibhakti"),
            ("ರಾಮನುರಾಮನು", "Repeated word"),
            ("12345", "Non-Kannada characters"),
        ]
        
        for word, description in test_cases:
            print(f"\n🔍 {description}: '{word}'")
            
            try:
                if word:
                    analysis = checker.analyze_word(word)
                    print(f"   Root: {analysis.root_word}")
                    print(f"   Detected vibhakti: {analysis.detected_vibhakti or 'None'}")
                    print(f"   ✓ Handled successfully")
                else:
                    print(f"   (Empty input - handled gracefully)")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
    
    
    @staticmethod
    def run_all_examples():
        """Run all examples"""
        print("\n" + "#"*70)
        print("# VIBHAKTI & SANDHI INTEGRATION EXAMPLES")
        print("#"*70)
        
        examples = [
            IntegrationExample.example_1_basic_word_correction,
            IntegrationExample.example_2_sandhi_application,
            IntegrationExample.example_3_sentence_processing,
            IntegrationExample.example_4_ambiguous_context,
            IntegrationExample.example_5_ui_integration,
            IntegrationExample.example_6_batch_processing,
            IntegrationExample.example_7_error_handling,
        ]
        
        for example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"\n✗ Example failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "#"*70)
        print("# EXAMPLES COMPLETE")
        print("#"*70)


if __name__ == "__main__":
    IntegrationExample.run_all_examples()
