"""
GRAMMAR INTEGRATION BRIDGE
==========================

Bridge layer that wraps 4 Kannada linguistic modules WITHOUT duplicating their logic.

This module acts as a single integration point for:
1. context_based_suggestion_para.py - Core grammar and vibhakti logic
2. pos_tagging.py - POS tagging via HMM
3. sandhi_formation.py - Phonetic combination rules
4. vibakti_checking.py - Case marker detection and manipulation

KEY DESIGN PRINCIPLES:
- NO logic duplication from the 4 modules
- Only wrapper functions that CALL the originals
- Single source of truth maintained
- Type hints and comprehensive docstrings
- Error handling for missing files/modules

Author: Grammar Integration
Date: January 5, 2026
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrammarIntegrationBridge:
    """
    Unified bridge for integrating 4 Kannada linguistic modules.
    
    Provides wrapper methods that call original functions WITHOUT duplication.
    
    Integrated Modules:
    - context_based_suggestion_para: Core grammar/vibhakti engine
    - pos_tagging: HMM-based POS tagging
    - sandhi_formation: Phonetic combination rules
    - vibakti_checking: Case marker manipulation
    """
    
    def __init__(self, 
                 verbosity: bool = False):
        """
        Initialize the grammar integration bridge.
        
        Loads all required modules and datasets. If any file is missing,
        logs a warning but doesn't crash (graceful degradation).
        
        Args:
            verbosity: Enable debug logging
            
        Raises:
            RuntimeError: If critical modules cannot be imported
        """
        self.logger = logger
        if verbosity:
            self.logger.setLevel(logging.DEBUG)
        
        self.logger.info("Initializing Grammar Integration Bridge...")
        
        # Import and cache module references
        self.context_module = None
        self.pos_tagging_module = None
        self.sandhi_module = None
        self.vibhakti_module = None
        
        # Dataset caches
        self.wordtypes = None
        self.emission_matrix = None
        self.transmission_matrix = None
        self.noun_df = None
        self.verb_df = None
        
        # Initialize modules
        self._load_modules()
        self._load_datasets()
        
        self.logger.info("Grammar Integration Bridge initialized successfully")
    
    def _load_modules(self) -> None:
        """
        Load the 4 linguistic modules.
        
        Attempts to import each module. Logs warnings for missing modules
        but doesn't crash (allows graceful degradation).
        """
        # Load context_based_suggestion_para
        try:
            import context_based_suggestion_para as context_module
            self.context_module = context_module
            self.logger.debug("Loaded context_based_suggestion_para")
        except ImportError as e:
            self.logger.warning(f"Failed to import context_based_suggestion_para: {e}")
        except Exception as e:
            self.logger.warning(f"Error loading context_based_suggestion_para: {e}")
        
        # Load pos_tagging
        try:
            import pos_tagging as pos_tagging_module
            self.pos_tagging_module = pos_tagging_module
            self.logger.debug("Loaded pos_tagging")
        except ImportError as e:
            # pos_tagging might be imported via context_based_suggestion_para
            self.logger.debug(f"pos_tagging import: {e}")
        except Exception as e:
            self.logger.warning(f"Error loading pos_tagging: {e}")
        
        # Load sandhi_formation (might be named differently)
        try:
            # Try various naming conventions
            try:
                import sandi_farmation as sandhi_module
                self.sandhi_module = sandhi_module
            except ImportError:
                import sandhi_formation as sandhi_module
                self.sandhi_module = sandhi_module
            self.logger.debug("Loaded sandhi_formation")
        except ImportError as e:
            # sandhi might be embedded in context_based_suggestion_para
            self.logger.debug(f"sandhi_formation import: {e}")
        except Exception as e:
            self.logger.warning(f"Error loading sandhi_formation: {e}")
        
        # Load vibakti_checking
        try:
            import vibakti_checking as vibhakti_module
            self.vibhakti_module = vibhakti_module
            self.logger.debug("Loaded vibakti_checking")
        except ImportError as e:
            # vibhakti functions might be in context_based_suggestion_para
            self.logger.debug(f"vibakti_checking import: {e}")
        except Exception as e:
            self.logger.warning(f"Error loading vibakti_checking: {e}")
    
    def _load_datasets(self) -> None:
        """
        Load required datasets for noun and verb processing.
        
        Looks for:
        - cleaned_kannada_dataset_noun.xlsx
        - kannada_verbs.xlsx
        - kannada_training.txt (for POS tagging)
        
        Logs warnings for missing files but doesn't crash.
        """
        try:
            import pandas as pd
            
            # Load noun dataset
            noun_path = os.path.join(os.path.dirname(__file__), "cleaned_kannada_dataset_noun.xlsx")
            if os.path.exists(noun_path):
                self.noun_df = pd.read_excel(noun_path)
                self.logger.debug(f"Loaded noun dataset: {len(self.noun_df)} rows")
            else:
                self.logger.warning(f"Noun dataset not found: {noun_path}")
            
            # Load verb dataset
            verb_path = os.path.join(os.path.dirname(__file__), "kannada_verbs.xlsx")
            if os.path.exists(verb_path):
                self.verb_df = pd.read_excel(verb_path)
                self.logger.debug(f"Loaded verb dataset: {len(self.verb_df)} rows")
            else:
                self.logger.warning(f"Verb dataset not found: {verb_path}")
        
        except ImportError:
            self.logger.warning("pandas not available; datasets cannot be loaded")
        except Exception as e:
            self.logger.warning(f"Error loading datasets: {e}")
        
        # Load POS tagging training file
        try:
            train_path = os.path.join(os.path.dirname(__file__), "kannada_training.txt")
            if os.path.exists(train_path):
                self.logger.debug(f"Found POS training file: {train_path}")
            else:
                self.logger.warning(f"POS training file not found: {train_path}")
        except Exception as e:
            self.logger.warning(f"Error checking POS training file: {e}")
    
    # ================================================================
    # SENTENCE STRUCTURE ANALYSIS
    # ================================================================
    
    def analyze_sentence_structure(self, sentence: str) -> Dict[str, Any]:
        """
        Identify subject, object, and verb in a sentence.
        
        Wraps: context_based_suggestion_para.identify_roles_kannada()
        
        Args:
            sentence: Kannada sentence to analyze
            
        Returns:
            Dict with keys:
            - 'subject': Subject word info
            - 'direct_object': Direct object word info
            - 'indirect_object': Indirect object word info
            - 'verb': Main verb word info
            - 'word_infos': List of all word info dicts
            - 'error': Error message if analysis failed
            
        Example:
            >>> result = bridge.analyze_sentence_structure("ರಾಜು ಪುಸ್ತಕ ಓದುತ್ತಾನೆ")
            >>> print(result['subject']['word'])
            'ರಾಜು'
        """
        if not sentence or not sentence.strip():
            return {
                'subject': None, 'direct_object': None, 'indirect_object': None,
                'verb': None, 'word_infos': [], 'error': 'Empty sentence'
            }
        
        if not self.context_module or not hasattr(self.context_module, 'identify_roles_kannada'):
            return {'error': 'context_module not available'}
        
        try:
            # Call original function from context_based_suggestion_para
            word_infos = self.context_module.identify_roles_kannada(
                sentence,
                wordtypes=getattr(self.context_module, 'wordtypes', None),
                emission_matrix=getattr(self.context_module, 'emission_matrix', None),
                transmission_matrix=getattr(self.context_module, 'transmission_matrix', None),
                verb_df=self.verb_df
            )
            
            # Extract specific roles
            result = {
                'word_infos': word_infos,
                'subject': self._extract_by_role(word_infos, 'subject'),
                'direct_object': self._extract_by_role(word_infos, 'direct_object'),
                'indirect_object': self._extract_by_role(word_infos, 'indirect_object'),
                'verb': self._extract_by_role(word_infos, 'verb')
            }
            
            self.logger.debug(f"Analyzed sentence: S={result['subject']}, V={result['verb']}")
            return result
        
        except Exception as e:
            self.logger.warning(f"Error analyzing sentence structure: {e}")
            return {
                'word_infos': [], 'subject': None, 'direct_object': None,
                'indirect_object': None, 'verb': None, 'error': str(e)
            }
    
    @staticmethod
    def _extract_by_role(word_infos: List[Dict], role: str) -> Optional[Dict]:
        """Extract first token with given role from word_infos list."""
        if not word_infos:
            return None
        for info in word_infos:
            if info.get('role') == role:
                return info
        return None
    
    # ================================================================
    # WORD GRAMMAR FEATURES
    # ================================================================
    
    def get_word_grammar_features(self, word: str) -> Dict[str, Any]:
        """
        Get grammatical features of a word (person, number, gender, vibhakti).
        
        Wraps: context_based_suggestion_para.analyze_kannada_word()
        
        Args:
            word: Kannada word to analyze
            
        Returns:
            Dict with keys:
            - 'person': Person (1st, 2nd, 3rd, etc.)
            - 'number': Number (singular, plural)
            - 'gender': Gender (masculine, feminine, neuter)
            - 'vibhakti': Case marker (if present)
            - 'pos_tag': Part of speech tag
            - 'error': Error message if analysis failed
            
        Example:
            >>> features = bridge.get_word_grammar_features("ರಾಜನನ್ನು")
            >>> print(features['vibhakti'])
            'Accusative'
        """
        if not word or not word.strip():
            return {'error': 'Empty word'}
        
        if not self.context_module or not hasattr(self.context_module, 'analyze_kannada_word'):
            return {'error': 'context_module not available'}
        
        try:
            # Call original function
            features = self.context_module.analyze_kannada_word(word)
            self.logger.debug(f"Analyzed word '{word}': {features}")
            return features
        
        except Exception as e:
            self.logger.warning(f"Error analyzing word features for '{word}': {e}")
            return {'error': str(e)}
    
    # ================================================================
    # GRAMMAR CORRECTION SUGGESTIONS
    # ================================================================
    
    def correct_sentence_grammar(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Get 3 grammar correction suggestions for a sentence.
        
        Wraps: context_based_suggestion_para.correct_paragraph()
        
        Args:
            sentence: Kannada sentence to correct
            
        Returns:
            List of (label, corrected_sentence) tuples
            
        Example:
            >>> suggestions = bridge.correct_sentence_grammar("ನಾನು ಪುಸ್ತಕ ಓದುತ್ತೇ")
            >>> for label, corrected in suggestions:
            ...     print(f"{label}: {corrected}")
        """
        if not sentence or not sentence.strip():
            return []
        
        if not self.context_module or not hasattr(self.context_module, 'correct_paragraph'):
            return []
        
        try:
            # Call original function
            suggestions = self.context_module.correct_paragraph(sentence)
            if not suggestions:
                suggestions = []
            self.logger.debug(f"Generated {len(suggestions)} suggestions for: {sentence}")
            return suggestions
        
        except Exception as e:
            self.logger.warning(f"Error getting grammar corrections: {e}")
            return []
    
    # ================================================================
    # VIBHAKTI (CASE MARKER) OPERATIONS
    # ================================================================
    
    def strip_word_vibhakti(self, word: str) -> str:
        """
        Remove case markers from a word.
        
        Wraps: context_based_suggestion_para.strip_existing_vibhakti()
                or vibakti_checking.strip_existing_vibhakti()
        
        Args:
            word: Kannada word with potential case marker
            
        Returns:
            Root word without case marker
            
        Example:
            >>> root = bridge.strip_word_vibhakti("ಮನೆಯಲ್ಲಿ")
            >>> print(root)
            'ಮನೆ'
        """
        if not word:
            return word
        
        try:
            # Try from context_based_suggestion_para first
            if self.context_module and hasattr(self.context_module, 'strip_existing_vibhakti'):
                return self.context_module.strip_existing_vibhakti(word)
            
            # Fall back to vibakti_checking
            if self.vibhakti_module and hasattr(self.vibhakti_module, 'strip_existing_vibhakti'):
                return self.vibhakti_module.strip_existing_vibhakti(word)
            
            self.logger.warning("No strip_vibhakti function available")
            return word
        
        except Exception as e:
            self.logger.warning(f"Error stripping vibhakti from '{word}': {e}")
            return word
    
    def attach_word_vibhakti(self, root: str, suffix: str) -> str:
        """
        Attach case marker with correct sandhi (phonetic) rules.
        
        Wraps: context_based_suggestion_para.attach_vibhakti()
                or vibakti_checking.attach_vibhakti()
        
        The original function internally uses sandhi_formation rules.
        
        Args:
            root: Root word without vibhakti
            suffix: Case marker suffix to attach
            
        Returns:
            Word with vibhakti attached and sandhi applied
            
        Example:
            >>> word = bridge.attach_word_vibhakti("ಮನೆ", "ಅಲ್ಲಿ")
            >>> print(word)
            'ಮನೆಯಲ್ಲಿ'  # Note: 'ಯ' inserted by sandhi rules
        """
        if not root or not suffix:
            return root
        
        try:
            # Try from context_based_suggestion_para first
            if self.context_module and hasattr(self.context_module, 'attach_vibhakti'):
                return self.context_module.attach_vibhakti(root, suffix)
            
            # Fall back to vibakti_checking
            if self.vibhakti_module and hasattr(self.vibhakti_module, 'attach_vibhakti'):
                return self.vibhakti_module.attach_vibhakti(root, suffix)
            
            # If no sandhi rules available, just concatenate
            self.logger.warning("No attach_vibhakti function available; returning concatenation")
            return root + suffix
        
        except Exception as e:
            self.logger.warning(f"Error attaching vibhakti to '{root}' + '{suffix}': {e}")
            return root + suffix
    
    # ================================================================
    # POS TAGGING
    # ================================================================
    
    def get_pos_tag(self, word: str) -> str:
        """
        Get part-of-speech tag for a word.
        
        Wraps: pos_tagging.decode_word_pos()
        
        Args:
            word: Kannada word to tag
            
        Returns:
            POS tag (e.g., 'NN', 'VM', 'NNP', etc.)
            
        Example:
            >>> tag = bridge.get_pos_tag("ಪುಸ್ತಕ")
            >>> print(tag)
            'NN'
        """
        if not word:
            return ''
        
        try:
            # Try from context_based_suggestion_para
            if self.context_module and hasattr(self.context_module, 'decode_word_pos'):
                return self.context_module.decode_word_pos(
                    word,
                    wordtypes=getattr(self.context_module, 'wordtypes', None),
                    emission_matrix=getattr(self.context_module, 'emission_matrix', None),
                    transmission_matrix=getattr(self.context_module, 'transmission_matrix', None)
                )
            
            # Try from pos_tagging module
            if self.pos_tagging_module and hasattr(self.pos_tagging_module, 'decode_word_pos'):
                return self.pos_tagging_module.decode_word_pos(
                    word,
                    wordtypes=getattr(self.pos_tagging_module, 'wordtypes', None),
                    emission_matrix=getattr(self.pos_tagging_module, 'emission_matrix', None),
                    transmission_matrix=getattr(self.pos_tagging_module, 'transmission_matrix', None)
                )
            
            self.logger.warning("No POS tagging function available")
            return ''
        
        except Exception as e:
            self.logger.warning(f"Error getting POS tag for '{word}': {e}")
            return ''
    
    # ================================================================
    # VERB CONJUGATION
    # ================================================================
    
    def conjugate_verb(self, 
                       verb: str,
                       person: str,
                       number: str,
                       gender: str = 'neuter',
                       tense: str = 'present',
                       form: str = 'Affirmative') -> str:
        """
        Conjugate verb to match grammatical features.
        
        Wraps: context_based_suggestion_para.extract_and_conjugate_full_sentence()
                or other conjugation functions
        
        Args:
            verb: Base verb to conjugate
            person: Person (1st, 2nd, 3rd)
            number: Number (singular, plural)
            gender: Gender (masculine, feminine, neuter)
            tense: Tense (present, past, future)
            form: Form (Affirmative, Negative, Question)
            
        Returns:
            Conjugated verb form
            
        Example:
            >>> conjugated = bridge.conjugate_verb("ಬರು", "3rd", "singular", "masculine", "present")
            >>> print(conjugated)
            'ಬರುತ್ತಾನೆ'
        """
        if not verb:
            return verb
        
        if not self.verb_df is not None:
            self.logger.warning("Verb dataset not available for conjugation")
            return verb
        
        try:
            # Call original function if available
            if self.context_module and hasattr(self.context_module, 'get_transformed_form'):
                return self.context_module.get_transformed_form(
                    verb, person, tense, gender, number, form, self.verb_df
                )
            
            self.logger.warning("No conjugation function available")
            return verb
        
        except Exception as e:
            self.logger.warning(f"Error conjugating verb '{verb}': {e}")
            return verb
    
    # ================================================================
    # HELPER/CONVENIENCE METHODS
    # ================================================================
    
    def validate_subject_verb_agreement(self, subject: str, verb: str) -> bool:
        """
        Check if subject and verb agree in person/number/gender.
        
        Uses: get_word_grammar_features() for both words
        
        Args:
            subject: Subject word
            verb: Verb word
            
        Returns:
            True if agreement is valid, False otherwise
            
        Example:
            >>> agreement = bridge.validate_subject_verb_agreement("ರಾಜು", "ಓದುತ್ತಾನೆ")
            >>> print(agreement)
            True
        """
        if not subject or not verb:
            return False
        
        try:
            subj_features = self.get_word_grammar_features(subject)
            verb_features = self.get_word_grammar_features(verb)
            
            # Check if person and number match
            subj_person = subj_features.get('person', '')
            subj_number = subj_features.get('number', '')
            verb_person = verb_features.get('person', '')
            verb_number = verb_features.get('number', '')
            
            agreement = (subj_person == verb_person and subj_number == verb_number)
            self.logger.debug(f"Subject-verb agreement for '{subject}'-'{verb}': {agreement}")
            return agreement
        
        except Exception as e:
            self.logger.warning(f"Error validating agreement: {e}")
            return False
    
    def get_vibhakti_type(self, word: str) -> str:
        """
        Identify what type of vibhakti a word has.
        
        Wraps: context_based_suggestion_para.identify_vibhakti()
                or vibakti_checking.identify_vibhakti()
        
        Args:
            word: Kannada word with potential vibhakti
            
        Returns:
            Vibhakti type (e.g., 'Nominative', 'Accusative', 'Dative', etc.)
            
        Example:
            >>> vibhakti = bridge.get_vibhakti_type("ಮನೆಯಲ್ಲಿ")
            >>> print(vibhakti)
            'Locative'
        """
        if not word:
            return ''
        
        try:
            # Try from context_based_suggestion_para
            if self.context_module and hasattr(self.context_module, 'identify_vibhakti'):
                return self.context_module.identify_vibhakti(word)
            
            # Try from vibakti_checking
            if self.vibhakti_module and hasattr(self.vibhakti_module, 'identify_vibhakti'):
                return self.vibhakti_module.identify_vibhakti(word)
            
            self.logger.warning("No identify_vibhakti function available")
            return ''
        
        except Exception as e:
            self.logger.warning(f"Error identifying vibhakti for '{word}': {e}")
            return ''
    
    def is_word_plural(self, word: str) -> bool:
        """
        Check if word is plural.
        
        Wraps: context_based_suggestion_para.is_plural_kannada()
                or vibakti_checking.is_plural_kannada()
        
        Args:
            word: Kannada word to check
            
        Returns:
            True if word is plural, False otherwise
        """
        if not word:
            return False
        
        try:
            # Try from context_based_suggestion_para
            if self.context_module and hasattr(self.context_module, 'is_plural_kannada'):
                return self.context_module.is_plural_kannada(word)
            
            # Try from vibakti_checking
            if self.vibhakti_module and hasattr(self.vibhakti_module, 'is_plural_kannada'):
                return self.vibhakti_module.is_plural_kannada(word)
            
            self.logger.warning("No is_plural function available")
            return False
        
        except Exception as e:
            self.logger.warning(f"Error checking plurality for '{word}': {e}")
            return False
    
    def get_word_gender(self, word: str) -> str:
        """
        Get gender of word.
        
        Wraps: context_based_suggestion_para.classify_gender_kannada()
                or vibakti_checking.classify_gender_kannada()
        
        Args:
            word: Kannada word to check
            
        Returns:
            Gender (e.g., 'masculine', 'feminine', 'neuter')
        """
        if not word:
            return ''
        
        try:
            # Try from context_based_suggestion_para
            if self.context_module and hasattr(self.context_module, 'classify_gender_kannada'):
                return self.context_module.classify_gender_kannada(word)
            
            # Try from vibakti_checking
            if self.vibhakti_module and hasattr(self.vibhakti_module, 'classify_gender_kannada'):
                return self.vibhakti_module.classify_gender_kannada(word)
            
            self.logger.warning("No classify_gender function available")
            return ''
        
        except Exception as e:
            self.logger.warning(f"Error classifying gender for '{word}': {e}")
            return ''
    
    # ================================================================
    # UTILITY/DEBUG METHODS
    # ================================================================
    
    def get_bridge_status(self) -> Dict[str, bool]:
        """
        Check which modules are available.
        
        Returns:
            Dict with availability status of each module
            
        Example:
            >>> status = bridge.get_bridge_status()
            >>> print(status)
            {'context_module': True, 'pos_tagging': True, ...}
        """
        return {
            'context_module': self.context_module is not None,
            'pos_tagging_module': self.pos_tagging_module is not None,
            'sandhi_module': self.sandhi_module is not None,
            'vibhakti_module': self.vibhakti_module is not None,
            'noun_dataset': self.noun_df is not None,
            'verb_dataset': self.verb_df is not None
        }
    
    def log_status(self) -> None:
        """Log the current status of all modules and datasets."""
        status = self.get_bridge_status()
        self.logger.info("=" * 50)
        self.logger.info("GRAMMAR INTEGRATION BRIDGE STATUS")
        self.logger.info("=" * 50)
        for module_name, available in status.items():
            status_str = "✓ Available" if available else "✗ Not available"
            self.logger.info(f"  {module_name}: {status_str}")
        self.logger.info("=" * 50)


# ========================================================================
# MODULE-LEVEL EXPORTS
# ========================================================================

def create_grammar_bridge(verbosity: bool = False) -> GrammarIntegrationBridge:
    """
    Factory function to create a GrammarIntegrationBridge instance.
    
    Args:
        verbosity: Enable debug logging
        
    Returns:
        Initialized GrammarIntegrationBridge instance
        
    Example:
        >>> bridge = create_grammar_bridge(verbosity=True)
        >>> structure = bridge.analyze_sentence_structure("ನಾನು ಪುಸ್ತಕ ಓದುತ್ತೇನೆ")
    """
    return GrammarIntegrationBridge(verbosity=verbosity)


if __name__ == "__main__":
    # Test the bridge
    print("Testing Grammar Integration Bridge...")
    bridge = GrammarIntegrationBridge(verbosity=True)
    bridge.log_status()
    print("\nBridge initialized successfully!")
