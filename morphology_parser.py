"""
Kannada FST Morphology Parser
Main class integrating all morphological components

This is the "FST engine" - it coordinates:
1. Lexicon lookup
2. Rule application  
3. Sandhi processing
4. Context analysis
"""

from verb_lexicon import VerbLexicon
from conjugation_rules import ConjugationRules
from sandhi_processor import SandhiProcessor
from context_analyzer import ContextAnalyzer


class KannadaMorphologyParser:
    """
    Main FST-based morphology parser for Kannada verbs
    
    Provides two main operations:
    1. GENERATE: root + features → conjugated form
    2. ANALYZE: conjugated form → root + features (optional)
    """
    
    def __init__(self):
        self.lexicon = VerbLexicon()
        self.conjugation = ConjugationRules()
        self.sandhi = SandhiProcessor()
        self.context = ContextAnalyzer()
    
    def generate(self, root, form_type, person=None, number=None, gender=None):
        """
        GENERATE conjugated form from root + grammatical features
        
        This is the main function you'll use for correction!
        
        Args:
            root: verb root (e.g., "ಕುಡಿ")
            form_type: "INFINITIVE", "PRESENT", "PAST", "FUTURE"
            person: "1", "2", "3" (optional for infinitive)
            number: "SG", "PL" (optional)
            gender: "MALE", "FEMALE", "NEUTRAL" (optional)
        
        Returns:
            conjugated verb form or None if cannot generate
            
        Example:
            generate("ಕುಡಿ", "INFINITIVE") → "ಕುಡಿಯಲು"
            generate("ಕುಡಿ", "PRESENT", "3", "SG", "MALE") → "ಕುಡಿಯುತ್ತಾನೆ"
        """
        # Get root information
        root_info = self.lexicon.get_root_info(root)
        if not root_info:
            # Unknown root - try anyway with inferred class
            root_info = self.lexicon._infer_root_class(root)
        
        # Handle irregular verbs specially
        if root_info.get("class") == "IRREGULAR" and form_type == "PAST":
            # Use special past stem
            past_stem = root_info.get("past_stem")
            if past_stem:
                # Get past suffix
                suffix = self.conjugation.get_suffix("PAST", person, number, gender)
                if suffix:
                    # Apply to irregular stem
                    return self._apply_suffix_to_stem(past_stem, suffix, root_info)
        
        # Get appropriate suffix
        suffix = self.conjugation.get_suffix(form_type, person, number, gender)
        if not suffix:
            return None
        
        # Apply sandhi rules
        result = self.sandhi.apply_sandhi(root, suffix, root_info)
        
        return result
    
    def _apply_suffix_to_stem(self, stem, suffix, root_info):
        """Helper to apply suffix with sandhi to irregular stems"""
        # For irregular past forms, usually just concatenate
        return stem + suffix
    
    def generate_from_context(self, root, sentence, verb_position=None):
        """
        Generate appropriate form based on sentence context
        
        This is even smarter - analyzes context automatically!
        
        Args:
            root: verb root
            sentence: full sentence or list of words
            verb_position: where verb should go (optional)
        
        Returns:
            best conjugated form for context
            
        Example:
            sentence = "ನಾನು ನೀರವನ್ನು [MASK] ಬೇಕು"
            generate_from_context("ಕುಡಿ", sentence) → "ಕುಡಿಯಲು"
        """
        # Analyze context
        if isinstance(sentence, str):
            context_info = self.context.analyze_context_from_sentence(sentence, verb_position)
        else:
            # List of words
            if verb_position is None:
                # Find [MASK] or use middle
                try:
                    verb_position = sentence.index("[MASK]")
                except (ValueError, AttributeError):
                    verb_position = len(sentence) // 2
            context_info = self.context.detect_required_form(sentence, verb_position)
        
        # Generate form based on detected context
        return self.generate(
            root,
            context_info["form_type"],
            context_info.get("person"),
            context_info.get("number"),
            context_info.get("gender")
        )
    
    def analyze(self, conjugated_form):
        """
        ANALYZE conjugated form to extract root + features (OPTIONAL)
        
        This is the reverse direction - useful for understanding what
        IndicBERT predictions are.
        
        Args:
            conjugated_form: full conjugated verb (e.g., "ಕುಡಿಯುತ್ತಾನೆ")
        
        Returns:
            dict with:
                - root: base form
                - form_type: tense
                - person, number, gender
                - confidence
            or None if cannot analyze
            
        Example:
            analyze("ಕುಡಿಯುತ್ತಾನೆ") → {
                "root": "ಕುಡಿ",
                "form_type": "PRESENT",
                "person": "3",
                "number": "SG", 
                "gender": "MALE",
                "confidence": 0.9
            }
        """
        # Try to match against known suffixes
        best_match = None
        best_score = 0
        
        # Check all present tense forms
        for key, form_data in self.conjugation.present.items():
            suffix = form_data["suffix"]
            if conjugated_form.endswith(suffix):
                # Found potential match
                potential_root = conjugated_form[:-len(suffix)]
                
                # Try to reverse sandhi
                if "ಯ" in potential_root and potential_root.endswith("ಯ"):
                    # Likely euphonic ಯ
                    potential_root = potential_root[:-1]
                elif "ವ" in potential_root and potential_root.endswith("ವ"):
                    potential_root = potential_root[:-1]
                
                # Check if this root exists
                root_info = self.lexicon.get_root_info(potential_root)
                if root_info or best_score < 0.7:
                    # Parse the key (e.g., "3SG_MALE")
                    parts = key.split("_")
                    if len(parts) >= 1:
                        person = parts[0][0]  # First digit
                        number = parts[0][1:]  # Rest (SG/PL)
                        gender = parts[1] if len(parts) > 1 else None
                        
                        match = {
                            "root": potential_root,
                            "form_type": "PRESENT",
                            "person": person,
                            "number": number,
                            "gender": gender,
                            "confidence": 0.9 if root_info else 0.7
                        }
                        
                        if root_info:
                            return match  # Found good match
                        elif not best_match:
                            best_match = match
                            best_score = 0.7
        
        # Check past tense forms similarly
        for key, form_data in self.conjugation.past.items():
            suffix = form_data["suffix"]
            if conjugated_form.endswith(suffix):
                potential_root = conjugated_form[:-len(suffix)]
                
                # Reverse sandhi
                if potential_root.endswith("ಯ"):
                    potential_root = potential_root[:-1]
                
                root_info = self.lexicon.get_root_info(potential_root)
                if root_info:
                    parts = key.split("_")
                    person = parts[0][0]
                    number = parts[0][1:]
                    gender = parts[1] if len(parts) > 1 else None
                    
                    return {
                        "root": potential_root,
                        "form_type": "PAST",
                        "person": person,
                        "number": number,
                        "gender": gender,
                        "confidence": 0.9
                    }
        
        # Check infinitive
        if conjugated_form.endswith("ಅಲು") or conjugated_form.endswith("ಲು"):
            if conjugated_form.endswith("ಯಲು"):
                # Likely had euphonic ಯ
                potential_root = conjugated_form[:-3]  # Remove ಯಲು
            elif conjugated_form.endswith("ಲು"):
                potential_root = conjugated_form[:-2]  # Remove ಲು
            else:
                potential_root = conjugated_form[:-3]  # Remove ಅಲು
            
            root_info = self.lexicon.get_root_info(potential_root)
            if root_info or len(potential_root) > 1:
                return {
                    "root": potential_root,
                    "form_type": "INFINITIVE",
                    "person": None,
                    "number": None,
                    "gender": None,
                    "confidence": 0.85 if root_info else 0.6
                }
        
        return best_match
    
    def filter_predictions_by_context(self, predictions, sentence, verb_position=None):
        """
        Re-rank IndicBERT predictions based on morphological context
        
        This is THE KEY FUNCTION for your thesis!
        
        Args:
            predictions: list of IndicBERT predictions
            sentence: context sentence
            verb_position: where verb goes
        
        Returns:
            re-ranked predictions with morphological scores
            
        Example:
            predictions = ["ಏಕೆ", "ಯಾಕೆ", "ಹೇಗೆ", "ಕುಡಿಯಲು", "ಕುಡಿಯ"]
            sentence = "ನಾನು ನೀರವನ್ನು [MASK] ಬೇಕು"
            filter_predictions_by_context(predictions, sentence)
            → Boosts "ಕುಡಿಯಲು" to rank #1 (correct infinitive!)
        """
        # Analyze context
        if isinstance(sentence, str):
            context_info = self.context.analyze_context_from_sentence(sentence, verb_position)
        else:
            if verb_position is None:
                try:
                    verb_position = sentence.index("[MASK]")
                except (ValueError, AttributeError):
                    verb_position = len(sentence) // 2
            context_info = self.context.detect_required_form(sentence, verb_position)
        
        required_form = context_info["form_type"]
        
        # Score each prediction
        scored_predictions = []
        for pred in predictions:
            # Analyze the prediction
            analysis = self.analyze(pred)
            
            base_score = 0.5  # Default
            
            if analysis:
                # Exact form match
                if analysis["form_type"] == required_form:
                    base_score = 0.95
                    
                    # Bonus for person/number/gender match
                    if context_info.get("person") and analysis.get("person") == context_info["person"]:
                        base_score += 0.03
                    if context_info.get("number") and analysis.get("number") == context_info["number"]:
                        base_score += 0.02
                
                # Partial credit for having correct root
                elif analysis.get("root") and self.lexicon.is_verb_root(analysis["root"]):
                    base_score = 0.6
            else:
                # Cannot analyze - might not be a verb
                base_score = 0.3
            
            scored_predictions.append({
                "word": pred,
                "morphology_score": base_score,
                "analysis": analysis,
                "matches_context": (analysis and analysis["form_type"] == required_form)
            })
        
        # Sort by morphology score
        scored_predictions.sort(key=lambda x: x["morphology_score"], reverse=True)
        
        return scored_predictions
    
    def correct_verb_form(self, wrong_verb, sentence, verb_position=None):
        """
        Auto-correct a verb to match sentence context
        
        Takes a verb in wrong form and generates correct form
        
        Args:
            wrong_verb: verb in incorrect form (e.g., "ಕುಡಿಯ")
            sentence: context sentence
            verb_position: where verb is
        
        Returns:
            corrected verb form
            
        Example:
            wrong_verb = "ಕುಡಿಯ"  (stem, wrong for context)
            sentence = "ನಾನು ನೀರವನ್ನು [MASK] ಬೇಕು"
            correct_verb_form(wrong_verb, sentence)
            → "ಕುಡಿಯಲು" (infinitive, correct!)
        """
        # First, try to extract root from wrong verb
        analysis = self.analyze(wrong_verb)
        
        if not analysis:
            # Cannot analyze - might already be a root
            root = wrong_verb
        else:
            root = analysis["root"]
        
        # Generate correct form from context
        return self.generate_from_context(root, sentence, verb_position)
    
    def get_all_forms(self, root):
        """
        Generate all possible forms of a verb (for testing/debugging)
        
        Returns:
            dict with all conjugated forms
        """
        forms = {}
        
        # Infinitive
        forms["infinitive"] = self.generate(root, "INFINITIVE")
        
        # Present tense - all persons
        forms["present"] = {}
        for key in self.conjugation.present.keys():
            parts = key.split("_")
            person = parts[0][0]
            number = parts[0][1:]
            gender = parts[1] if len(parts) > 1 else None
            
            form = self.generate(root, "PRESENT", person, number, gender)
            forms["present"][key] = form
        
        # Past tense - all persons
        forms["past"] = {}
        for key in self.conjugation.past.keys():
            parts = key.split("_")
            person = parts[0][0]
            number = parts[0][1:]
            gender = parts[1] if len(parts) > 1 else None
            
            form = self.generate(root, "PAST", person, number, gender)
            forms["past"][key] = form
        
        return forms