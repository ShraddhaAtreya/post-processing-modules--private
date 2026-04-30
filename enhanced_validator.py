"""
Enhanced Validation Module with Context Awareness (N-Gram) + Fuzzy Logic + Cosine Similarity + FST Morphology + Vibhakti Validation
====================================================================================================================================

Kannada OCR post-processing validator.

Features:
- ✅ Fuzzy Logic (6 metrics: Levenshtein + LCS + Cosine + First Char + Length + Vowel)
- ✅ Cosine Similarity (Character N-gram based)
- ✅ Dual dictionary support (Padakosha + Scraped)
- ✅ Context Awareness (Bigram Probability from IndicCorp)
- ✅ Hybrid Scoring (Fuzzy + Frequency + Context)
- ✅ Vibhakti-aware matching & Auto-punctuation stripping
- ✅ Vibhakti Error Detection & Correction (NEW!)
- ✅ Increased suggestions to Top-10
- ✅ FST Morphology Integration

Author: MTech Thesis Project
Date: January 2026
Version: 5.0 (Vibhakti Validation)
"""

import os
import json
import sqlite3
import csv
import math
from datetime import datetime
from threading import Lock
from collections import OrderedDict
from typing import Dict, List, Tuple, Set

# ============================================================
# CONFIGURATION
# ============================================================
WEIGHT_LEVENSHTEIN = 0.4
WEIGHT_FREQUENCY   = 0.1
WEIGHT_CONTEXT     = 0.5

# Normalization constants (Estimated from corpus size)
MAX_UNIGRAM_LOG = math.log(1000000)  # Max frequency normalization factor
MAX_BIGRAM_LOG  = math.log(100000)   # Max context normalization factor

# Fuzzy Logic Weights (UPDATED for Cosine)
FUZZY_WEIGHT_LEVENSHTEIN = 0.25  # Reduced from 0.30
FUZZY_WEIGHT_LCS = 0.20          # Reduced from 0.25
FUZZY_WEIGHT_COSINE = 0.25       # NEW! Character n-gram similarity
FUZZY_WEIGHT_FIRST_CHAR = 0.15   # Reduced from 0.20
FUZZY_WEIGHT_LENGTH = 0.10       # Reduced from 0.15
FUZZY_WEIGHT_VOWEL = 0.05        # Reduced from 0.10


# ============================================================
# LEVENSHTEIN DISTANCE (ORIGINAL)
# ============================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


# ============================================================
# COSINE SIMILARITY FUNCTIONS (NEW)
# ============================================================

def extract_character_ngrams(word: str, n: int = 2) -> List[str]:
    """
    Extract character n-grams from a word.
    
    Args:
        word: Input word
        n: N-gram size (2 for bigrams, 3 for trigrams)
    
    Returns:
        List of n-grams
    
    Example:
        extract_character_ngrams("ಕನ್ನಡ", 2) → ["ಕನ", "ನ್", "್ನ", "ನಡ"]
    """
    if len(word) < n:
        return [word]
    
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i:i+n])
    
    return ngrams


def word_to_ngram_vector(word: str, all_ngrams: Set[str]) -> List[int]:
    """
    Convert word to binary n-gram vector.
    
    Args:
        word: Input word
        all_ngrams: Set of all possible n-grams (vocabulary)
    
    Returns:
        Binary vector where 1 = n-gram present, 0 = absent
    
    Example:
        word = "ಕನ್ನಡ"
        all_ngrams = {"ಕನ", "ನ್", "್ನ", "ನಡ", "ಡಾ", "ತನ"}
        → [1, 1, 1, 1, 0, 0]  (first 4 present, last 2 absent)
    """
    word_ngrams = set(extract_character_ngrams(word, n=2))
    
    # Create binary vector
    vector = []
    for ngram in sorted(all_ngrams):  # Sort for consistent ordering
        vector.append(1 if ngram in word_ngrams else 0)
    
    return vector


def cosine_similarity_vectors(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Args:
        vec1, vec2: Numerical vectors (same length)
    
    Returns:
        Similarity score 0.0 to 1.0
    """
    if len(vec1) != len(vec2):
        return 0.0
    
    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]


def calculate_cosine_similarity(word1: str, word2: str) -> float:
    """
    Calculate cosine similarity between two words using character bigrams.
    
    Args:
        word1: First word (e.g., OCR error)
        word2: Second word (e.g., candidate correction)
    
    Returns:
        Similarity score 0.0 to 1.0
    
    Example:
        calculate_cosine_similarity("ಕನ್ನಡಾ", "ಕನ್ನಡ")
        → Extract bigrams from both
        → Create vectors
        → Calculate cosine
        → Returns ~0.89
    """
    if not word1 or not word2:
        return 0.0
    
    if word1 == word2:
        return 1.0
    
    # Extract bigrams from both words
    ngrams1 = set(extract_character_ngrams(word1, n=2))
    ngrams2 = set(extract_character_ngrams(word2, n=2))
    
    # Union of all n-grams (vocabulary)
    all_ngrams = ngrams1.union(ngrams2)
    
    if len(all_ngrams) == 0:
        return 0.0
    
    # Convert words to vectors
    vec1 = word_to_ngram_vector(word1, all_ngrams)
    vec2 = word_to_ngram_vector(word2, all_ngrams)
    
    # Calculate cosine similarity
    return cosine_similarity_vectors(vec1, vec2)


# ============================================================
# FUZZY LOGIC HELPER FUNCTIONS
# ============================================================

def longest_common_substring_length(s1: str, s2: str) -> int:
    """
    Find the length of the longest common substring.
    Example: "ಕನ್ನಡಾ" and "ಕನ್ನಡ" → LCS = "ಕನ್ನಡ" (length 4)
    """
    if not s1 or not s2:
        return 0
    
    m, n = len(s1), len(s2)
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0
    
    return max_length


def first_character_match(s1: str, s2: str) -> float:
    """
    Check if first characters match.
    Returns: 1.0 if match, 0.0 if different
    """
    if not s1 or not s2:
        return 0.0
    return 1.0 if s1[0] == s2[0] else 0.0


def length_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity based on length difference.
    Formula: 1.0 - (|len1 - len2| / max_len)
    
    Examples:
    - Same length: 1.0
    - Diff 1: 0.8-0.9
    - Diff 5: 0.0-0.5
    """
    if not s1 or not s2:
        return 0.0
    
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    
    if max_len == 0:
        return 1.0
    
    diff = abs(len1 - len2)
    similarity = 1.0 - (diff / max_len)
    
    return max(0.0, similarity)


def vowel_mark_tolerance(s1: str, s2: str) -> float:
    """
    Kannada-specific: Check if differences are only vowel marks.
    OCR commonly confuses these pairs:
    - ಾ (aa) ↔ '' (no vowel)
    - ಿ (i) ↔ ೀ (ee)
    - ು (u) ↔ ೂ (oo)
    - ೆ (e) ↔ ೇ (ee)
    
    Returns: 1.0 if only vowel mark differences, 0.0 otherwise
    """
    if not s1 or not s2:
        return 0.0
    
    # Kannada vowel marks (matras) Unicode range
    vowel_marks = {
        '\u0CBE',  # ಾ (aa)
        '\u0CBF',  # ಿ (i)
        '\u0CC0',  # ೀ (ee)
        '\u0CC1',  # ು (u)
        '\u0CC2',  # ೂ (oo)
        '\u0CC6',  # ೆ (e)
        '\u0CC7',  # ೇ (ee)
        '\u0CC8',  # ೈ (ai)
        '\u0CCA',  # ೊ (o)
        '\u0CCB',  # ೋ (oo)
    }
    
    # Remove all vowel marks from both strings
    s1_no_vowels = ''.join(c for c in s1 if c not in vowel_marks)
    s2_no_vowels = ''.join(c for c in s2 if c not in vowel_marks)
    
    # If strings are same after removing vowel marks, only vowel marks differed
    if s1_no_vowels == s2_no_vowels:
        return 1.0
    
    # Partial credit: if consonant structure is very similar
    if len(s1_no_vowels) > 0 and len(s2_no_vowels) > 0:
        common_len = longest_common_substring_length(s1_no_vowels, s2_no_vowels)
        max_len = max(len(s1_no_vowels), len(s2_no_vowels))
        return common_len / max_len if max_len > 0 else 0.0
    
    return 0.0


def calculate_fuzzy_similarity(word1: str, word2: str) -> float:
    """
    Calculate comprehensive fuzzy similarity score combining 6 metrics.
    
    Args:
        word1: The error word (from OCR)
        word2: The candidate correct word
    
    Returns:
        Float 0.0 to 1.0 (higher = more similar)
    
    Metrics:
    1. Levenshtein distance (normalized) - 25%
    2. Longest Common Substring - 20%
    3. Cosine Similarity (character n-grams) - 25% [NEW!]
    4. First character match - 15%
    5. Length similarity - 10%
    6. Vowel mark tolerance (Kannada-specific) - 5%
    """
    if not word1 or not word2:
        return 0.0
    
    if word1 == word2:
        return 1.0
    
    # Metric 1: Levenshtein Distance (normalized)
    lev_dist = levenshtein_distance(word1, word2)
    max_len = max(len(word1), len(word2))
    lev_score = 1.0 - (lev_dist / max_len) if max_len > 0 else 0.0
    lev_score = max(0.0, lev_score)
    
    # Metric 2: Longest Common Substring
    lcs_length = longest_common_substring_length(word1, word2)
    lcs_score = lcs_length / max_len if max_len > 0 else 0.0
    
    # Metric 3: Cosine Similarity (NEW!)
    cosine_score = calculate_cosine_similarity(word1, word2)
    
    # Metric 4: First Character Match
    first_char_score = first_character_match(word1, word2)
    
    # Metric 5: Length Similarity
    len_score = length_similarity(word1, word2)
    
    # Metric 6: Vowel Mark Tolerance
    vowel_score = vowel_mark_tolerance(word1, word2)
    
    # Combined Fuzzy Score (6 metrics!)
    fuzzy_score = (
        FUZZY_WEIGHT_LEVENSHTEIN * lev_score +
        FUZZY_WEIGHT_LCS * lcs_score +
        FUZZY_WEIGHT_COSINE * cosine_score +      # NEW!
        FUZZY_WEIGHT_FIRST_CHAR * first_char_score +
        FUZZY_WEIGHT_LENGTH * len_score +
        FUZZY_WEIGHT_VOWEL * vowel_score
    )
    
    return min(1.0, max(0.0, fuzzy_score))


# ============================================================
# CACHE
# ============================================================

class ValidationCache:
    def __init__(self, db_path="data/user_corrections.db", memory_size=10000):
        self.db_path = db_path
        self.memory_size = memory_size
        self.memory_cache = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS validation_cache (
                word TEXT PRIMARY KEY,
                is_valid INTEGER,
                suggestions TEXT,
                confidence REAL,
                source TEXT,
                created_at TEXT,
                last_used TEXT,
                use_count INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()
        print(f"[Cache] Database initialized: {self.db_path}")

    def get(self, word):
        with self.lock:
            if word in self.memory_cache:
                self.hits += 1
                return self.memory_cache[word]

        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT is_valid, suggestions, confidence, source FROM validation_cache WHERE word=?", (word,))
            row = cur.fetchone()
            
            if row:
                # Update last used
                cur.execute("UPDATE validation_cache SET last_used=?, use_count=use_count+1 WHERE word=?", 
                           (datetime.now().isoformat(), word))
                conn.commit()
                conn.close()

                self.hits += 1
                result = {
                    "word": word,
                    "valid": bool(row[0]),
                    "suggestions": json.loads(row[1]),
                    "confidence": row[2],
                    "source": row[3],
                    "cached": True
                }
                self._add_to_memory(word, result)
                return result
            conn.close()
        except Exception:
            pass

        self.misses += 1
        return None

    def set(self, word, result):
        with self.lock:
            self._add_to_memory(word, result)

        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO validation_cache
                (word, is_valid, suggestions, confidence, source, created_at, last_used, use_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT use_count FROM validation_cache WHERE word=?), 0) + 1)
            """, (word, int(result["valid"]), json.dumps(result["suggestions"]), result["confidence"], 
                  result["source"], datetime.now().isoformat(), datetime.now().isoformat(), word))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _add_to_memory(self, word, result):
        self.memory_cache[word] = result
        if len(self.memory_cache) > self.memory_size:
            self.memory_cache.popitem(last=False)

    def get_stats(self):
        total = self.hits + self.misses
        return {"hit_rate": round((self.hits / total) * 100, 2) if total else 0}


# ============================================================
# VIBHAKTI (CASE MARKER) VALIDATOR (NEW!)
# ============================================================

class VibhaktiValidator:
    """
    Validates and corrects Kannada noun case markers (vibhakti)
    
    Handles object markers (accusative case):
    - ಅನ್ನು (for consonants and ಅ/ಆ endings)
    - ಯನ್ನು (for ಇ/ಈ/ಎ/ಏ endings)
    - ವನ್ನು (for ಉ/ಊ/ಒ/ಓ endings)
    """
    
    def __init__(self):
        # Object markers by noun ending type
        self.object_markers = {
            "consonant_a": ["ಅನ್ನು", "ನ್ನು"],      # Default for consonants and ಅ/ಆ
            "i_e_endings": ["ಯನ್ನು"],               # For ಇ/ಈ/ಎ/ಏ
            "u_o_endings": ["ವನ್ನು"],               # For ಉ/ಊ/ಒ/ಓ
        }
        
        # All possible object markers
        self.all_object_markers = ["ಅನ್ನು", "ನ್ನು", "ಯನ್ನು", "ವನ್ನು"]
    
    def detect_vibhakti_error(self, word, dictionary_words):
        """
        Detect if word has incorrect vibhakti (case marker)
        
        Args:
            word: Full word with potential vibhakti error
            dictionary_words: Set of valid base words
        
        Returns:
            (is_error: bool, base_word: str, wrong_marker: str, correct_marker: str)
        
        Example:
            "ನೀರವನ್ನು" → (True, "ನೀರ", "ವನ್ನು", "ಅನ್ನು")
        """
        # Try all possible marker parsings and collect valid ones
        valid_parsings = []
        
        # Check markers in order of length (longest first)
        sorted_markers = sorted(self.all_object_markers, key=len, reverse=True)
        
        for marker in sorted_markers:
            if word.endswith(marker):
                # Extract base word by removing the marker
                base = word[:-len(marker)]
                
                # CRITICAL FIX: Remove trailing ಅ if present (dictionary inconsistency)
                # Some dictionaries store "ನೀರ" as "ನೀರಅ" with explicit inherent vowel
                if base.endswith("ಅ"):
                    base_without_a = base[:-1]
                else:
                    base_without_a = base
                
                # Try both versions - with and without trailing ಅ
                base_to_use = None
                if base in dictionary_words:
                    base_to_use = base
                elif base_without_a in dictionary_words and base_without_a:
                    base_to_use = base_without_a
                
                # Skip if neither version is in dictionary
                if not base_to_use:
                    continue
                
                # Determine correct marker for this base
                correct_marker = self._get_correct_object_marker(base_to_use)
                
                # Store this parsing
                # Treat 'ನ್ನು' as valid surface form of 'ಅನ್ನು'
                if correct_marker == "ಅನ್ನು" and marker == "ನ್ನು":
                    is_error = False
                else:
                    is_error = (marker != correct_marker)

# Store this parsing
                valid_parsings.append({
    "base": base_to_use,
    "current_marker": marker,
    "correct_marker": correct_marker,
    "is_error": is_error
})
                
        
        # If we found any parsings with errors, return the first error
        # Priority: Find the parsing where marker is WRONG
        for parsing in valid_parsings:
            if parsing["is_error"]:
                return True, parsing["base"], parsing["current_marker"], parsing["correct_marker"]
        
        # If all parsings are correct (or no parsings found), no error
        return False, None, None, None
    
    def _get_correct_object_marker(self, base_word):
        """
        Determine correct object marker for a base noun
        
        Args:
            base_word: Base noun (e.g., "ನೀರ", "ಕವಿ", "ಮನೆ")
        
        Returns:
            Correct object marker string
        
        Examples:
            "ನೀರ" (ends in ರ - consonant) → "ಅನ್ನು"
            "ಕವಿ" (ends in ಇ) → "ಯನ್ನು"
            "ಮನೆ" (ends in ಎ) → "ಯನ್ನು"
        """
        if not base_word:
            return "ಅನ್ನು"  # Default
        
        last_char = base_word[-1]
        
        # Check ending type
        if last_char in ["ಇ", "ಈ", "ಎ", "ಏ"]:
            return "ಯನ್ನು"
        elif last_char in ["ಉ", "ಊ", "ಒ", "ಓ"]:
            return "ವನ್ನು"
        else:
            # Consonant or ಅ/ಆ endings
            return "ಅನ್ನು"
    
    def correct_vibhakti(self, word, base_word, correct_marker):
    

        if not base_word:
            return word

        last_char = base_word[-1]

    # vowel categories
        i_e_vowels = ["ಇ","ಈ","ಎ","ಏ"]
        u_o_vowels = ["ಉ","ಊ","ಒ","ಓ"]

        if last_char in i_e_vowels:
            return base_word + "ಯನ್ನು"

        elif last_char in u_o_vowels:
            return base_word + "ವನ್ನು"

        else:
            return base_word + "ನ್ನು"
    
    def get_all_valid_forms(self, base_word):
        """
        Generate all valid vibhakti forms for a noun
        Useful for testing and validation
        """
        forms = {}
        forms["nominative"] = base_word
        forms["accusative"] = base_word + self._get_correct_object_marker(base_word)
        return forms


# ============================================================
# FST MORPHOLOGY COMPONENTS
# ============================================================

class VerbLexicon:
    """Kannada verb root lexicon with morphological classifications"""
    
    def __init__(self):
        # Core verb roots from collocation database
        self.roots = {
            # VOWEL-ENDING VERBS
            "ಕುಡಿ": {"type": "VOWEL_ENDING", "ending": "ಇ", "class": "REGULAR", "gloss": "drink", "conjugation_class": "CLASS_1"},
            "ತಿನ್ನು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "NASAL_CONSONANT", "gloss": "eat", "conjugation_class": "CLASS_2"},
            "ಓದು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "REGULAR", "gloss": "read", "conjugation_class": "CLASS_2"},
            "ಮಾಡು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "REGULAR", "gloss": "do/make", "conjugation_class": "CLASS_2"},
            "ಬರೆ": {"type": "VOWEL_ENDING", "ending": "ಎ", "class": "REGULAR", "gloss": "write", "conjugation_class": "CLASS_3"},
            "ನೋಡು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "REGULAR", "gloss": "see/look", "conjugation_class": "CLASS_2"},
            "ಕೇಳು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "REGULAR", "gloss": "hear/ask", "conjugation_class": "CLASS_2"},
            "ಕೊಡು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "REGULAR", "gloss": "give", "conjugation_class": "CLASS_2"},
            "ತೆಗೆ": {"type": "VOWEL_ENDING", "ending": "ಎ", "class": "REGULAR", "gloss": "take/remove", "conjugation_class": "CLASS_3"},
            "ಹಾಕು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "REGULAR", "gloss": "put/place", "conjugation_class": "CLASS_2"},
            "ತರು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "REGULAR", "gloss": "bring", "conjugation_class": "CLASS_2"},
            "ಕರೆ": {"type": "VOWEL_ENDING", "ending": "ಎ", "class": "REGULAR", "gloss": "call", "conjugation_class": "CLASS_3"},
            "ಹೇಳು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "REGULAR", "gloss": "say/tell", "conjugation_class": "CLASS_2"},
            # IRREGULAR VERBS
            "ಹೋಗು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "IRREGULAR", "gloss": "go", "conjugation_class": "IRREGULAR", "past_stem": "ಹೋದ"},
            "ಬರು": {"type": "VOWEL_ENDING", "ending": "ಉ", "class": "IRREGULAR", "gloss": "come", "conjugation_class": "IRREGULAR", "past_stem": "ಬಂದ"},
        }
        
        # Stem variants (IndicBERT may produce these)
        self.stem_variants = {
            "ಕುಡಿಯ": "ಕುಡಿ",
            "ಓದ": "ಓದು",
            "ಬರೆಯ": "ಬರೆ",
            "ತಿನ": "ತಿನ್ನು",
            "ಮಾಡ": "ಮಾಡು",
            "ನೋಡ": "ನೋಡು",
        }
    
    def get_root_info(self, word):
        if word in self.roots:
            return self.roots[word]
        if word in self.stem_variants:
            canonical_root = self.stem_variants[word]
            return self.roots.get(canonical_root)
        return self._infer_root_class(word)
    
    def _infer_root_class(self, word):
        if not word:
            return None
        last_char = word[-1]
        kannada_vowels = ["ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ೠ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ"]
        
        if last_char in kannada_vowels:
            if last_char in ["ಇ", "ಈ"]:
                conj_class = "CLASS_1"
            elif last_char in ["ಉ", "ಊ"]:
                conj_class = "CLASS_2"
            elif last_char in ["ಎ", "ಏ"]:
                conj_class = "CLASS_3"
            else:
                conj_class = "CLASS_2"
            return {"type": "VOWEL_ENDING", "ending": last_char, "class": "REGULAR", "gloss": "unknown", "conjugation_class": conj_class}
        else:
            return {"type": "CONSONANT_ENDING", "ending": last_char, "class": "REGULAR", "gloss": "unknown", "conjugation_class": "CLASS_4"}
    
    def is_verb_root(self, word):
        return word in self.roots or word in self.stem_variants


class ConjugationRules:
    """Kannada verb conjugation rules"""
    
    def __init__(self):
        self.infinitive = {"suffix": "ಅಲು", "contexts": ["ಬೇಕು", "ಬಹುದು", "ಆಗು", "ಸಾಧ್ಯ"]}
        
        self.present = {
            "1SG": {"suffix": "ಉತ್ತೇನೆ", "pronouns": ["ನಾನು"]},
            "1PL": {"suffix": "ಉತ್ತೇವೆ", "pronouns": ["ನಾವು"]},
            "2SG_INFORMAL": {"suffix": "ಉತ್ತೀಯ", "pronouns": ["ನೀನು"]},
            "2SG_FORMAL": {"suffix": "ಉತ್ತೀರಿ", "pronouns": ["ನೀವು"]},
            "2PL": {"suffix": "ಉತ್ತೀರಿ", "pronouns": ["ನೀವು"]},
            "3SG_MALE": {"suffix": "ಉತ್ತಾನೆ", "pronouns": ["ಅವನು", "ಈತನು"]},
            "3SG_FEMALE": {"suffix": "ಉತ್ತಾಳೆ", "pronouns": ["ಅವಳು", "ಈಕೆ"]},
            "3SG_NEUTRAL": {"suffix": "ಉತ್ತದೆ", "pronouns": ["ಅದು", "ಇದು"]},
            "3PL": {"suffix": "ಉತ್ತಾರೆ", "pronouns": ["ಅವರು", "ಇವರು"]},
        }
        
        self.past = {
            "1SG": {"suffix": "ಇದ್ದೇನೆ", "pronouns": ["ನಾನು"]},
            "1PL": {"suffix": "ಇದ್ದೇವೆ", "pronouns": ["ನಾವು"]},
            "2SG_INFORMAL": {"suffix": "ಇದ್ದೀಯ", "pronouns": ["ನೀನು"]},
            "2SG_FORMAL": {"suffix": "ಇದ್ದೀರಿ", "pronouns": ["ನೀವು"]},
            "2PL": {"suffix": "ಇದ್ದೀರಿ", "pronouns": ["ನೀವು"]},
            "3SG_MALE": {"suffix": "ಇದ್ದಾನೆ", "pronouns": ["ಅವನು", "ಈತನು"]},
            "3SG_FEMALE": {"suffix": "ಇದ್ದಾಳೆ", "pronouns": ["ಅವಳು", "ಈಕೆ"]},
            "3SG_NEUTRAL": {"suffix": "ಇತ್ತು", "pronouns": ["ಅದು", "ಇದು"]},
            "3PL": {"suffix": "ಇದ್ದಾರೆ", "pronouns": ["ಅವರು", "ಇವರು"]},
        }
    
    def get_suffix(self, tense, person=None, number=None, gender=None):
        if tense == "INFINITIVE":
            return self.infinitive["suffix"]
        
        if tense == "PRESENT":
            rule_dict = self.present
        elif tense == "PAST":
            rule_dict = self.past
        else:
            return None
        
        if person and number:
            key = f"{person}{number}"
            if gender and person == "3" and number == "SG":
                key = f"{key}_{gender}"
            elif person == "2" and number == "SG":
                key = f"{key}_FORMAL"
            
            if key in rule_dict:
                return rule_dict[key]["suffix"]
        
        return None
    
    def get_infinitive_contexts(self):
        return self.infinitive["contexts"]
    
    def detect_tense_from_suffix(self, suffix):
        for form_data in self.present.values():
            if suffix == form_data["suffix"]:
                return "PRESENT"
        for form_data in self.past.values():
            if suffix == form_data["suffix"]:
                return "PAST"
        if suffix == self.infinitive["suffix"] or suffix.endswith("ಅಲು"):
            return "INFINITIVE"
        return None


class SandhiProcessor:
    """Kannada sandhi (phonological) rules processor"""
    
    def __init__(self):
        self.vowels = ["ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ೠ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ"]
        self.consonants = ["ಕ", "ಖ", "ಗ", "ಘ", "ಙ", "ಚ", "ಛ", "ಜ", "ಝ", "ಞ", "ಟ", "ಠ", "ಡ", "ಢ", "ಣ", 
                          "ತ", "ಥ", "ದ", "ಧ", "ನ", "ಪ", "ಫ", "ಬ", "ಭ", "ಮ", "ಯ", "ರ", "ಲ", "ವ", "ಶ", "ಷ", "ಸ", "ಹ", "ಳ", "ೞ"]
        
        # Vowel to vowel sign mapping
        self.vowel_to_sign = {
            "ಅ": "",
            "ಆ": "ಾ",
            "ಇ": "ಿ",
            "ಈ": "ೀ",
            "ಉ": "ು",
            "ಊ": "ೂ",
            "ಋ": "ೃ",
            "ೠ": "ೄ",
            "ಎ": "ೆ",
            "ಏ": "ೇ",
            "ಐ": "ೈ",
            "ಒ": "ೊ",
            "ಓ": "ೋ",
            "ಔ": "ೌ"
        }
    
    def apply_sandhi(self, root, suffix, root_info):
        if not root or not suffix:
            return root + suffix
        
        last_char = root[-1]
        first_char = suffix[0] if suffix else ""
        
        # Rule 1: Vowel + Vowel → Insert ಯ and merge
        if self._is_vowel_or_has_vowel_sign(last_char) and first_char in self.vowels:
            return self._apply_euphonic_insertion(root, suffix, root_info)
        
        # Rule 2: Consonant + Vowel → Direct
        if self._is_pure_consonant(last_char) and first_char in self.vowels:
            return self._apply_direct_concatenation(root, suffix)
        
        # Rule 3: Nasal assimilation
        if root_info and root_info.get("class") == "NASAL_CONSONANT":
            return self._apply_nasal_assimilation(root, suffix)
        
        return root + suffix
    
    def _is_vowel_or_has_vowel_sign(self, char):
        if char in self.vowels:
            return True
        if char in self.consonants:
            return True
        return False
    
    def _is_pure_consonant(self, char):
        return char == "್"
    
    def _apply_euphonic_insertion(self, root, suffix, root_info):
        last_char = root[-1]
        
        if last_char in ["ಇ", "ಈ", "ಎ", "ಏ", "ಐ"]:
            euphonic = "ಯ"
        elif last_char in ["ಉ", "ಊ", "ಒ", "ಓ", "ಔ"]:
            euphonic = "ವ"
        else:
            euphonic = "ಯ"
        
        # Special handling for ಉ-ending verbs
        if root_info and root_info.get("ending") == "ಉ":
            if suffix.startswith("ಅ"):
                return root[:-1] + suffix[1:]
        
        # Vowel sign conversion (CRITICAL FIX!)
        first_vowel = suffix[0] if suffix else ""
        
        if first_vowel in self.vowel_to_sign:
            vowel_sign = self.vowel_to_sign[first_vowel]
            rest_of_suffix = suffix[1:]
            
            if vowel_sign == "":
                return root + euphonic + rest_of_suffix
            else:
                return root + euphonic + vowel_sign + rest_of_suffix
        else:
            return root + euphonic + suffix
    
    def _apply_direct_concatenation(self, root, suffix):
        if root.endswith("್"):
            if suffix.startswith("ಅ"):
                return root[:-1] + suffix[1:]
            return root[:-1] + suffix
        return root + suffix
    
    def _apply_nasal_assimilation(self, root, suffix):
        if root.endswith("ನ್ನು") and suffix.startswith("ಅ"):
            return root[:-1] + suffix[1:]
        if root.endswith("ಉ") and suffix.startswith("ಅ"):
            return root[:-1] + suffix[1:]
        return root + suffix


class ContextAnalyzer:
    """Analyzes sentence context to determine required verb form"""
    
    def __init__(self):
        self.infinitive_triggers = ["ಬೇಕು", "ಬೇಕಾಗುತ್ತದೆ", "ಬೇಡ", "ಬಹುದು", "ಆಗು", "ಸಾಧ್ಯ", "ಕೂಡಾದು", "ಹೊರತು", "ಮುಂಚೆ"]
        
        self.subject_pronouns = {
            "ನಾನು": {"person": "1", "number": "SG", "gender": None},
            "ನಾವು": {"person": "1", "number": "PL", "gender": None},
            "ನೀನು": {"person": "2", "number": "SG", "gender": "INFORMAL"},
            "ನೀವು": {"person": "2", "number": "PL", "gender": None},
            "ಅವನು": {"person": "3", "number": "SG", "gender": "MALE"},
            "ಈತನು": {"person": "3", "number": "SG", "gender": "MALE"},
            "ಅವಳು": {"person": "3", "number": "SG", "gender": "FEMALE"},
            "ಈಕೆ": {"person": "3", "number": "SG", "gender": "FEMALE"},
            "ಅದು": {"person": "3", "number": "SG", "gender": "NEUTRAL"},
            "ಇದು": {"person": "3", "number": "SG", "gender": "NEUTRAL"},
            "ಅವರು": {"person": "3", "number": "PL", "gender": None},
            "ಇವರು": {"person": "3", "number": "PL", "gender": None},
        }
    
    def detect_required_form(self, words, mask_index):
        result = {"form_type": "PRESENT", "person": None, "number": None, "gender": None, "confidence": 0.5}
       # Check next 2 words for infinitive triggers (e.g., ಬೇಕು, ಸಾಧ್ಯ)
        for i in range(mask_index + 1, min(mask_index + 3, len(words))):
            next_word = words[i]
            if next_word in self.infinitive_triggers:
                result["form_type"] = "INFINITIVE"
                result["confidence"] = 0.95
                return result
        
        # Check for subject pronoun
        subject = None
        for i in range(mask_index - 1, -1, -1):
            word = words[i]
            if word in self.subject_pronouns:
                subject = word
                break
            if i < mask_index - 5:
                break
        
        if subject:
            features = self.subject_pronouns[subject]
            result["person"] = features["person"]
            result["number"] = features["number"]
            result["gender"] = features["gender"]
            result["confidence"] = 0.8
        
        return result
    
    def analyze_context_from_sentence(self, sentence, verb_position):
        words = sentence.split()
        
        if isinstance(verb_position, int):
            mask_index = verb_position
        else:
            try:
                mask_index = words.index("[MASK]")
            except ValueError:
                mask_index = len(words) // 2
        
        return self.detect_required_form(words, mask_index)


class KannadaMorphologyParser:
    """Main FST morphology parser coordinating all components"""
    
    def __init__(self):
        self.lexicon = VerbLexicon()
        self.conjugation = ConjugationRules()
        self.sandhi = SandhiProcessor()
        self.context = ContextAnalyzer()
    
    def generate(self, root, form_type, person=None, number=None, gender=None):
        root_info = self.lexicon.get_root_info(root)
        if not root_info:
            root_info = self.lexicon._infer_root_class(root)
        
        # Handle irregular verbs
        if root_info.get("class") == "IRREGULAR" and form_type == "PAST":
            past_stem = root_info.get("past_stem")
            if past_stem:
                suffix = self.conjugation.get_suffix("PAST", person, number, gender)
                if suffix:
                    return self._apply_suffix_to_stem(past_stem, suffix, root_info)
        
        suffix = self.conjugation.get_suffix(form_type, person, number, gender)
        if not suffix:
            return None
        
        result = self.sandhi.apply_sandhi(root, suffix, root_info)
        return result
    
    def _apply_suffix_to_stem(self, stem, suffix, root_info):
        return stem + suffix
    
    def generate_from_context(self, root, sentence, verb_position=None):
        if isinstance(sentence, str):
            context_info = self.context.analyze_context_from_sentence(sentence, verb_position)
        else:
            if verb_position is None:
                try:
                    verb_position = sentence.index("[MASK]")
                except (ValueError, AttributeError):
                    verb_position = len(sentence) // 2
            context_info = self.context.detect_required_form(sentence, verb_position)
        
        return self.generate(root, context_info["form_type"], context_info.get("person"), 
                           context_info.get("number"), context_info.get("gender"))
    
    def analyze(self, conjugated_form):
        best_match = None
        best_score = 0
        
        # Check present tense
        for key, form_data in self.conjugation.present.items():
            suffix = form_data["suffix"]
            if conjugated_form.endswith(suffix):
                potential_root = conjugated_form[:-len(suffix)]
                if "ಯ" in potential_root and potential_root.endswith("ಯ"):
                    potential_root = potential_root[:-1]
                elif "ವ" in potential_root and potential_root.endswith("ವ"):
                    potential_root = potential_root[:-1]
                
                root_info = self.lexicon.get_root_info(potential_root)
                if root_info or best_score < 0.7:
                    parts = key.split("_")
                    if len(parts) >= 1:
                        person = parts[0][0]
                        number = parts[0][1:]
                        gender = parts[1] if len(parts) > 1 else None
                        
                        match = {"root": potential_root, "form_type": "PRESENT", "person": person, 
                                "number": number, "gender": gender, "confidence": 0.9 if root_info else 0.7}
                        
                        if root_info:
                            return match
                        elif not best_match:
                            best_match = match
                            best_score = 0.7
        
        # Check past tense
        for key, form_data in self.conjugation.past.items():
            suffix = form_data["suffix"]
            if conjugated_form.endswith(suffix):
                potential_root = conjugated_form[:-len(suffix)]
                if potential_root.endswith("ಯ"):
                    potential_root = potential_root[:-1]
                
                root_info = self.lexicon.get_root_info(potential_root)
                if root_info:
                    parts = key.split("_")
                    person = parts[0][0]
                    number = parts[0][1:]
                    gender = parts[1] if len(parts) > 1 else None
                    return {"root": potential_root, "form_type": "PAST", "person": person, 
                           "number": number, "gender": gender, "confidence": 0.9}
        
        # Check infinitive
        if conjugated_form.endswith("ಅಲು") or conjugated_form.endswith("ಲು"):
            if conjugated_form.endswith("ಯಲು"):
                potential_root = conjugated_form[:-3]
            elif conjugated_form.endswith("ಲು"):
                potential_root = conjugated_form[:-2]
            else:
                potential_root = conjugated_form[:-3]
            
            root_info = self.lexicon.get_root_info(potential_root)
            if root_info or len(potential_root) > 1:
                return {"root": potential_root, "form_type": "INFINITIVE", "person": None, 
                       "number": None, "gender": None, "confidence": 0.85 if root_info else 0.6}
        
        return best_match
    
    def filter_predictions_by_context(self, predictions, sentence, verb_position=None):
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
        
        scored_predictions = []
        for pred in predictions:
            analysis = self.analyze(pred)
            base_score = 0.5
            
            if analysis:
                if analysis["form_type"] == required_form:
                    base_score = 0.95
                    if context_info.get("person") and analysis.get("person") == context_info["person"]:
                        base_score += 0.03
                    if context_info.get("number") and analysis.get("number") == context_info["number"]:
                        base_score += 0.02
                elif analysis.get("root") and self.lexicon.is_verb_root(analysis["root"]):
                    base_score = 0.6
            else:
                base_score = 0.3
            
            scored_predictions.append({"word": pred, "morphology_score": base_score, "analysis": analysis, 
                                      "matches_context": (analysis and analysis["form_type"] == required_form)})
        
        scored_predictions.sort(key=lambda x: x["morphology_score"], reverse=True)
        return scored_predictions
    
    def correct_verb_form(self, wrong_verb, sentence, verb_position=None):
        analysis = self.analyze(wrong_verb)
        
        if not analysis:
            root = wrong_verb
        else:
            root = analysis["root"]
        
        return self.generate_from_context(root, sentence, verb_position)


# ============================================================
# ENHANCED VALIDATOR WITH CONTEXT + FUZZY + COSINE + FST + VIBHAKTI
# ============================================================

class EnhancedValidator:
    def __init__(
        self,
        dictionary_paths=[
            "data/dictionaries/Padakosha_kannada_csv.csv",
            "data/dictionaries/combined_word_scrapped_csv.csv"
        ],
        context_db_path="data/ngram_context.db",
        cache_size=10000,
        max_edit_distance=2,
        enable_fst=True,
        enable_vibhakti_validation=True
    ):
        self.dictionary_paths = dictionary_paths if isinstance(dictionary_paths, list) else [dictionary_paths]
        self.cache = ValidationCache(memory_size=cache_size)
        self.max_edit_distance = max_edit_distance
        self.enable_fst = enable_fst
        self.enable_vibhakti_validation = enable_vibhakti_validation

        # Load Dictionaries
        self.dictionary_words = set()
        self.dictionary_list = []
        self._load_dictionaries()

        # Connect to Context Database
        self.context_db_path = context_db_path
        self.context_enabled = self._check_context_db()

        # FST Morphology
        if self.enable_fst:
            self.morphology = KannadaMorphologyParser()
            print("[EnhancedValidator] FST Morphology: ✅ ENABLED")
        else:
            self.morphology = None
            print("[EnhancedValidator] FST Morphology: ⚠️ DISABLED")
        
        # Vibhakti Validator (NEW!)
        if self.enable_vibhakti_validation:
            self.vibhakti = VibhaktiValidator()
            print("[EnhancedValidator] Vibhakti Validation: ✅ ENABLED")
        else:
            self.vibhakti = None
            print("[EnhancedValidator] Vibhakti Validation: ⚠️ DISABLED")

        # Explicit verb lemma map (Hardcoded rules)
        self.verb_lemma_map = {
            "ಬರುತ್ತಾನೆ": "ಬರುವುದು", "ಬರುತ್ತಾಳೆ": "ಬರುವುದು", "ಬರುತ್ತಾರೆ": "ಬರುವುದು",
            "ಹೋಗುತ್ತಾನೆ": "ಹೋಗುವುದು", "ಮಾಡುತ್ತಾನೆ": "ಮಾಡುವುದು",
        }

        self.total_validations = 0
        
        print("[EnhancedValidator] Initialized with FUZZY LOGIC + COSINE SIMILARITY + FST MORPHOLOGY + VIBHAKTI VALIDATION ✨")
        print(f"  - Fuzzy Metrics: Levenshtein + LCS + Cosine + FirstChar + Length + Vowel")
        print(f"  - Context Awareness: {'✅ ENABLED' if self.context_enabled else '⚠️ DISABLED (Run build_ngram_db.py)'}")
        if self.context_enabled:
            print(f"  - Ranking Weights: Fuzzy={WEIGHT_LEVENSHTEIN}, Freq={WEIGHT_FREQUENCY}, Context={WEIGHT_CONTEXT}")

    # --------------------------------------------------------
    # INITIALIZATION HELPERS
    # --------------------------------------------------------

    def _check_context_db(self):
        """Check if N-gram database exists and is valid"""
        if not os.path.exists(self.context_db_path):
            return False
        try:
            conn = sqlite3.connect(self.context_db_path)
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM unigrams")
            count = cur.fetchone()[0]
            conn.close()
            return count > 0
        except:
            return False

    def _load_dictionaries(self):
        """Load words from BOTH dictionaries"""
        for dict_path in self.dictionary_paths:
            if not os.path.exists(dict_path):
                print(f"[Validator] Warning: Dictionary not found: {dict_path}")
                continue
            
            print(f"[Validator] Loading: {os.path.basename(dict_path)}...")
            try:
                with open(dict_path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    
                    for row in reader:
                        word = None
                        if len(row) >= 2 and row[1].strip(): word = row[1].strip()
                        elif len(row) >= 1 and row[0].strip(): word = row[0].strip()
                        
                        if word:
                            word = word.replace(':', '').replace(';', '').replace('¹', '').strip()
                            if word and len(word) >= 2:
                                if any('\u0C80' <= c <= '\u0CFF' for c in word):
                                    if word not in self.dictionary_words:
                                        self.dictionary_words.add(word)
                                        self.dictionary_list.append(word)
            except Exception as e:
                print(f"[Validator] Error loading {dict_path}: {e}")
        
        print(f"[Validator] ✅ Total loaded: {len(self.dictionary_words):,} words")

    # --------------------------------------------------------
    # CONTEXT QUERY HELPERS
    # --------------------------------------------------------

    def _get_scores(self, prev_word, candidate):
        """Query DB to get Frequency Score and Context Score."""
        freq_score = 0.0
        context_score = 0.0
        
        if not self.context_enabled:
            return 0.0, 0.0

        try:
            conn = sqlite3.connect(self.context_db_path)
            cur = conn.cursor()
            
            # Unigram Score
            cur.execute("SELECT count FROM unigrams WHERE word=?", (candidate,))
            row = cur.fetchone()
            if row and row[0] > 0:
                freq_score = math.log(row[0]) / MAX_UNIGRAM_LOG
                freq_score = min(1.0, max(0.0, freq_score))

            # Bigram Score
            if prev_word:
                cur.execute("SELECT count FROM bigrams WHERE prev_word=? AND next_word=?", (prev_word, candidate))
                row = cur.fetchone()
                if row and row[0] > 0:
                    context_score = math.log(row[0]) / MAX_BIGRAM_LOG
                    context_score = min(1.0, max(0.0, context_score))
            
            conn.close()
        except Exception:
            pass
        
        return freq_score, context_score

    # --------------------------------------------------------
    # FST & VIBHAKTI ERROR DETECTION (NEW!)
    # --------------------------------------------------------

    def _detect_error_type(self, word, context_sentence):
        """Detect error type: SPELLING, SEMANTIC, MORPHOLOGICAL, VIBHAKTI, or None"""
        
        # Check vibhakti errors FIRST (before dictionary check)
        if self.enable_vibhakti_validation and self.vibhakti:
            is_vibhakti_error, base, wrong_marker, correct_marker = self.vibhakti.detect_vibhakti_error(word, self.dictionary_words)
            if is_vibhakti_error:
                return "VIBHAKTI"
        
        # Check morphology if FST enabled
        morpho_analysis = None
        if self.enable_fst and self.morphology:
            morpho_analysis = self.morphology.analyze(word)
        
        # Check if it's a verb stem (wrong form) even if in dictionary
        if self.enable_fst and morpho_analysis and context_sentence:
            context_info = self.morphology.context.analyze_context_from_sentence(context_sentence, None)
            # If form doesn't match context, it's morphological error
            if morpho_analysis["form_type"] != context_info["form_type"]:
                return "MORPHOLOGICAL"
            # Special check: if it's detected as a stem variant (ಕುಡಿಯ) in infinitive context
            if context_info["form_type"] == "INFINITIVE" and word in self.morphology.lexicon.stem_variants:
                return "MORPHOLOGICAL"
        
        # Check dictionary
        in_dict = word in self.dictionary_words
        
        # Not in dict and not valid morphology → spelling error
        if not in_dict and not morpho_analysis:
            return "SPELLING"
        
        return None

    # --------------------------------------------------------
    # FST MORPHOLOGY SCORE
    # --------------------------------------------------------

    def _get_morphology_score(self, candidate, context_sentence, position):
        """Calculate morphological appropriateness score using FST"""
        
        if not self.enable_fst or not self.morphology or not context_sentence:
            return 0.5
        
        # Analyze the candidate
        analysis = self.morphology.analyze(candidate)
        
        if not analysis:
            return 0.3  # Not a verb
        
        # Analyze context
        context_info = self.morphology.context.analyze_context_from_sentence(context_sentence, position)
        
        # Check if form matches context
        if analysis["form_type"] == context_info["form_type"]:
            score = 0.95
            
            # Bonus for person/number/gender match
            if context_info.get("person") == analysis.get("person"):
                score += 0.03
            if context_info.get("number") == analysis.get("number"):
                score += 0.02
            
            return min(score, 1.0)
        else:
            return 0.4  # Wrong form

    # --------------------------------------------------------
    # CORE VALIDATION LOGIC (UPDATED WITH FST + VIBHAKTI)
    # --------------------------------------------------------

    def validate_word(self, word: str, prev_word: str = None, full_sentence: str = None, position: int = None) -> Dict:
        """
        Validate a word and provide ranked suggestions using FUZZY LOGIC + COSINE + FST + VIBHAKTI.
        
        Args:
            word: The word to validate
            prev_word: The previous word in the sentence (for n-gram context)
            full_sentence: Full sentence (for FST morphology context)
            position: Word position in sentence (for FST)
        
        Returns:
            Dict with validation results and top-10 suggestions
        """
        # Input Cleaning
        clean_word = word.replace(':', '').replace(';', '').strip()
        if not clean_word:
             return {"word": word, "valid": False, "suggestions": [], "confidence": 0.0, "source": "empty"}

        self.total_validations += 1

        # Detect error type (now includes MORPHOLOGICAL and VIBHAKTI!)
        error_type = None
        if full_sentence or self.enable_vibhakti_validation:
            error_type = self._detect_error_type(clean_word, full_sentence)

        # Check Validity
        is_valid = False
        if clean_word in self.dictionary_words:
            is_valid = True
        elif clean_word in self.verb_lemma_map:
            is_valid = True
        else:
            stripped = self._strip_vibhakti(clean_word)
            if stripped != clean_word and stripped in self.dictionary_words:
                is_valid = True
        
        # Override validity if vibhakti error detected
        if error_type == "VIBHAKTI":
            is_valid = False
        
        # If valid and no morphological/vibhakti error, return early
        if is_valid and error_type not in ["MORPHOLOGICAL", "VIBHAKTI"]:
            return {"word": word, "valid": True, "suggestions": [], "confidence": 1.0, "source": "dictionary"}

        # Generate Candidates
        raw_candidates = self._find_similar_words_levenshtein(clean_word, max_suggestions=20)
        
        # Add FST-generated candidates for morphological errors
        fst_corrected_form = None
        if error_type == "MORPHOLOGICAL" and self.enable_fst and self.morphology:
            fst_corrected = self.morphology.correct_verb_form(clean_word, full_sentence, position)
            if fst_corrected:
                fst_corrected_form = fst_corrected  # Store for priority ranking
                if fst_corrected not in raw_candidates:
                    raw_candidates.insert(0, fst_corrected)
        
        # Add vibhakti-corrected candidates (NEW!)
        vibhakti_corrected_form = None
        if error_type == "VIBHAKTI" and self.enable_vibhakti_validation and self.vibhakti:
            is_error, base, wrong_marker, correct_marker = self.vibhakti.detect_vibhakti_error(clean_word, self.dictionary_words)
            print(f"[DEBUG Vibhakti] Word: {clean_word}, Base: {base}, Wrong: {wrong_marker}, Correct: {correct_marker}")
            if is_error and base:
                # FIXED: Direct concatenation
                vibhakti_corrected = self.vibhakti.correct_vibhakti(clean_word, base, correct_marker)
                print(f"[DEBUG Vibhakti] Correction: {base} + {correct_marker} = {vibhakti_corrected}")
                if vibhakti_corrected:
                    vibhakti_corrected_form = vibhakti_corrected  # Store for priority ranking
                    if vibhakti_corrected not in raw_candidates:
                        raw_candidates.insert(0, vibhakti_corrected)
        
        # Hybrid Ranking with FUZZY + COSINE + FST
        ranked_candidates = []
        
        for cand in raw_candidates:
            # Score A: FUZZY SIMILARITY (includes Cosine!)
            fuzzy_score = calculate_fuzzy_similarity(clean_word, cand)
            
            # Score B & C: Frequency and Context
            freq_score, context_score = self._get_scores(prev_word, cand)
            
            # Score D: MORPHOLOGY
            morpho_score = self._get_morphology_score(cand, full_sentence, position) if full_sentence else 0.5
            
            # CRITICAL: Boost corrected forms to top
            vibhakti_boost = 0.0
            fst_boost = 0.0
            
            if vibhakti_corrected_form and cand == vibhakti_corrected_form:
                vibhakti_boost = 10.0  # Massive boost to GUARANTEE it ranks #1
            
            if fst_corrected_form and cand == fst_corrected_form:
                fst_boost = 10.0  # Massive boost for FST-corrected form
            
            # Final Combined Score (with morphology!)
            if error_type == "MORPHOLOGICAL":
                # Heavy morphology weight for morphological errors + FST boost
                final_score = (fuzzy_score * 0.2) + (freq_score * 0.1) + (context_score * 0.2) + (morpho_score * 0.5) + fst_boost
            elif error_type == "VIBHAKTI":
                # Heavy fuzzy weight for vibhakti errors (similar spelling) + BOOST for corrected form
                final_score = (fuzzy_score * 0.6) + (freq_score * 0.2) + (context_score * 0.2) + vibhakti_boost
            else:
                # Standard weighting
                final_score = (fuzzy_score * WEIGHT_LEVENSHTEIN) + (freq_score * WEIGHT_FREQUENCY) + (context_score * WEIGHT_CONTEXT) + (morpho_score * 0.1)
            
            ranked_candidates.append({
                "word": cand,
                "score": final_score,
                "fuzzy": fuzzy_score,
                "freq": freq_score,
                "context": context_score,
                "morphology": morpho_score
            })

        # Sort by Final Score
        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Extract top 10
        final_suggestions = [x["word"] for x in ranked_candidates[:10]]

        # Construct Result
        source = "fuzzy_cosine_fst_vibhakti" if (self.enable_fst and self.enable_vibhakti_validation and full_sentence) else ("fuzzy_cosine_fst" if (self.enable_fst and full_sentence) else ("fuzzy_cosine_context" if (prev_word and self.context_enabled) else "fuzzy_cosine"))
        
        result = {
            "word": word,
            "valid": is_valid,
            "suggestions": final_suggestions,
            "confidence": 1.0 if is_valid else 0.5,
            "source": source,
            "error_type": error_type
        }
        
        # Cache Result
        if not prev_word and not full_sentence:
            self.cache.set(clean_word, result)
            
        return result

    # --------------------------------------------------------
    # SENTENCE-LEVEL CORRECTION
    # --------------------------------------------------------

    def correct_sentence(self, sentence: str) -> str:
        """
        Correct entire sentence using FST morphology and vibhakti validation
        
        Args:
            sentence: Input sentence with potential errors
        
        Returns:
            Corrected sentence
        """
        if not sentence:
            return sentence
        
        words = sentence.split()
        corrected_words = []
        
        for i, word in enumerate(words):
            prev_word = words[i-1] if i > 0 else None
            
            # Validate word in full sentence context
            result = self.validate_word(word, prev_word=prev_word, full_sentence=sentence, position=i)
            
            if result["valid"]:
                corrected_words.append(word)
            else:
                # Take best suggestion
                if result["suggestions"]:
                    best = result["suggestions"][0]
                    corrected_words.append(best)
                else:
                    corrected_words.append(word)
        
        return " ".join(corrected_words)

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _find_similar_words_levenshtein(self, word: str, max_suggestions: int = 20) -> List[str]:
        """Find similar words using basic Levenshtein distance."""
        if not word or len(self.dictionary_words) == 0: 
            return []
        
        suggestions_with_distance = []
        word_len = len(word)
        min_len = max(2, word_len - self.max_edit_distance)
        max_len = word_len + self.max_edit_distance
        
        candidates = [w for w in self.dictionary_list if min_len <= len(w) <= max_len and w != word]
        
        if len(candidates) > 10000:
            first_char = word[0] if word else ''
            same_start = [w for w in candidates if w[0] == first_char]
            if len(same_start) > 100: 
                candidates = same_start
        
        for candidate in candidates:
            distance = levenshtein_distance(word, candidate)
            if distance <= self.max_edit_distance:
                suggestions_with_distance.append((candidate, distance))
        
        suggestions_with_distance.sort(key=lambda x: (x[1], abs(len(x[0]) - word_len)))
        
        seen = set()
        suggestions = []
        for w, _ in suggestions_with_distance:
            if w not in seen:
                seen.add(w)
                suggestions.append(w)
                if len(suggestions) >= max_suggestions: 
                    break
        
        return suggestions

    def _strip_vibhakti(self, word: str) -> str:
        """Strip Kannada case markers (vibhakti)"""
        if len(word) < 3: 
            return word
        
        vibhakti_suffixes = [
            'ಗಳಿಂದ', 'ಗಳಲ್ಲಿ', 'ಗಳನ್ನು', 'ಗಳಿಗೆ', 'ಯವರು', 'ಯವರ', 'ಯಲ್ಲಿ', 'ಯಿಂದ', 'ಯನ್ನು',
            'ಅಲ್ಲಿ', 'ಇಂದ', 'ಅನ್ನು', 'ವನ್ನು', 'ಗಳು', 'ಗಳ', 'ವರು', 'ವರ', 'ತ್ತಾನೆ', 'ತ್ತಾಳೆ', 'ತ್ತಾರೆ',
            'ಲ್ಲಿ', 'ನ್ನು', 'ಗೆ', 'ನು', 'ಯ', 'ರ', 'ವ'
        ]
        
        for suffix in vibhakti_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]
        
        return word

    def get_statistics(self) -> Dict:
        """Get validator statistics"""
        return {
            "total_validations": self.total_validations,
            "dictionary_size": len(self.dictionary_words),
            "context_enabled": self.context_enabled,
            "fuzzy_logic": "enabled",
            "cosine_similarity": "enabled",
            "fst_morphology": "enabled" if self.enable_fst else "disabled",
            "vibhakti_validation": "enabled" if self.enable_vibhakti_validation else "disabled"
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Validation Module - Fuzzy + Cosine + FST + Vibhakti Test")
    print("=" * 70)

    validator = EnhancedValidator(
        dictionary_paths=[
            "data/dictionaries/Padakosha_kannada_csv.csv",
            "data/dictionaries/combined_word_scrapped_csv.csv"
        ],
        context_db_path="data/ngram_context.db",
        enable_fst=True,
        enable_vibhakti_validation=True
    )
    
    print("\n--- Test 1: Vibhakti Error Detection and Correction ---")
    sentence = "ಕುಡಿಯುತ್ತೇನೆ"
    print(f"Input sentence: {sentence}")
    corrected = validator.correct_sentence(sentence)
    print(f"Corrected: {corrected}")
    
    print("\n--- Test 2: Single Word Vibhakti Error ---")
    word = "ಕುಡಿಯತೇನೆ "
    result = validator.validate_word(word, full_sentence=sentence, position=1)
    print(f"Word: {word}")
    print(f"Valid: {result['valid']}")
    print(f"Error Type: {result.get('error_type')}")
    print(f"Top 5 Suggestions: {result['suggestions'][:5]}")
    
    print("\n--- Test 3: FST Morphological Correction ---")
    sentence2 = "ನಾನು ನೀರನ್ನು ಕುಡಿಯ ಬೇಕು"
    print(f"Input sentence: {sentence2}")
    corrected2 = validator.correct_sentence(sentence2)
    print(f"Corrected: {corrected2}")

    print("\n--- Test 4: Cosine Similarity (No FST/Vibhakti) ---")
    w = "ಕನ್ರಾಟಕ"
    print(f"Input: {w}")
    r = validator.validate_word(w)
    print(f"Valid: {r['valid']}")
    print(f"Top 3 Suggestions: {r['suggestions'][:3]}")

    print("\n" + "=" * 70)
    print("✅ Vibhakti Validation Integration Complete!")
    print("=" * 70)