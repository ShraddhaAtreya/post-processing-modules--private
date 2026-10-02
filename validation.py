import os
import re
import math
import json
import sqlite3
from collections import defaultdict
from datetime import datetime

try:
    from rapidfuzz.distance import Levenshtein
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


class KannadaWordValidator:
    def __init__(self, dictionary_path=None, corpus_path=None, db_path=None):
        # Default paths
        self.dictionary_path = dictionary_path or os.path.join('data', 'dictionaries', 'Padakosha_kannada_csv.csv')
        self.corpus_path = corpus_path or os.path.join('data', 'dictionaries', 'combined_word_scrapped_csv.csv')
        self.db_path = db_path or os.path.join('data', 'user_corrections.db')

        # Data structures
        self.dictionary_words = set()
        self.corpus_words_set = set()
        self.corpus_freq = {}
        self.all_words = set()
        self.first_char_map = defaultdict(list)
        self.length_map = defaultdict(list)
        self.vibhaktis = []

        # SQLite connection
        self.db_conn = None

        # Initialize sources
        self._init_database()
        self._load_dictionary()
        self._load_corpus()
        self._build_indices()
        self._init_vibhaktis()
        
        print(f"Sources loaded: dictionary={len(self.dictionary_words)}, corpus={len(self.corpus_words_set)}, total_words={len(self.all_words)}")

    def _init_database(self):
        """Initialize SQLite database with schema"""
        try:
            # Create data directory if needed
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to database
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_word TEXT NOT NULL,
                    corrected_word TEXT NOT NULL,
                    correction_count INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(original_word, corrected_word)
                )
            """)
            
            # Create indices for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_original 
                ON user_corrections(original_word)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_corrected 
                ON user_corrections(corrected_word)
            """)
            
            self.db_conn.commit()
            
        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
            self.db_conn = None

    def _load_dictionary(self):
        """Load dictionary from CSV"""
        if not os.path.exists(self.dictionary_path):
            return
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    w = parts[0].strip()
                    if w and len(w) > 1:  # Filter single-char garbage
                        self.dictionary_words.add(w)
        except Exception as e:
            print(f"Warning: Could not load dictionary: {e}")

    def _load_corpus(self):
        """Load corpus from CSV"""
        if not os.path.exists(self.corpus_path):
            return
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    w = parts[0].strip()
                    if w and len(w) > 1:  # Filter single-char garbage
                        self.corpus_words_set.add(w)
                        # Try to get frequency if available
                        if len(parts) > 1:
                            try:
                                freq = int(parts[1])
                                self.corpus_freq[w] = freq
                            except:
                                pass
        except Exception as e:
            print(f"Warning: Could not load corpus: {e}")

    def _init_vibhaktis(self):
        """Initialize vibhakti suffixes"""
        items = [
            'ಯವರು', 'ಯವರ', 'ಗಳನ್ನು', 'ಗಳಿಗೆ', 'ಗಳಿಂದ', 'ಗಳಲ್ಲಿ', 'ಗಳಲ್ಲಿನ',
            'ಯನ್ನು', 'ಯಿಂದ', 'ಯಲ್ಲಿ', 'ಯಲ್ಲಿನ', 'ವರು', 'ವರ', 'ಗಳು', 'ಗಳ',
            'ಅನ್ನು', 'ಇಂದ', 'ಅಲ್ಲಿ', 'ಅಲ್ಲಿನ', 'ನ್ನು',
            'ಗೆ', 'ನು', 'ಯ', 'ರ', 'ವ', 'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಎ', 'ಏ', 'ಒ', 'ಓ'
        ]
        items = [i.strip() for i in items]
        items = sorted(set(items), key=lambda x: len(x), reverse=True)
        self.vibhaktis = items

    def _build_indices(self):
        """Build search indices for fast lookup"""
        self.all_words = set(self.dictionary_words) | set(self.corpus_words_set)
        for w in self.all_words:
            if not w:
                continue
            self.first_char_map[w[0]].append(w)
            self.length_map[len(w)].append(w)

    def _check_database_corrections(self, word, max_results=10):
        """Check database for user corrections"""
        if not self.db_conn or not word:
            return []
        
        try:
            cursor = self.db_conn.cursor()
            
            # Query for corrections where this word was original
            cursor.execute("""
                SELECT corrected_word, correction_count, confidence
                FROM user_corrections
                WHERE original_word = ?
                ORDER BY correction_count DESC, confidence DESC
                LIMIT ?
            """, (word, max_results))
            
            results = []
            for row in cursor.fetchall():
                corrected, count, conf = row
                results.append({
                    'word': corrected,
                    'distance': 0,  # Exact match from DB
                    'confidence': float(conf),
                    'source': 'database',
                    'source_priority': 0,
                    'count': count
                })
            
            return results
            
        except Exception as e:
            print(f"Warning: Database query error: {e}")
            return []

    def save_user_correction(self, original_word, corrected_word, confidence=1.0):
        """Save user correction to database"""
        if not self.db_conn or not original_word or not corrected_word:
            return
        
        try:
            cursor = self.db_conn.cursor()
            
            # Insert or update correction
            cursor.execute("""
                INSERT INTO user_corrections 
                    (original_word, corrected_word, correction_count, confidence, last_used)
                VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(original_word, corrected_word) 
                DO UPDATE SET 
                    correction_count = correction_count + 1,
                    last_used = CURRENT_TIMESTAMP
            """, (original_word, corrected_word, confidence))
            
            self.db_conn.commit()
            
        except Exception as e:
            print(f"Warning: Could not save correction: {e}")

    def is_valid_word(self, word):
        """Check if word is valid"""
        if not word or len(word) <= 1:  # Filter single-char garbage
            return False
        
        # Exact match in dictionary (highest priority)
        if word in self.dictionary_words:
            return True
        
        # Check in all words
        if word in self.all_words:
            return True
        
        # Vibhakti stripping
        for v in self.vibhaktis:
            if word.endswith(v):
                root = word[:-len(v)] if len(v) > 0 else word
                if root and root in self.dictionary_words:
                    return True
        
        # Proper noun heuristic (6+ chars)
        if len(word) >= 6:
            return True
        
        return False

    def _levenshtein_distance(self, a, b, max_distance=None):
        """Calculate Levenshtein distance with optional early stopping"""
        if _HAS_RAPIDFUZZ:
            return Levenshtein.distance(a, b)
        
        # Fallback DP implementation
        la, lb = len(a), len(b)
        if max_distance is not None and abs(la - lb) > max_distance:
            return max_distance + 1
        
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * lb
            min_cur = cur[0]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
                if cur[j] < min_cur:
                    min_cur = cur[j]
            if max_distance is not None and min_cur > max_distance:
                return max_distance + 1
            prev = cur
        return prev[lb]

    def find_closest_matches(self, word, max_distance=3, top_n=10):
        """
        Find closest matches with priority-based search.
        
        Priority:
        1. Database (user corrections) - highest
        2. Dictionary (official words)
        3. Corpus (scraped words) - lowest
        
        Returns list of dicts with word, distance, confidence, source
        """
        word = str(word).strip()
        if not word or len(word) <= 1:  # Filter garbage
            return []

        matches_map = {}  # word -> best match dict

        # PRIORITY 0: Database user corrections
        db_matches = self._check_database_corrections(word, max_results=5)
        for match in db_matches:
            cand = match['word']
            matches_map[cand] = match

        # Helper to collect candidates from word sets
        def collect_from_pool(pool, source_name, source_priority):
            first = word[0] if word else ''
            cand_set = set()
            
            # First char matching
            if first:
                bucket = self.first_char_map.get(first, [])
                for w in bucket:
                    if abs(len(w) - len(word)) <= 3:
                        cand_set.add(w)
            
            # Length-based matching
            for l in range(max(1, len(word)-3), len(word)+4):
                for w in self.length_map.get(l, []):
                    cand_set.add(w)

            for cand in cand_set:
                if cand in matches_map:
                    continue  # Already have better match
                
                dist = self._levenshtein_distance(word, cand, max_distance)
                if dist <= max_distance:
                    base_conf = 1.0 - (dist / max(len(word), len(cand)))
                    conf = base_conf
                    
                    # Boost for corpus frequency
                    if source_name == 'corpus':
                        freq = self.corpus_freq.get(cand, 0)
                        if freq > 0:
                            boost = min(math.log(freq + 1) / 10.0, 0.25)
                            conf = min(1.0, conf + boost)
                    
                    matches_map[cand] = {
                        'word': cand,
                        'distance': dist,
                        'confidence': conf,
                        'source': source_name,
                        'source_priority': source_priority
                    }

        # PRIORITY 1: Dictionary
        collect_from_pool(self.dictionary_words, 'dictionary', 1)
        
        # PRIORITY 2: Corpus
        collect_from_pool(self.corpus_words_set, 'corpus', 2)

        # Sort by: source_priority (asc), distance (asc), confidence (desc)
        matches = sorted(
            matches_map.values(),
            key=lambda x: (x['source_priority'], x['distance'], -x['confidence'])
        )

        return matches[:top_n]

    def validate_and_correct(self, word, confidence_threshold=0.7):
        """
        Validate and correct a word.
        
        FIXED: Returns structure compatible with Gradio UI:
        {
            'valid': True/False,
            'suggestions': ['word1', 'word2', ...]  # Simple list
        }
        
        IMPORTANT: ALWAYS returns suggestions (even for valid words)
        This allows users to see alternatives and verify correctness.
        """
        # Filter garbage input
        if not word or len(word) <= 1:
            # For single-char, try to find words starting with that char
            if word and len(word) == 1:
                first_char_words = self.first_char_map.get(word, [])[:10]
                return {
                    'valid': False,
                    'suggestions': first_char_words
                }
            return {
                'valid': False,
                'suggestions': []
            }
        
        word = word.strip()
        
        # Check if valid
        is_valid = self.is_valid_word(word)
        
        # ALWAYS get suggestions (even for valid words)
        # This helps users verify and see alternatives
        matches = self.find_closest_matches(word, max_distance=3, top_n=10)
        
        # Extract just the words for Gradio UI
        suggestions = [m['word'] for m in matches]
        
        # If word is valid but no close matches, get words starting with same letter
        if is_valid and not suggestions:
            first_char = word[0] if word else ''
            if first_char:
                similar_words = self.first_char_map.get(first_char, [])
                # Get words of similar length
                suggestions = [w for w in similar_words if abs(len(w) - len(word)) <= 2][:10]
        
        return {
            'valid': is_valid,
            'suggestions': suggestions  # Always populated!
        }

    def validate_text(self, text):
        """Validate entire text and return corrections"""
        print("🔎 Starting text validation...")
        
        # Split preserving Kannada words
        parts = re.split(r'([\u0C80-\u0CFF]+)', text)
        corrected_parts = []
        word_validations = []
        
        total_words = 0
        valid_words = 0
        corrected_words = 0
        unknown_words = 0
        
        for part in parts:
            if not part:
                continue
            
            if re.fullmatch(r'[\u0C80-\u0CFF]+', part):
                # Filter single-char garbage
                if len(part) <= 1:
                    continue
                
                total_words += 1
                res = self.validate_and_correct(part)
                
                # Convert to old format for compatibility
                word_validations.append({
                    'word': part,
                    'is_valid': res['valid'],
                    'suggestions': res['suggestions']
                })
                
                if res['valid']:
                    valid_words += 1
                    corrected_parts.append(part)
                else:
                    # Use first suggestion if available
                    if res['suggestions']:
                        corrected_parts.append(res['suggestions'][0])
                        corrected_words += 1
                    else:
                        corrected_parts.append(part)
                        unknown_words += 1
            else:
                corrected_parts.append(part)
        
        corrected_text = ''.join(corrected_parts)
        
        stats = {
            'total_words': total_words,
            'valid_words': valid_words,
            'corrected_words': corrected_words,
            'unknown_words': unknown_words
        }
        
        print(f"🔎 Validation complete: {valid_words} valid, {corrected_words} corrected, {unknown_words} unknown out of {total_words} words")
        
        return {
            'original_text': text,
            'corrected_text': corrected_text,
            'word_validations': word_validations,
            'stats': stats
        }

    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


if __name__ == "__main__":
    # Test the validator
    validator = KannadaWordValidator()
    
    print("\n" + "="*60)
    print("TESTING KANNADA WORD VALIDATOR")
    print("="*60)
    
    # Test words
    test_words = ["ನಾನು", "ಶಾಲೆಗೆ", "ಬೆಂಗಳೂರು", "ಸ್ಕೂಲಿಗೆ", "i", "a"]
    
    for word in test_words:
        print(f"\n📝 Testing: '{word}'")
        result = validator.validate_and_correct(word)
        print(f"   Valid: {result['valid']}")
        if result['suggestions']:
            print(f"   Suggestions: {result['suggestions'][:3]}")
        else:
            print(f"   Suggestions: None")
    
    # Test saving correction
    print("\n" + "="*60)
    print("TESTING USER CORRECTION SAVE")
    print("="*60)
    validator.save_user_correction("ಬೆಳ್ಳಗೆ", "ಬೆಳಗ್ಗೆ", confidence=0.95)
    print("✅ Saved: ಬೆಳ್ಳಗೆ → ಬೆಳಗ್ಗೆ")
    
    # Test retrieval
    result = validator.validate_and_correct("ಬೆಳ್ಳಗೆ")
    print(f"\n📝 Testing retrieval: 'ಬೆಳ್ಳಗೆ'")
    print(f"   Suggestions: {result['suggestions'][:3]}")
    print(f"   (Should show 'ಬೆಳಗ್ಗೆ' first from database)")
    
    validator.close()
    print("\n✅ All tests complete!")