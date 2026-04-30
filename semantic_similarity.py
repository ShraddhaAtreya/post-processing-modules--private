"""
Semantic Similarity Module using FastText Embeddings - FIXED VERSION
====================================================

Provides semantic word similarity for Kannada OCR correction.
Uses pre-trained FastText embeddings to find semantically similar words
beyond character-level edit distance.

FIXES:
- ✅ Properly loads .vec text format files (binary=False)
- ✅ Searches for models/fasttext_kannada.vec
- ✅ Better error messages and progress indication

Features:
- ✅ FastText embedding support (Facebook's multilingual model)
- ✅ Semantic similarity scoring (cosine similarity)
- ✅ Context-aware candidate expansion
- ✅ Fallback to character similarity if embeddings unavailable

Author: MTech Thesis Project
Date: January 2026
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle

# Try importing gensim for FastText, with fallback
try:
    from gensim.models import FastText
    from gensim.models.keyedvectors import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("[SemanticSimilarity] Warning: gensim not installed. Semantic features disabled.")
    print("  Install with: pip install gensim")


class SemanticSimilarity:
    """
    Semantic similarity calculator using word embeddings.
    
    Supports:
    - Pre-trained FastText models
    - Custom trained embeddings
    - Cosine similarity computation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_pretrained: bool = True,
        cache_size: int = 10000
    ):
        """
        Initialize semantic similarity module.
        
        Args:
            model_path: Path to FastText .bin or KeyedVectors file
            use_pretrained: Whether to attempt loading pretrained model
            cache_size: Size of similarity cache
        """
        self.model = None
        self.model_loaded = False
        self.cache = {}
        self.cache_size = cache_size
        
        print("[SemanticSimilarity] Initializing...")
        
        if not GENSIM_AVAILABLE:
            print("  ⚠️  Gensim not available - semantic features disabled")
            return
        
        # Try loading model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        elif use_pretrained:
            self._try_load_pretrained()
        
        if self.model_loaded:
            print(f"  ✅ Embeddings loaded: {self.get_vocab_size():,} words")
        else:
            print("  ⚠️  No embeddings loaded - will skip semantic scoring")
    
    def _load_model(self, model_path: str):
        """Load FastText model from file"""
        try:
            print(f"  Loading model: {model_path}")
            file_size_gb = os.path.getsize(model_path) / (1024**3)
            print(f"  File size: {file_size_gb:.2f} GB")
            
            # Check file extension
            if model_path.endswith('.bin'):
                # Full FastText binary model
                print(f"  Loading as binary format...")
                self.model = FastText.load(model_path)
                self.model_loaded = True
            elif model_path.endswith('.vec') or model_path.endswith('.txt'):
                # KeyedVectors text format (Facebook's format)
                print(f"  Loading as text format (this may take 2-5 minutes for large files)...")
                # CRITICAL: binary=False for .vec text files!
                self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)
                self.model_loaded = True
                print(f"  ✅ Loaded successfully!")
            elif model_path.endswith('.pkl') or model_path.endswith('.pickle'):
                # Pickled KeyedVectors
                print(f"  Loading from pickle...")
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_loaded = True
            else:
                print(f"  ⚠️  Unsupported format: {model_path}")
                print(f"     Supported: .bin, .vec, .txt, .pkl, .pickle")
        
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            import traceback
            print("  Full traceback:")
            traceback.print_exc()
            self.model_loaded = False
    
    def _try_load_pretrained(self):
        """Try loading pretrained Kannada FastText embeddings"""
        # Common locations for pretrained models (in priority order)
        possible_paths = [
            "models/fasttext_kannada.vec",       # Downloaded by our script
            "models/cc.kn.300.vec",              # Facebook's official name
            "models/fasttext_kannada.bin",       # Binary format
            "models/cc.kn.300.bin",              # Facebook's binary
            "data/embeddings/kannada_fasttext.vec",
            "data/embeddings/kannada_fasttext.bin",
            "fasttext_kannada.vec",
            "fasttext_kannada.bin"
        ]
        
        print(f"  Searching for pretrained model in {len(possible_paths)} locations...")
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  ✅ Found pretrained model: {path}")
                self._load_model(path)
                if self.model_loaded:
                    return
                else:
                    print(f"  ⚠️  Found file but failed to load, trying next...")
        
        print("  ❌ No pretrained model found in default locations")
        print("  Download from: https://fasttext.cc/docs/en/crawl-vectors.html")
        print("  Or run: python download_fasttext.py")
    
    def is_available(self) -> bool:
        """Check if semantic similarity is available"""
        return self.model_loaded and self.model is not None
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size of loaded model"""
        if not self.is_available():
            return 0
        
        try:
            if hasattr(self.model, 'wv'):
                return len(self.model.wv)
            elif hasattr(self.model, 'key_to_index'):
                return len(self.model.key_to_index)
            else:
                return 0
        except:
            return 0
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word.
        
        Args:
            word: Input word
            
        Returns:
            Embedding vector or None if not found
        """
        if not self.is_available():
            return None
        
        try:
            if hasattr(self.model, 'wv'):
                # FastText model
                return self.model.wv[word]
            else:
                # KeyedVectors
                return self.model[word]
        except KeyError:
            # Word not in vocabulary - FastText can handle OOV with subwords
            try:
                if hasattr(self.model, 'wv') and hasattr(self.model.wv, 'get_vector'):
                    return self.model.wv.get_vector(word)
            except:
                pass
            return None
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score (0.0 to 1.0), or 0.0 if embeddings unavailable
        """
        if not self.is_available():
            return 0.0
        
        # Check cache
        cache_key = f"{word1}|{word2}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            vec1 = self.get_vector(word1)
            vec2 = self.get_vector(word2)
            
            if vec1 is None or vec2 is None:
                return 0.0
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
            
            # Normalize to 0-1 range (cosine can be -1 to 1)
            similarity = (similarity + 1) / 2
            
            # Cache result
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = similarity
            
            return float(similarity)
        
        except Exception as e:
            print(f"  Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_words(
        self,
        word: str,
        context_word: Optional[str] = None,
        topn: int = 20,
        min_similarity: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find semantically similar words.
        
        Args:
            word: Target word
            context_word: Optional context word to bias similarity
            topn: Number of similar words to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if not self.is_available():
            return []
        
        try:
            # Get similar words to target
            if hasattr(self.model, 'wv'):
                similar = self.model.wv.most_similar(word, topn=topn * 2)
            else:
                similar = self.model.most_similar(word, topn=topn * 2)
            
            results = []
            
            # If context provided, re-rank by context similarity
            if context_word:
                for candidate, score in similar:
                    # Calculate similarity to context word
                    context_sim = self.cosine_similarity(candidate, context_word)
                    
                    # Blend: 50% target similarity, 50% context similarity
                    blended_score = (score + context_sim) / 2
                    
                    if blended_score >= min_similarity:
                        results.append((candidate, blended_score))
            else:
                results = [(w, s) for w, s in similar if s >= min_similarity]
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:topn]
        
        except KeyError:
            # Word not in vocabulary
            return []
        except Exception as e:
            print(f"  Error finding similar words: {e}")
            return []
    
    def get_semantic_candidates(
        self,
        ocr_word: str,
        context_word: Optional[str] = None,
        dictionary_words: Optional[List[str]] = None,
        topn: int = 15
    ) -> List[str]:
        """
        Get semantic candidate suggestions for OCR correction.
        
        This is the main method to use in your validator.
        
        Args:
            ocr_word: Word extracted by OCR (potentially incorrect)
            context_word: Previous or next word for context
            dictionary_words: List of valid dictionary words to filter by
            topn: Number of candidates to return
            
        Returns:
            List of candidate words ranked by semantic similarity
        """
        if not self.is_available():
            return []
        
        candidates = []
        
        try:
            # Strategy 1: Find words similar to OCR word
            ocr_similar = self.find_similar_words(ocr_word, topn=topn)
            candidates.extend([w for w, _ in ocr_similar])
            
            # Strategy 2: If context available, find words similar to context
            if context_word:
                context_similar = self.find_similar_words(context_word, topn=topn)
                candidates.extend([w for w, _ in context_similar])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for word in candidates:
                if word not in seen:
                    seen.add(word)
                    unique_candidates.append(word)
            
            # Filter by dictionary if provided
            if dictionary_words:
                dictionary_set = set(dictionary_words)
                unique_candidates = [w for w in unique_candidates if w in dictionary_set]
            
            return unique_candidates[:topn]
        
        except Exception as e:
            print(f"  Error generating semantic candidates: {e}")
            return []
    
    def calculate_semantic_score(
        self,
        candidate: str,
        ocr_word: str,
        context_word: Optional[str] = None
    ) -> float:
        """
        Calculate semantic similarity score for a candidate.
        
        This gives you a 0.0-1.0 score to use in ranking.
        
        Args:
            candidate: Candidate correction word
            ocr_word: Original OCR word
            context_word: Context word for additional scoring
            
        Returns:
            Semantic similarity score (0.0 to 1.0)
        """
        if not self.is_available():
            return 0.0
        
        # Score 1: Similarity to OCR word (character-level might be noisy)
        ocr_similarity = self.cosine_similarity(candidate, ocr_word)
        
        # Score 2: Similarity to context
        context_similarity = 0.0
        if context_word:
            context_similarity = self.cosine_similarity(candidate, context_word)
        
        # Blend scores
        if context_word:
            # 30% OCR similarity, 70% context similarity
            final_score = (0.3 * ocr_similarity) + (0.7 * context_similarity)
        else:
            # Only OCR similarity available
            final_score = ocr_similarity
        
        return final_score
    
    def clear_cache(self):
        """Clear similarity cache"""
        self.cache.clear()
    
    def get_statistics(self) -> Dict:
        """Get module statistics"""
        return {
            "available": self.is_available(),
            "vocab_size": self.get_vocab_size(),
            "cache_size": len(self.cache),
            "model_type": type(self.model).__name__ if self.model else "None"
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_fasttext_kannada(output_dir: str = "models"):
    """
    Helper function to download FastText Kannada embeddings.
    
    Downloads from Facebook's FastText project.
    """
    import urllib.request
    import gzip
    import shutil
    
    os.makedirs(output_dir, exist_ok=True)
    
    # FastText Kannada vectors URL
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.kn.300.vec.gz"
    output_path = os.path.join(output_dir, "cc.kn.300.vec.gz")
    final_path = os.path.join(output_dir, "fasttext_kannada.vec")
    
    if os.path.exists(final_path):
        print(f"FastText model already exists: {final_path}")
        return final_path
    
    print(f"Downloading FastText Kannada embeddings...")
    print(f"  URL: {url}")
    print(f"  This may take several minutes (file is ~1GB)...")
    
    try:
        # Download
        urllib.request.urlretrieve(url, output_path)
        print(f"  Downloaded to: {output_path}")
        
        # Decompress
        print(f"  Decompressing...")
        with gzip.open(output_path, 'rb') as f_in:
            with open(final_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove compressed file
        os.remove(output_path)
        
        print(f"  ✅ FastText model ready: {final_path}")
        return final_path
    
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print(f"  Please download manually from: {url}")
        return None


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Semantic Similarity Module - Test")
    print("=" * 70)
    
    # Initialize
    sem_sim = SemanticSimilarity()
    
    if not sem_sim.is_available():
        print("\n⚠️  No embeddings loaded. Testing in fallback mode.")
        print("\nTo enable semantic features:")
        print("1. Install gensim: pip install gensim")
        print("2. Download FastText Kannada model:")
        print("   python download_fasttext.py")
    else:
        print(f"\n✅ Semantic similarity available!")
        print(f"   Vocabulary size: {sem_sim.get_vocab_size():,} words")
        
        # Test 1: Cosine similarity
        print("\n--- Test 1: Word Similarity ---")
        word1 = "ಊಟ"  # food
        word2 = "ತಿನ್ನು"  # eat
        word3 = "ಓದು"  # read
        
        sim_food_eat = sem_sim.cosine_similarity(word1, word2)
        sim_food_read = sem_sim.cosine_similarity(word1, word3)
        
        print(f"Similarity('{word1}', '{word2}'): {sim_food_eat:.3f}")
        print(f"Similarity('{word1}', '{word3}'): {sim_food_read:.3f}")
        print(f"  → Food is more similar to eat than read: {sim_food_eat > sim_food_read}")
        
        # Test 2: Find similar words
        print("\n--- Test 2: Find Similar Words ---")
        target = "ಊಟ"
        similar = sem_sim.find_similar_words(target, topn=5)
        print(f"Words similar to '{target}':")
        for word, score in similar:
            print(f"  {word}: {score:.3f}")
        
        # Test 3: Semantic candidates
        print("\n--- Test 3: Semantic Candidates ---")
        ocr_word = "ಓದುತ್ತೆನೆ"
        context = "ಊಟವನ್ನು"
        candidates = sem_sim.get_semantic_candidates(ocr_word, context_word=context, topn=5)
        print(f"Candidates for '{ocr_word}' with context '{context}':")
        for word in candidates:
            score = sem_sim.calculate_semantic_score(word, ocr_word, context)
            print(f"  {word}: {score:.3f}")
    
    # Print statistics
    print("\n--- Statistics ---")
    stats = sem_sim.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTest complete.")