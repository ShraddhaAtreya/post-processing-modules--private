"""
Finalize N-Gram Database (Run this if you stopped training early)
===============================================================
This creates the necessary indexes so the Validator works fast.
"""
import sqlite3
import time

DB_PATH = "data/ngram_context.db"

def finalize():
    print(f"🔌 Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("⏳ Building Indexes (This makes lookup instant)...")
    print("   1. Indexing Unigrams...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_unigram_count ON unigrams(count)")
    
    print("   2. Indexing Bigrams (Previous Word)...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bigram_prev ON bigrams(prev_word)")
    
    # Optional: Index strictly for the query we use (prev_word, next_word)
    print("   3. Optimizing Bigram Lookup...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bigram_pair ON bigrams(prev_word, next_word)")

    conn.commit()
    conn.close()
    print("✅ Database Finalized! You can now run the UI.")

if __name__ == "__main__":
    finalize()