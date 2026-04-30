import os
import shutil
try:
    from validation import KannadaWordValidator
except Exception:
    KannadaWordValidator = None


def ensure_dirs():
    os.makedirs(os.path.join('data', 'dictionaries'), exist_ok=True)


def move_file(src, dst_dir):
    if not src:
        return
    if not os.path.exists(src):
        return
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    shutil.move(src, dst)


def initialize(dictionary_file='Padakosha_kannada_csv.csv', corpus_file='combined_word_scrapped_csv.csv'):
    ensure_dirs()
    dict_dir = os.path.join('data', 'dictionaries')

    # Try common locations for provided files
    candidates = [os.path.join(os.getcwd(), dictionary_file), dictionary_file]
    for c in candidates:
        if os.path.exists(c):
            move_file(c, dict_dir)
            break

    candidates = [os.path.join(os.getcwd(), corpus_file), corpus_file]
    for c in candidates:
        if os.path.exists(c):
            move_file(c, dict_dir)
            break

    # Initialize validator if available (best-effort)
    if KannadaWordValidator is not None:
        try:
            v = KannadaWordValidator()
            print(f"Initialized validator. Dictionary words: {len(v.dictionary_words)}, Corpus words: {len(v.corpus_words_set)}")
        except Exception:
            print("Validator initialization failed (continuing)")


if __name__ == '__main__':
    initialize()
