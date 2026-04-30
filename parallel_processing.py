# =============================================================
# parallel_processor.py
# Kannada OCR Post-Processing System
# Batch + parallel word processing
# Integrates with: gradio_ui_main.py, new_validator.py
# =============================================================

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict


# -------------------------------------------------------------
# BATCH SIZE CALCULATION
# Agreed logic:
#   batch_size = total_words / 2
#   if batch_size > 20: batch_size = batch_size / 2
#   if batch_size > 20: batch_size = 20   (hard cap)
#   if batch_size < 10: process all in one batch
# -------------------------------------------------------------

def calculate_batch_size(total_words: int) -> int:
    """
    Dynamically calculate batch size based on paragraph length.

    Rules:
        1. batch_size = total_words / 2
        2. If batch_size > 20: batch_size = batch_size / 2
        3. If batch_size > 20: cap at 20
        4. If batch_size < 10: return total_words (single batch)

    Args:
        total_words: Total number of words in the paragraph.

    Returns:
        Calculated batch size as an integer.
    """
    if total_words <= 0:
        return 1

    batch_size = total_words / 2

    if batch_size > 20:
        batch_size = batch_size / 2

    if batch_size > 20:
        batch_size = 20  # hard cap

    batch_size = math.floor(batch_size)

    # If batch size is too small, no point splitting — one batch
    if batch_size < 10:
        return total_words

    return batch_size


def split_into_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches of the given size.
    The last batch may be smaller if items don't divide evenly.

    Args:
        items:      The full list to split.
        batch_size: Maximum number of items per batch.

    Returns:
        List of batches (each batch is a list of items).
    """
    if batch_size <= 0 or batch_size >= len(items):
        return [items]

    batches = []
    for start in range(0, len(items), batch_size):
        batches.append(items[start: start + batch_size])
    return batches


# -------------------------------------------------------------
# PARALLEL BATCH PROCESSOR
# -------------------------------------------------------------

def process_words_in_batches(
    word_list: List[Any],
    process_fn: Callable[[Any], Any],
    max_workers: int = 8,
) -> List[Any]:
    """
    Process a list of words in parallel batches.

    Each word is processed independently by process_fn.
    Results are returned in the SAME ORDER as the input word_list.
    If a single word fails, it returns None for that position
    without affecting other words.

    Args:
        word_list:   List of items to process (word dicts or strings).
        process_fn:  Function that takes one item and returns a result.
        max_workers: Maximum number of parallel threads (default 8).

    Returns:
        List of results in original word order.
        Failed items return None at their position.
    """
    if not word_list:
        return []

    total_words = len(word_list)
    batch_size  = calculate_batch_size(total_words)

    print(f"[ParallelProcessor] {total_words} words → batch size: {batch_size}")

    batches     = split_into_batches(word_list, batch_size)
    num_batches = len(batches)

    print(f"[ParallelProcessor] {num_batches} batch(es) with up to {max_workers} workers")

    # results dict: original index → result
    # This guarantees order is preserved regardless of thread completion order
    results: Dict[int, Any] = {}

    # Build a flat index map so we know the original position of every item
    indexed_words = list(enumerate(word_list))  # [(0, item0), (1, item1), ...]

    for batch_num, batch in enumerate(split_into_batches(indexed_words, batch_size)):
        print(f"[ParallelProcessor] Processing batch {batch_num + 1}/{num_batches} "
              f"({len(batch)} words)...")

        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
            # Submit all words in this batch
            future_to_idx = {
                executor.submit(process_fn, item): orig_idx
                for orig_idx, item in batch
            }

            for future in as_completed(future_to_idx):
                orig_idx = future_to_idx[future]
                try:
                    results[orig_idx] = future.result()
                except Exception as e:
                    print(f"[ParallelProcessor] Error at word index {orig_idx}: {e}")
                    results[orig_idx] = None  # safe fallback — caller handles None

    # Reconstruct in original order
    ordered_results = [results.get(i) for i in range(total_words)]

    print(f"[ParallelProcessor] Done. {total_words} words processed.")
    return ordered_results