def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two sequences (strings or lists)."""
    # Allow lists/tuples for word-level WER
    if isinstance(s1, (list, tuple)) or isinstance(s2, (list, tuple)):
        a = list(s1)
        b = list(s2)
    else:
        a = list(s1)
        b = list(s2)

    if len(a) < len(b):
        a, b = b, a

    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, c1 in enumerate(a, 1):
        current_row = [i]
        for j, c2 in enumerate(b, 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j-1] + 1
            substitutions = previous_row[j-1] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER)
    Returns float between 0 and 1
    """
    if reference is None:
        return 0.0
    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "") if hypothesis is not None else ""
    if len(ref_chars) == 0:
        return 0.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER)
    Returns float between 0 and 1
    """
    if reference is None:
        return 0.0
    ref_words = reference.split()
    hyp_words = hypothesis.split() if hypothesis is not None else []
    if len(ref_words) == 0:
        return 0.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def calculate_metrics(reference_text, extracted_text):
    cer = calculate_cer(reference_text, extracted_text)
    wer = calculate_wer(reference_text, extracted_text)
    return {
        'cer': cer,
        'wer': wer,
        'cer_display': f"{cer:.2%} ({cer:.4f})",
        'wer_display': f"{wer:.2%} ({wer:.4f})"
    }
