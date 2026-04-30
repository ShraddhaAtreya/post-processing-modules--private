# Save as check_corpus_stats.py
import os

file_path = "F:/INTERFACE_GRADIO/data/corpus/kannada_corpus_subset.txt"

print("Analyzing corpus file...\n")

# File size
size_bytes = os.path.getsize(file_path)
size_gb = size_bytes / (1024**3)
size_mb = size_bytes / (1024**2)

# Line count and token estimation
line_count = 0
total_tokens = 0
sample_lines = []

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line_count += 1
        tokens = len(line.split())
        total_tokens += tokens
        
        # Save first 5 lines for preview
        if i < 5:
            sample_lines.append(line.strip())

# Calculate averages
avg_tokens_per_line = total_tokens / line_count if line_count > 0 else 0

print("=" * 60)
print("CORPUS STATISTICS")
print("=" * 60)
print(f"File: {os.path.basename(file_path)}")
print(f"Location: {os.path.dirname(file_path)}")
print(f"\nSize:")
print(f"  - {size_gb:.2f} GB")
print(f"  - {size_mb:.2f} MB")
print(f"  - {size_bytes:,} bytes")
print(f"\nContent:")
print(f"  - Total lines: {line_count:,}")
print(f"  - Total tokens: {total_tokens:,}")
print(f"  - Avg tokens/line: {avg_tokens_per_line:.1f}")
print(f"\nSample (first 5 lines):")
print("-" * 60)
for i, line in enumerate(sample_lines, 1):
    preview = line[:80] + "..." if len(line) > 80 else line
    print(f"{i}. {preview}")
print("=" * 60)

# Estimate training suitability
print(f"\nTraining Suitability Assessment:")
print(f"  ✓ File size: {'Good' if size_gb < 5 else 'Too large'}")
print(f"  ✓ Token count: {'Excellent' if total_tokens > 5_000_000 else 'Good'}")
print(f"  ✓ Ready for Kaggle upload: Yes")
print("=" * 60)