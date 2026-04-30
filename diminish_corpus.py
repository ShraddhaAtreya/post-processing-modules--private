import os

# Use raw string or forward slashes for Windows paths
input_file = "F:/INTERFACE_GRADIO/data/corpus/kn.txt"
output_file = "F:/INTERFACE_GRADIO/data/corpus/kannada_corpus_subset.txt"
sample_ratio = 0.2  # Keep 20%

# Verify input file exists
if not os.path.exists(input_file):
    print(f"ERROR: Input file not found at {input_file}")
    exit()

print(f"Input file: {input_file}")
print(f"Output file: {output_file}")
print(f"Sample ratio: {sample_ratio * 100}%\n")

# First pass: count total lines
print("Counting lines...")
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

print(f"Total lines: {total_lines:,}")
estimated_output_lines = int(total_lines * sample_ratio)
print(f"Expected output lines: ~{estimated_output_lines:,}\n")

# Calculate sampling interval
interval = int(1 / sample_ratio)
print(f"Sampling every {interval}th line...\n")

# Second pass: extract
line_count = 0
with open(input_file, 'r', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if i % interval == 0:
                f_out.write(line)
                line_count += 1
            if i % 1000000 == 0:
                print(f"Processed {i:,} lines... (wrote {line_count:,} lines)")

print(f"\nDone!")
print(f"Total lines written: {line_count:,}")
print(f"Output file size: {os.path.getsize(output_file) / (1024**3):.2f} GB")
print(f"Saved to: {output_file}")