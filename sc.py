from selective_context import SelectiveContext  # Ensure SelectiveContext is accessible

# Initialize SelectiveContext with the desired configuration
sc = SelectiveContext(model_type='gpt2', lang='en')

# Path to the JSONL file
file_path = "/home/20231567/dataforgisting/sc/filtered_docstrings.jsonl"

# Read all lines from the input file
with open(file_path, "r") as infile:
    lines = infile.readlines()

# Compress each line and store in a list
compressed_lines = []
for line in lines:
    # Remove any extraneous whitespace from each line
    text = line.strip()
    
    # Apply compression with desired reduce_ratio and reduce_level
    compressed_context, _ = sc(text, reduce_ratio=0.5, reduce_level='phrase')
    
    # Add the compressed text to the list
    compressed_lines.append(compressed_context)

# Overwrite the original file with compressed content in the same format
with open(file_path, "w") as outfile:
    outfile.write('\n'.join(compressed_lines) + '\n')

print("Compression completed. The file has been updated with compressed data.")