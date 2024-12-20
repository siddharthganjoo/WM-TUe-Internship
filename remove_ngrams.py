import json
from nltk.util import ngrams
from collections import Counter
from tqdm import tqdm

# Load shared n-grams with top 500 frequencies
def load_top_ngrams(shared_ngrams_file, k=500):
    with open(shared_ngrams_file, 'r') as f:
        shared_ngrams = json.load(f)
    top_ngrams = {eval(key) for key, _ in Counter(shared_ngrams).most_common(k)}
    return top_ngrams

# Function to tokenize the code and remove shared n-grams
def remove_shared_ngrams(code, top_ngrams):
    tokens = code.split()
    filtered_tokens = []
    max_n = max(len(ngram) for ngram in top_ngrams)
    i = 0
    while i < len(tokens):
        matched = False
        for n in range(max_n, 0, -1):
            if i + n <= len(tokens):
                current_ngram = tuple(tokens[i:i+n])
                if current_ngram in top_ngrams:
                    matched = True
                    i += n - 1
                    break
        if not matched:
            filtered_tokens.append(tokens[i])
        i += 1
    return ' '.join(filtered_tokens), len(tokens), len(filtered_tokens)

# Load shared n-grams file
shared_ngrams_file = "/home/sganjoo/internship/python_shared_ngrams.json"
top_ngrams = load_top_ngrams(shared_ngrams_file, k=500)

# Process the JSONL file
input_file = "/home/sganjoo/internship/dataset/python/train_nocomments.jsonl"
output_file = "/home/sganjoo/internship/dataset/python/train_processed.jsonl"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in tqdm(infile, desc="Processing lines"):
        data = json.loads(line)
        original_code = data.get("code", "")
        filtered_code, original_token_count, filtered_token_count = remove_shared_ngrams(original_code, top_ngrams)
        
        # Calculate reduction factor
        reduction_factor = 100 * (original_token_count - filtered_token_count) / original_token_count if original_token_count > 0 else 0
        
        # Update the fields
        data["code"] = original_code  # Keep the original code
        data["filtered_code"] = filtered_code
        data["factor"] = reduction_factor

        # Write updated data to output file
        json.dump(data, outfile)
        outfile.write("\n")

print("Processing completed. Processed data saved to:", output_file)