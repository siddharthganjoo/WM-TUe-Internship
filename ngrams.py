import json
from collections import Counter
from nltk.util import ngrams

# Step 1: Load your dataset from a JSONL file
def load_code_tokens(file_path):
    code_tokens = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Collect tokens from the "code_tokens" field
            if "code_tokens" in data:
                code_tokens.extend(data["docstring_tokens"])
    return code_tokens

# Step 2: Generate n-grams from the dataset and save their frequencies
def extract_all_ngrams(tokens, max_n=4):
    all_ngrams = []
    for n in range(1, max_n + 1):
        all_ngrams.extend(ngrams(tokens, n))
    
    # Count frequencies of all n-grams
    frequencies = Counter(all_ngrams)
    
    return frequencies

# Load tokens from your dataset (adjust the path as needed)
dataset_path = "/home/sganjoo/internship/dataset/python/train.jsonl"
tokens = load_code_tokens(dataset_path)

# Extract all n-grams
ngrams_frequencies = extract_all_ngrams(tokens, max_n=4)

# Step 3: Save the n-grams frequencies to a file
with open("sum_ngrams_python.json", "w") as f:
    json.dump({str(k): v for k, v in ngrams_frequencies.items()}, f, indent=4)

print("All shared n-grams extracted and saved to 'all_shared_ngrams.json'")