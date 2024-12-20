

## Scripts Overview

### avg_compression.py
This script calculates the average compression factor from a JSONL file. It extracts the `factor` field and computes the average across all entries, returning 0 if no factors are found.

### bleu.py
Evaluates summaries in a JSONL file using BLEU and METEOR metrics, calculating averages for both. Ideal for summarization performance evaluation.

### crystal0.py
Processes a JSONL file by applying `<GIST>` tokens to docstrings and combining them with function signatures. Outputs a new dataset for model training or evaluation.

### gen_gpt4_codereval.py
Generates valid Java code using OpenAI GPT-4 API, based on method signatures and docstrings. Saves generated code and metadata for further evaluation.

### gen_gpt4_sum.py
Uses OpenAI GPT-4 API to generate concise summaries for Python code snippets, adding the results to a JSONL file for analysis.

### gen_llama_sum.py
Generates concise summaries for Python functions using the CodeLLaMA model. Saves original code, reference summaries, and generated summaries for evaluation.

### gen_model.py
Generates multiple samples per prompt using a fine-tuned model, creating diverse outputs for evaluation or further analysis.

### lora_train.py
Fine-tunes a T5-large model with LoRA (Low-Rank Adaptation) and 4-bit quantization for efficient training. Includes metrics like ROUGE and BLEU for evaluation.

### ngrams.py
Extracts n-grams from docstring tokens in a dataset, counts their frequencies, and saves the results. Useful for shared n-gram analysis.

### passk.py
Calculates Pass@k metrics (k=1, 3, 5, 10, 20) for model outputs. Evaluates the model’s performance based on varying prediction tolerances.

### py_extract.py
Processes a JSONL file by extracting function signatures and applying `<GIST>` tokens to docstrings. Outputs a new dataset for prompt optimization.

### remove_ngrams.py
Removes the top 500 most frequent shared n-grams from the `code` field in a dataset. Calculates reduction factors and saves filtered code for training.

### rouge.py
Evaluates fine-tuned models using ROUGE and BLEU metrics. Processes a test dataset, compares generated summaries with references, and calculates averages.

### sc.py
Uses the SelectiveContext library to compress JSONL content, reducing context size by 50% while retaining essential information for resource-constrained NLP tasks.

### train_model.py
Fine-tunes the CodeT5 model for code generation tasks, incorporating shared n-gram filtering and metrics like CrystalBLEU, ROUGE, and BLEU for robust evaluation.

### train_sum.py
Fine-tunes the Salesforce/codet5p-220m model for code summarization using a dataset of Python code and summaries. Includes features like gradient accumulation and mixed precision training.

---

## Setup and Dependencies

1. Install required Python libraries:
   ```bash
   pip install transformers datasets evaluate nltk tqdm
