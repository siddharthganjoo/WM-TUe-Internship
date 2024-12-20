import os
import torch
import json
import ast
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from crystalbleu import corpus_bleu
from collections import Counter
import evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Step 1: Load the extracted n-grams frequencies from file
def load_all_ngrams(file_path):
    with open(file_path, "r") as f:
        all_ngrams = json.load(f)
    # Convert string keys back to tuples
    return {ast.literal_eval(k): v for k, v in all_ngrams.items()}

# Step 2: Function to dynamically select top `k` trivially shared n-grams
def get_trivially_shared_ngrams(all_ngrams, k):
    # Sort n-grams by frequency and pick the top `k`
    sorted_ngrams = Counter(all_ngrams).most_common(k)
    return dict(sorted_ngrams)

# Load all shared n-grams from your saved file
all_ngrams_path = "/home/20231567/dataforgisting/python_shared_ngrams.json"  # Adjust the path to your JSON file
all_ngrams = load_all_ngrams(all_ngrams_path)

# Choose the value of `k` to dynamically select top `k` shared n-grams
k = 100  # You can adjust this number based on your requirements
trivially_shared_ngrams = get_trivially_shared_ngrams(all_ngrams, k)

# Load tokenizer and add <GIST> token if needed
model_name_or_path = 'Salesforce/codet5p-220m'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if "<GIST>" not in tokenizer.get_vocab():
    tokenizer.add_tokens(["<GIST>"])

# Load CodeT5 model
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_or_path, 
    cache_dir='./models'
)

# Adjust token embeddings to accommodate new tokens if added
model.resize_token_embeddings(len(tokenizer))

# Load your custom dataset
train_dataset = load_dataset("json", data_files="/home/20231567/dataforgisting/pytrain100.jsonl")["train"]
validation_dataset = load_dataset("json", data_files="/home/20231567/dataforgisting/pyvalid100n.jsonl")["train"]
test_dataset = load_dataset("json", data_files="/home/20231567/dataforgisting/pytest100.jsonl")["train"]

# Preprocessing function
def preprocess_function(example):
    input_text = example["Signature + Gisted Docstring"]
    output_text = example["code"]

    # Tokenize inputs and outputs
    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=256)
    outputs = tokenizer(output_text, padding="max_length", truncation=True, max_length=256)

    # Set up labels for training; mask padding tokens
    inputs["labels"] = outputs["input_ids"]
    inputs["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in inputs["labels"]]
    
    return inputs

# Apply preprocessing
processed_train_dataset = train_dataset.map(preprocess_function, batched=True)
processed_test_dataset = test_dataset.map(preprocess_function, batched=True)
processed_valid_dataset = validation_dataset.map(preprocess_function, batched=True)

# Define evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Check if predictions are logits; if so, convert them
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert predictions into sequences of token IDs
    if isinstance(predictions, np.ndarray):
        predictions = np.argmax(predictions, axis=-1)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Prepare references for BLEU (needs nested list format)
    decoded_labels_bleu = [[label] for label in decoded_labels]

    # Print example outputs to check differences
    print("Sample Prediction:", decoded_preds[0])
    print("Sample Label:", decoded_labels[0])

    # Compute ROUGE scores
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Compute BLEU scores
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels_bleu)

    # Calculate CrystalBLEU using Pre-loaded Shared N-grams
    tokenized_preds = [pred.split() for pred in decoded_preds]
    tokenized_refs = [[ref.split()] for ref in decoded_labels]

    # Calculate CrystalBLEU
    crystal_bleu_result = corpus_bleu(
        tokenized_refs, tokenized_preds, ignoring=trivially_shared_ngrams
    )

    # Combine results
    result = {k: round(v, 4) for k, v in rouge_result.items()}
    result['bleu'] = round(bleu_result['score'], 4)
    result['crystal_bleu'] = round(crystal_bleu_result, 4)

    return result

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results/py_code_t5_finetune_100", 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,  # Smaller batch size to avoid OOM
    gradient_accumulation_steps=8,
    warmup_steps=500,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=10,
    logging_steps=100,
    optim="adamw_torch",
    bf16=True,
    logging_strategy="steps",
    dataloader_num_workers=8,
    save_total_limit=3,
    save_strategy='steps',
    save_steps=50000,
    do_eval=True,
    evaluation_strategy="steps",
    eval_accumulation_steps=10,
    eval_steps=10000
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train and Save
trainer.train()
trainer.save_model("./results/py_code_t5_finetune_100")