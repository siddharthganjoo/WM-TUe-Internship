import os
import torch
import json
import ast
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from collections import Counter
import evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load tokenizer and model
model_name_or_path = 'Salesforce/codet5p-220m'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Load CodeT5 model
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_or_path, 
    cache_dir='./models'
)

# Adjust token embeddings if new tokens are added
model.resize_token_embeddings(len(tokenizer))

# Load your dataset
train_dataset = load_dataset("json", data_files="/home/sganjoo/internship/dataset/python/train_processed.jsonl")["train"]
validation_dataset = load_dataset("json", data_files="/home/sganjoo/internship/dataset/python/validprocn.jsonl")["train"]
test_dataset = load_dataset("json", data_files="/home/sganjoo/internship/dataset/python/test_processed.jsonl")["train"]

# Preprocessing function for code summarization
def preprocess_function(example):
    input_text = example["code"]
    output_text = example["summary"]

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

    # Combine results
    result = {k: round(v, 4) for k, v in rouge_result.items()}
    result['bleu'] = round(bleu_result['score'], 4)

    return result

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results/py_code_t5_finetune_summarization_normal", 
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,  # Smaller batch size to avoid OOM
    gradient_accumulation_steps=4,
    warmup_steps=500,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=10,
    logging_steps=100,
    optim="adamw_torch",
    bf16=False,  # Disable bf16 if unsupported
    fp16=True,
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
trainer.save_model("./results/py_code_t5_finetune_summarization_normal")