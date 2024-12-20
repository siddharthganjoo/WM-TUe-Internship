# import random
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from datasets import load_dataset
# import evaluate

# # Load evaluation metrics
# rouge = evaluate.load("rouge")
# bleu = evaluate.load("sacrebleu")

# # Paths to the fine-tuned models
# model_paths = {
#     "normal_code": "/home/sganjoo/internship/code_sum/results/py_code_t5_finetune_summarization_base",  # Path to the model trained on normal code
#     "filtered_code": "/home/sganjoo/internship/code_sum/results/py_code_t5_finetune_summarization_ngram",  # Path to the model trained on filtered code
# }

# # Load tokenizers and models
# models = {}
# tokenizers = {}
# for key, path in model_paths.items():
#     tokenizers[key] = AutoTokenizer.from_pretrained(path)
#     models[key] = AutoModelForSeq2SeqLM.from_pretrained(path).eval()

# # Load the first 10,000 lines of the test dataset
# test_file_path = "/home/sganjoo/internship/dataset/python/test_processed.jsonl"
# test_dataset = load_dataset("json", data_files=test_file_path)["train"].select(range(230))

# # Select two random examples
# random_indices = random.sample(range(len(test_dataset)), 2)
# selected_examples = [test_dataset[i] for i in random_indices]

# # Function to generate a summary
# def generate_summary(model, tokenizer, code):
#     inputs = tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Evaluate models on the selected examples
# def evaluate_random_examples(selected_examples, models, tokenizers):
#     for model_name in models.keys():
#         print(f"\nEvaluating model: {model_name}")
        
#         for example in selected_examples:
#             code = example["code"]  # Original code
#             reference_summary = example["summary"]  # Original summary
            
#             # Generate summary
#             generated_summary = generate_summary(models[model_name], tokenizers[model_name], code)
            
#             # Compute ROUGE
#             rouge_result = rouge.compute(predictions=[generated_summary], references=[reference_summary], use_stemmer=True)
            
#             # Compute BLEU
#             bleu_result = bleu.compute(predictions=[generated_summary], references=[[reference_summary]])
            
#             # Print details
#             print("\nExample:")
#             print(f"Code:\n{code}")
#             print(f"Reference Summary:\n{reference_summary}")
#             print(f"Generated Summary:\n{generated_summary}")
#             print(f"ROUGE-1: {round(rouge_result['rouge1'], 4)}")
#             print(f"ROUGE-2: {round(rouge_result['rouge2'], 4)}")
#             print(f"ROUGE-L: {round(rouge_result['rougeL'], 4)}")
#             print(f"BLEU: {round(bleu_result['score'], 4)}")

# # Run evaluation on selected examples
# evaluate_random_examples(selected_examples, models, tokenizers)


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import evaluate
from tqdm import tqdm

# Load evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

# Paths to the fine-tuned models
model_paths = {
    # "normal_code": "/home/sganjoo/internship/code_sum/results/py_code_t5_finetune_summarization_normal",  # Path to the model trained on normal code
    "filtered_code": "/home/sganjoo/internship/code_sum/results/py_code_t5_finetune_summarization_filtered",  # Path to the model trained on filtered code
}

# Load tokenizers and models
models = {}
tokenizers = {}
for key, path in model_paths.items():
    tokenizers[key] = AutoTokenizer.from_pretrained(path)
    models[key] = AutoModelForSeq2SeqLM.from_pretrained(path).eval()

# Path to the test JSONL file
test_file_path = "/home/sganjoo/internship/dataset/python/test_processed.jsonl"

# Function to generate a summary
def generate_summary(model, tokenizer, code):
    inputs = tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate all entries in the JSONL file
def evaluate_all_lines(file_path, models, tokenizers):
    results = {key: {"rouge1": [], "rouge2": [], "rougeL": [], "bleu": []} for key in models.keys()}

    with open(file_path, 'r') as infile:
        for line in tqdm(infile, desc="Processing lines"):
            data = json.loads(line.strip())
            code = data["code"]
            reference_summary = data["summary"]

            for model_name in models.keys():
                generated_summary = generate_summary(models[model_name], tokenizers[model_name], code)

                # Compute ROUGE
                rouge_result = rouge.compute(predictions=[generated_summary], references=[reference_summary], use_stemmer=True)
                results[model_name]["rouge1"].append(rouge_result["rouge1"])
                results[model_name]["rouge2"].append(rouge_result["rouge2"])
                results[model_name]["rougeL"].append(rouge_result["rougeL"])

                # Compute BLEU
                bleu_result = bleu.compute(predictions=[generated_summary], references=[[reference_summary]])
                results[model_name]["bleu"].append(bleu_result["score"])

    # Calculate average scores
    avg_results = {}
    for model_name, scores in results.items():
        avg_results[model_name] = {metric: sum(values) / len(values) for metric, values in scores.items()}

    return avg_results

# Run evaluation
avg_scores = evaluate_all_lines(test_file_path, models, tokenizers)

# Print results
for model_name, scores in avg_scores.items():
    print(f"\nAverage Scores for {model_name}:")
    for metric, score in scores.items():
        print(f"{metric}: {round(score, 4)}")