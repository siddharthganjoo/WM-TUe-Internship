import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import re
import evaluate
import numpy as np
from trl import SFTTrainer

# Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the tokenizer and add <GIST> token if not present
model_name_or_path = "google-t5/t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if "<GIST>" not in tokenizer.get_vocab():
    tokenizer.add_tokens(["<GIST>"])

# # Load the model with quantization settings
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     quantization_config=bnb_config,
#     device_map="auto",
#     cache_dir='./models'
# )
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir='./models'
)

# model.gradient_checkpointing_enable()
model.resize_token_embeddings(len(tokenizer))

# Apply LORA on all linear layers
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)

names = []
for name in linear_layer_names:
    names.append(name)
target_modules = list(set(names))

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, #SEQ_2_SEQ_LM
    target_modules=target_modules,
    inference_mode=False, 
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1
)

# Enable gradient computation for LoRA parameters
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load your custom dataset
train_dataset = load_dataset("json", data_files="/home/sganjoo/internship/train/train.jsonl")["train"]
validation_dataset = load_dataset("json", data_files="/home/sganjoo/internship/valid/valid.jsonl")["train"]
test_dataset = load_dataset("json", data_files="/home/sganjoo/internship/test/test.jsonl")["train"]

# Preprocessing function
def preprocess_function(example):
    input_text = example["Modified Natural Language + Signature"]
    output_text = example["code"]

    # Tokenize inputs and outputs
    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=256)
    outputs = tokenizer(output_text, padding="max_length", truncation=True, max_length=256)

    # Set up labels for training; mask padding tokens
    inputs["labels"] = outputs["input_ids"]
    inputs["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in inputs["labels"]]
    
    return inputs

# Apply the preprocessing
processed_train_dataset = train_dataset.map(preprocess_function, batched=True)
processed_validation_dataset = validation_dataset.map(preprocess_function, batched=True)
processed_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define evaluation metrics
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE Score
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # BLEU Score
    decoded_labels = [[label.split()] for label in decoded_labels]
    decoded_preds = [pred.split() for pred in decoded_preds]
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Combine Results
    result = {
        "rouge": rouge_result,
        "bleu": round(bleu_result["bleu"], 4),
        "gen_len": np.mean([len(pred.split()) for pred in decoded_preds])
    }
    
    return result

# Training arguments
training_args = TrainingArguments(
    output_dir="./gemma_finetuned_t5",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,  # Smaller batch size for evaluation
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    save_steps=10000,
    fp16=True,  # Use mixed precision
)


# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_validation_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    compute_metrics=compute_metrics
)

# Train the model
trainer.train(resume_from_checkpoint=False)
# Clear CUDA cache after training
torch.cuda.empty_cache()

predictions, labels = manual_evaluation(model, processed_validation_dataset, tokenizer, batch_size=1)
print("Manual evaluation completed.")

# Save the fine-tuned model
trainer.save_model("./finetuned_t5")
# Clear CUDA cache after training
torch.cuda.empty_cache()
# Evaluate the model on the test set
eval_results = trainer.evaluate(processed_test_dataset)
print(eval_results)