import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Paths to the input data and output file
input_file = '/home/sganjoo/internship/realpython100.jsonl' #change inputs
output_file = 'dsc_100.jsonl'

# Load your fine-tuned model
model_path = 'deepseek-ai/deepseek-coder-1.3b-base' #change model here
 # Hugging Face model path
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Add <GIST> to the tokenizer
if "<GIST>" not in tokenizer.vocab:   #added for <GIST>
    tokenizer.add_tokens(["<GIST>"])

model = AutoModelForCausalLM.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))
model.to('cuda' if torch.cuda.is_available() else 'cpu')

def generate_samples(prompt, num_samples=20):
    """Generate multiple samples for a given prompt."""
    generated_codes = []
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    for _ in range(num_samples):
        outputs = model.generate(**inputs, max_length=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_codes.append(code)
    
    return generated_codes

# Process each entry in the input file
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in tqdm(infile, desc="Processing entries"):
        data = json.loads(line.strip())
        
        # Use the `input` field directly as the prompt
        prompt = data['input']
        
        # Generate samples
        generated_samples = generate_samples(prompt, num_samples=20)
        
        # Format output for this entry
        output_data = {
            "_id": data["question_id"],
            "generate_results": generated_samples
        }
        
        # Write the output data as a JSON line
        outfile.write(json.dumps(output_data) + '\n')

print(f"Generated predictions saved to {output_file}")