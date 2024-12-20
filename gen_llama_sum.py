import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load the CodeLLaMA model and tokenizer
model_name = "codellama/CodeLlama-7b-hf"  # Replace with your model's name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to generate a summary for the code
def generate_summary(code_snippet):
    try:
        # Format the input for the model
        input_prompt = f"Summarize the following Python function in one sentence:\n\n{code_snippet}\n\nSummary:"
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Allow full processing of long input code
        )
        
        # Generate the summary
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,  # Limit the output to a very short summary
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the summary after "Summary:" for clarity
        if "Summary:" in summary:
            summary = summary.split("Summary:")[1].strip()
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "<ERROR>"

# Main function to process the input file and save the output
def main():
    input_file = "/home/sganjoo/internship/code_sum/code_sum_processed.jsonl"  # Replace with your input file path
    output_file = "llamacode_normal.jsonl"  # Replace with your desired output file path

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in tqdm(infile, desc="Processing lines"):
            entry = json.loads(line.strip())
            code = entry.get("code", "")
            original_summary = entry.get("summary", "")

            # Skip entries without valid code
            if not code:
                print("Skipped an entry with no code.")
                continue

            # Generate the new summary
            generated_summary = generate_summary(code)

            # Create a new entry with required fields
            output_entry = {
                "code": code,
                "summary": original_summary,
                "generated_summary": generated_summary
            }

            # Write the updated entry to the output file
            json.dump(output_entry, outfile)
            outfile.write("\n")

    print(f"Processing completed. Summaries saved to: {output_file}")

if __name__ == "__main__":
    main()