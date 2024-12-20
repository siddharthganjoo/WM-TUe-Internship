import openai
import json
from tqdm import tqdm

# Set your API key
openai.api_key = ""

# System message to guide the model
content = "You are a code summarization expert. Generate a concise and short summary of the given Python code."

# Function to generate predictions
def get_response(prompt):
    try:
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=50,  # Limit the length of the summary
            temperature=0.0,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating response for prompt: {prompt}. Error: {e}")
        return "<NONE>"

# Main function to process input file and save output
def main():
    input_file = '/home/sganjoo/internship/code_sum/code_sum_processed.jsonl'  # Input JSONL file
    output_file = 'filter_pred.jsonl'  # Output JSONL file with predictions

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, desc="Processing Prompts"):
            try:
                # Load JSONL entry
                entry = json.loads(line.strip())
                code = entry.get("filtered_code", "")  # Use 'code' field as the input prompt

                # Skip entries without valid code
                if not code:
                    print("Skipped an entry with no code.")
                    continue

                # Generate predictions
                prediction = get_response(code)

                # Add prediction to the entry
                entry["prediction"] = prediction

                # Write the updated entry to the output file
                json.dump(entry, outfile)
                outfile.write('\n')
            except json.JSONDecodeError as json_error:
                print(f"Error decoding JSON line: {line.strip()}. Error: {json_error}")
            except Exception as e:
                print(f"Unexpected error processing line: {line.strip()}. Error: {e}")

    print(f"Processing completed. Predictions saved to: {output_file}")

if __name__ == "__main__":
    main()