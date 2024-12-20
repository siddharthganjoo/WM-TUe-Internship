import openai
import json
from tqdm import tqdm

# Set your API key
openai.api_key = ""

# System message to generate only Python code
content = "You are a Java expert. Generate complete and valid Java code starting with the provided method signature. Use the docstring to guide the implementation, ensure the function is robust (handles null inputs and null elements), and return only the complete code starting with the method signature."

# Function to generate code
def get_response(prompt):
    try:
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with "gpt-4o" if applicable and accessible
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating response for prompt: {prompt}. Error: {e}")
        return "<NONE>"

# Main function to process input file and save output
def main():
    input_file = '/home/sganjoo/internship/java/realjava.jsonl'  # Input file
    output_file = 'realjava_output.jsonl'  # Output file

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, desc="Processing Prompts"):
            entry = json.loads(line)  # Load each JSONL entry
            _id = entry.get('question_id', None)  # Extract the question ID
            prompt = entry.get('input', '')  # Extract the input field

            # Generate code
            generated_code = get_response(prompt)

            # Prepare output entry
            output_entry = {
                "_id": _id,
                "generate_results": [generated_code]
            }

            # Write to output file
            json.dump(output_entry, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    main()