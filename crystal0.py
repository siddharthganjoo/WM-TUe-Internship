import json

def construct_signature(func_name, original_string):
    # Extract the function signature from the original string up to the first newline
    signature_line = original_string.split("\n")[0]
    return signature_line

def replace_with_gist(docstring, gist_percentage=0):
    tokens = docstring.split()
    num_tokens = len(tokens)
    replace_count = max(1, num_tokens * gist_percentage // 100)
    if gist_percentage == 0:
        return docstring
    gist_tokens = tokens[:-replace_count] + ["<GIST>"]
    return " ".join(gist_tokens)

def process_jsonl_file(input_file, output_file, gist_percentage=0):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            docstring = data.get("docstring", "")
            original_string = data.get("original_string", "")
            
            # Extract the function signature from the original string
            signature = construct_signature(data.get("func_name", ""), original_string)
            
            # Apply gisting to the docstring
            modified_docstring = replace_with_gist(docstring, gist_percentage=gist_percentage)
            
            # Combine the signature and modified docstring
            modified_combined = f"{signature}\n\n{modified_docstring}"
            
            # Create a new JSON object with additional fields
            new_data = data.copy()
            new_data.update({
                "Signature": signature,
                "Gisted Docstring": modified_docstring,
                "Signature + Gisted Docstring": modified_combined
            })
            
            # Write the updated data to the output file
            outfile.write(json.dumps(new_data) + "\n")

# Example usage:
input_file = "/home/20231567/dataforgisting/dataset/python/valid.jsonl"  # Replace with your input file path
output_file = "pyvalid100.jsonl"  # Output file path
gist_percentage = 100  # Adjust as needed for 20% gisting

process_jsonl_file(input_file, output_file, gist_percentage)