import json

def calculate_average_compression_factor(jsonl_file):
    """
    Calculate the average compression factor from a JSONL file.

    Args:
        jsonl_file (str): Path to the input JSONL file.

    Returns:
        float: Average compression factor.
    """
    total_factor = 0
    total_lines = 0

    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if "factor" in data:  # Check if the "factor" field exists
                total_factor += data["factor"]
                total_lines += 1

    if total_lines == 0:
        return 0  # Avoid division by zero if no lines with "factor" field

    return total_factor / total_lines


# Input JSONL file path
jsonl_file = "/home/sganjoo/internship/dataset/python/valid_processed.jsonl"  # Replace with the actual file path

# Calculate average compression factor
average_factor = calculate_average_compression_factor(jsonl_file)
print(f"Average Compression Factor: {average_factor:.2f}")

