import json

# Initialize counters
total_lines = 0
pass_1_count = 0
pass_3_count = 0
pass_5_count = 0
pass_10_count = 0
pass_20_count = 0

# Open and read each line in the JSONL file
with open('/home/20231567/dataforgisting/py_100.jsonl_out.jsonl', 'r') as file:
    for line in file:
        # Skip empty or whitespace-only lines
        if not line.strip():
            continue
            
        try:
            # Parse JSON line
            data = json.loads(line)
            
            # Check Pass@1, Pass@3, Pass@5, Pass@10, and Pass@20
            if any(result["is_pass"] for result in data.get("generate_results", [])[:1]):
                pass_1_count += 1
            if any(result["is_pass"] for result in data.get("generate_results", [])[:3]):
                pass_3_count += 1
            if any(result["is_pass"] for result in data.get("generate_results", [])[:5]):
                pass_5_count += 1
            if any(result["is_pass"] for result in data.get("generate_results", [])[:10]):
                pass_10_count += 1
            if any(result["is_pass"] for result in data.get("generate_results", [])[:20]):
                pass_20_count += 1

            total_lines += 1
            
        except json.JSONDecodeError:
            print("Warning: Skipping a line due to JSONDecodeError")

# Calculate overall pass ratios
pass_1_ratio = pass_1_count / total_lines if total_lines > 0 else 0
pass_3_ratio = pass_3_count / total_lines if total_lines > 0 else 0
pass_5_ratio = pass_5_count / total_lines if total_lines > 0 else 0
pass_10_ratio = pass_10_count / total_lines if total_lines > 0 else 0
pass_20_ratio = pass_20_count / total_lines if total_lines > 0 else 0

# Report results
print(f"Total lines: {total_lines}")
print(f"Pass@1 count: {pass_1_count}, Ratio: {pass_1_ratio:.2f}")
print(f"Pass@3 count: {pass_3_count}, Ratio: {pass_3_ratio:.2f}")
print(f"Pass@5 count: {pass_5_count}, Ratio: {pass_5_ratio:.2f}")
print(f"Pass@10 count: {pass_10_count}, Ratio: {pass_10_ratio:.2f}")
print(f"Pass@20 count: {pass_20_count}, Ratio: {pass_20_ratio:.2f}")