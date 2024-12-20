import json
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# Ensure NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def calculate_bleu(reference, prediction):
    """
    Calculate BLEU score for a single pair of reference and prediction.
    """
    reference_tokens = [reference.split()]
    prediction_tokens = prediction.split()
    smooth_fn = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smooth_fn)

def calculate_meteor(reference, prediction):
    """
    Calculate METEOR score for a single pair of reference and prediction.
    """
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    return meteor_score([reference_tokens], prediction_tokens)

def main():
    input_file = "/home/sganjoo/internship/code_sum/llamacode_normal.jsonl"  # Replace with your input JSONL file
    bleu_scores = []
    meteor_scores = []

    with open(input_file, 'r') as infile:
        for line in tqdm(infile, desc="Processing lines"):
            entry = json.loads(line.strip())
            reference = entry.get("summary", "")
            prediction = entry.get("generated_summary", "")

            # Skip if either reference or prediction is missing
            if not reference or not prediction:
                continue

            # Calculate BLEU and METEOR scores
            bleu = calculate_bleu(reference, prediction)
            meteor = calculate_meteor(reference, prediction)

            # Append scores for averaging later
            bleu_scores.append(bleu)
            meteor_scores.append(meteor)

    # Calculate average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average METEOR Score: {avg_meteor:.4f}")

if __name__ == "__main__":
    main()