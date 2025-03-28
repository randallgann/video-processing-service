import os
import tiktoken

def estimate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Returns the number of tokens for a given text string
    based on the specified OpenAI model's tokenizer.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def calculate_total_tokens_in_directory(directory: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Reads all .json files in the given directory and sums their token counts.
    """
    total_tokens = 0
    
    for filename in os.listdir(directory):
        # Only process .json files
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            # Read the contents of the file
            with open(file_path, 'r', encoding='utf-8') as f:
                file_contents = f.read()
            # Estimate the tokens for this file
            total_tokens += estimate_tokens(file_contents, model_name)
    
    return total_tokens

if __name__ == "__main__":
    directory_path = "../transcript_outputs_json"
    # Adjust the model name if needed, e.g., "gpt-4"
    total = calculate_total_tokens_in_directory(directory_path, model_name="gpt-4")
    print(f"Total estimated tokens for all JSON files: {total}")
