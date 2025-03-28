import ast
import json

# Read the single-quoted string from a file
with open('single_quoted_output.txt', 'r') as f:
    single_quoted = f.read()

# Convert string to Python dictionary
data = ast.literal_eval(single_quoted)

# Save as proper JSON with double quotes
with open('diarization_output.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Conversion complete! Check diarization_output.json")