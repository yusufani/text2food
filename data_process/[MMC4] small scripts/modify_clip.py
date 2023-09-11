import json

# Load the data
with open('distilled_clip.json') as f:
    data = json.load(f)

# Remove directory path from keys
modified_data = {key.split('/')[-1]: value for key, value in data.items()}

# Count keys in the modified_data
key_count = len(modified_data)

print(f"The modified JSON file contains {key_count} keys.")

# Write the modified data to a new file
with open('modified_clip_final.json', 'w') as f:
    json.dump(modified_data, f, indent=4)
