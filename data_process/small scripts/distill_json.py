import json

# Load the data
with open('clip_data_all.json') as f:
    data = json.load(f)

# Set threshold
threshold = 0.15

# Filter data
filtered_data = {}
for key, value in data.items():
    if any(val > threshold for val in value.values()):
        filtered_data[key] = value

# Write the filtered data to a new file
with open('distilled_clip.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(len(filtered_data))
