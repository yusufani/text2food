import json
import pandas as pd
import os

# Load the data
with open('modified_clip_final.json') as f:
    data = json.load(f)

# Load the csv data
df = pd.read_csv('combined_csv.csv')

# Create a new directory
if not os.path.exists('output_txt_files'):
    os.makedirs('output_txt_files')

# Iterate over the keys (image ids) in json data
for key in data.keys():
    # Check if the image id exists in the csv data
    if key in df['image_name'].values:
        # Retrieve the 'matched_text' for the matching image id
        matched_text = df[df['image_name'] == key]['matched_text'].values[0]
        
        # Write the matched_text into a txt file named after the image id
        with open(f"output_txt_files/{key.split('.')[0]}.txt", 'w') as f:
            f.write(str(matched_text))
