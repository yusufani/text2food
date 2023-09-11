import pandas as pd
import glob

# # Get a list of all csv file paths
# csv_files = glob.glob('/Users/arda/Downloads/all_csv/*.csv')

# # Create a list to hold dataframes
# dfs = []

# # Loop over the csv files and for each, read the content into a dataframe and add it to the list
# for filename in csv_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     dfs.append(df)

# # Concatenate all dataframes
# combined_csv = pd.concat(dfs, axis=0, ignore_index=True)

# # Write the concatenated data to a csv file
# combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')



import pandas as pd

# Read the CSV file
df = pd.read_csv('combined_csv.csv')

# Print total number of entries
print(f"Total number of entries: {len(df)}")

# Check for duplicate image_name entries
duplicates = df[df['image_name'].duplicated(keep=False)]

# If there are duplicates, print them
if not duplicates.empty:
    print("\nFound the following duplicate entries in the 'image_name' column:")
    print(duplicates.sort_values('image_name').head())
    print(len(duplicates))
    print(duplicates)
else:
    print("\nNo duplicate entries found in the 'image_name' column.")

