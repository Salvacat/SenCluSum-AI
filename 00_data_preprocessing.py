""" This module merges the files from the dataset, elmiinates duplicates
and keeps onlyu columns of ineterest to finally output a csv file
"""

import pandas as pd

file_path1 = 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv'
file_path2 = 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'

# Load each file into a DataFrame
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Concatenate all DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Remove duplicates based on 'reviews.text' column
deduplicated_df = combined_df.drop_duplicates(subset='reviews.text')

# Select only necessary columns
columns_to_keep = ['name', 'brand', 'categories', 'primaryCategories', 
                   'reviews.date', 'reviews.title', 'reviews.text', 'reviews.rating']
filtered_df = combined_df[columns_to_keep]

# Drop duplicates
filtered_df = filtered_df.drop_duplicates()

# Convert text fields to lowercase and remove special characters
for col in ['name', 'categories', 'primaryCategories', 'reviews.title', 'reviews.text']:
    filtered_df[col] = filtered_df[col].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)

# Display the cleaned dataframe
print(filtered_df.head())
filtered_df.to_csv('filtered_data.csv', index=False)
