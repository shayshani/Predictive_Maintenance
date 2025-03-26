import pandas as pd
import pdb

# Load the dataset
data = pd.read_csv('/home/shays/LIGHTBITS/UniDataSet.csv')

# Select only the numeric columns (excluding the first column containing dates)
numeric_data = data.iloc[:, 1:]  # Excluding the first column (dates) using iloc

# Save the first 60 elements (rows) into a list
first_60_elements = numeric_data.head(10).round(3)

# Print the list
print("First 60 rows of numeric data as a list:")
print(first_60_elements)