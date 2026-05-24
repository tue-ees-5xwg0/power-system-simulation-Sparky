import pandas as pd

# Update this path if you are running it from a different folder
file_path = "tests/PGM_TestData/input/active_power_profile.parquet"

# Load the file
df = pd.read_parquet(file_path)

# 1. Look at the actual table layout
print("--- FIRST 5 ROWS ---")
print(df.head())

# 2. Look at the exact data types and structure
print("\n--- DATAFRAME INFO ---")
print(df.info())

# 3. Check what the index and columns are explicitly named
print("\n--- INDEX ---")
print(df.index.name)
print("\n--- COLUMNS ---")
print(df.columns)
