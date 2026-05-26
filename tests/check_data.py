from pathlib import Path

import pandas as pd

# Safely resolve the path relative to the current script's directory
file_path = Path(__file__).parent / "PGM_TestData" / "input" / "active_power_profile.parquet"

df = pd.read_parquet(file_path)

print("--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATAFRAME INFO ---")
df.info()

print("\n--- INDEX ---")
print(df.index.name)
print("\n--- COLUMNS ---")
print(df.columns)

active_profiles = pd.DataFrame(
    {1: [100.0, 200.0], 2: [100.0, 220.0]}, index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:15"])
)

print(active_profiles.head())


active_profiles = pd.DataFrame({"node": [1, 2, 3], "active_load": [100, 200, 300]})

print(active_profiles.head())
