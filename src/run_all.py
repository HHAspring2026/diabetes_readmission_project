import subprocess
from data_cleaning import clean_data

# Step 1: clean the raw dataset
clean_data("data/diabetic_data.csv")

# Step 2: run the analysis script
subprocess.run(["python", "src/analysis.py"], check=True)

print("Full pipeline complete.")