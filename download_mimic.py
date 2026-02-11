import os
import kagglehub
import pandas as pd

DATASET = "simhadrisadaram/mimic-cxr-dataset"

# Download dataset
root = kagglehub.dataset_download(DATASET)
print("Downloaded to:", root)

# List files
files = []
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        files.append(os.path.relpath(os.path.join(dirpath, fn), root))
files = sorted(files)

csvs = [f for f in files if f.lower().endswith(".csv")]
print("CSV files found:", csvs)

# Get train/test csv paths
train_candidates = [f for f in csvs if f.lower().endswith("train.csv")]
test_candidates  = [f for f in csvs if f.lower().endswith("validate.csv")]

if not train_candidates or not test_candidates:
    raise FileNotFoundError(
        f"Train/test files not found"
    )

train_path = os.path.join(root, train_candidates[0])
test_path  = os.path.join(root, test_candidates[0])

print("Using train:", train_candidates[0])
print("Using validate :", test_candidates[0])

# Load csv
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\nTrain head:\n", train_df.head())
print("\nTest head:\n", test_df.head())
