import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METADATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HAM10000_metadata.csv"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "splits"
)

os.makedirs(SPLIT_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================

RANDOM_STATE = 42

# =========================
# LOAD DATA
# =========================

print("\n========== CREATING LESION-LEVEL SPLITS ==========\n")

df = pd.read_csv(METADATA_PATH)

print(f"Total images: {len(df):,}")
print(f"Total unique lesions: {df['lesion_id'].nunique():,}")

# =========================
# CREATE ONE ROW PER LESION
# =========================
#
# Each lesion gets one diagnosis label.
# We split lesions, NOT individual images.
#
# This prevents the same lesion from appearing
# in multiple datasets.
# =========================

lesions = (
    df.groupby("lesion_id")
    .agg(dx=("dx", "first"))
    .reset_index()
)

print(f"Lesions available for splitting: {len(lesions):,}")

# =========================
# 70% TRAIN / 30% TEMP
# =========================

train_lesions, temp_lesions = train_test_split(
    lesions,
    test_size=0.30,
    stratify=lesions["dx"],
    random_state=RANDOM_STATE
)

# =========================
# 15% VALIDATION / 15% TEST
# =========================

val_lesions, test_lesions = train_test_split(
    temp_lesions,
    test_size=0.50,
    stratify=temp_lesions["dx"],
    random_state=RANDOM_STATE
)

print("\n========== LESION SPLIT ==========\n")

print(f"Train lesions:      {len(train_lesions):,}")
print(f"Validation lesions: {len(val_lesions):,}")
print(f"Test lesions:       {len(test_lesions):,}")

# =========================
# GET IMAGE ROWS
# =========================

train_df = df[df["lesion_id"].isin(train_lesions["lesion_id"])].copy()

val_df = df[df["lesion_id"].isin(val_lesions["lesion_id"])].copy()

test_df = df[df["lesion_id"].isin(test_lesions["lesion_id"])].copy()

# =========================
# VERIFY NO LEAKAGE
# =========================

train_ids = set(train_df["lesion_id"])
val_ids = set(val_df["lesion_id"])
test_ids = set(test_df["lesion_id"])

train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids

print("\n========== LEAKAGE CHECK ==========\n")

print(f"Train ∩ Validation: {len(train_val_overlap)}")
print(f"Train ∩ Test:       {len(train_test_overlap)}")
print(f"Validation ∩ Test:  {len(val_test_overlap)}")

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("\n✓ NO LESION LEAKAGE DETECTED")
else:
    raise RuntimeError("LEAKAGE DETECTED!")

# =========================
# DATASET DISTRIBUTION
# =========================

print("\n========== IMAGE DISTRIBUTION ==========\n")

print(f"Train images:      {len(train_df):,}")
print(f"Validation images: {len(val_df):,}")
print(f"Test images:       {len(test_df):,}")

print("\nTrain class distribution:")
print(train_df["dx"].value_counts())

print("\nValidation class distribution:")
print(val_df["dx"].value_counts())

print("\nTest class distribution:")
print(test_df["dx"].value_counts())

# =========================
# SAVE SPLITS
# =========================

train_path = os.path.join(SPLIT_DIR, "train.csv")
val_path = os.path.join(SPLIT_DIR, "val.csv")
test_path = os.path.join(SPLIT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n========== FILES SAVED ==========\n")

print(f"✓ {train_path}")
print(f"✓ {val_path}")
print(f"✓ {test_path}")

print("\n========== SPLIT COMPLETE ==========\n")