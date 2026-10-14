import os
import pandas as pd

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METADATA_PATH = os.path.join(
    BASE_DIR, "data", "HAM10000_metadata.csv"
)

IMAGE_DIRS = [
    os.path.join(BASE_DIR, "data", "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "data", "HAM10000_images_part_2")
]

# =========================
# LOAD METADATA
# =========================
print("\n========== HAM10000 DATASET AUDIT ==========\n")

df = pd.read_csv(METADATA_PATH)

print(f"Metadata rows: {len(df):,}")
print(f"Unique image IDs: {df['image_id'].nunique():,}")
print(f"Unique lesion IDs: {df['lesion_id'].nunique():,}")

# =========================
# IMAGE FILE CHECK
# =========================
image_files = set()

for folder in IMAGE_DIRS:
    if not os.path.exists(folder):
        print(f"WARNING: Folder not found: {folder}")
        continue

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.add(os.path.splitext(file)[0])

print(f"Actual image files found: {len(image_files):,}")

metadata_images = set(df["image_id"])

missing_images = metadata_images - image_files
extra_images = image_files - metadata_images

print(f"Missing image files: {len(missing_images):,}")
print(f"Extra image files: {len(extra_images):,}")

# =========================
# DUPLICATES
# =========================
duplicate_images = df["image_id"].duplicated().sum()

lesion_counts = df["lesion_id"].value_counts()

multi_image_lesions = (lesion_counts > 1).sum()
max_images_per_lesion = lesion_counts.max()

print("\n========== DUPLICATE / LESION CHECK ==========")

print(f"Duplicate image IDs: {duplicate_images:,}")
print(f"Lesions with multiple images: {multi_image_lesions:,}")
print(f"Maximum images for one lesion: {max_images_per_lesion}")

# =========================
# CLASS DISTRIBUTION
# =========================
print("\n========== DIAGNOSIS DISTRIBUTION ==========\n")

class_counts = df["dx"].value_counts()

print(class_counts)

print("\nClass percentages:")
print((class_counts / len(df) * 100).round(2))

# =========================
# MISSING VALUES
# =========================
print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())

# =========================
# METADATA SUMMARY
# =========================
print("\n========== METADATA SUMMARY ==========\n")

print("Sex:")
print(df["sex"].value_counts(dropna=False))

print("\nLocalization:")
print(df["localization"].value_counts())

print("\nDiagnosis type:")
print(df["dx_type"].value_counts())

print("\nAge statistics:")
print(df["age"].describe())

# =========================
# LESION DISTRIBUTION
# =========================
print("\n========== LESION IMAGE COUNTS ==========\n")

print(lesion_counts.describe())

# =========================
# FINAL STATUS
# =========================
print("\n========== FINAL STATUS ==========\n")

if len(df) == 10015:
    print("✓ Metadata contains expected 10,015 records.")
else:
    print("⚠ Metadata record count differs from 10,015.")

if len(image_files) == 10015:
    print("✓ Found expected 10,015 image files.")
else:
    print("⚠ Image count differs from 10,015.")

if duplicate_images == 0:
    print("✓ No duplicate image IDs.")
else:
    print("⚠ Duplicate image IDs detected.")

if len(missing_images) == 0:
    print("✓ Every metadata image has a corresponding image file.")
else:
    print("⚠ Some metadata images are missing.")

print("\nIMPORTANT:")
print("Train/validation/test splitting MUST be performed at the lesion_id level")
print("to prevent the same lesion from appearing in both training and testing.\n")

print("========== AUDIT COMPLETE ==========\n")